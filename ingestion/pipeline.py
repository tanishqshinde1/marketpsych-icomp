"""End-to-end compliance-aware ingestion pipeline."""

from __future__ import annotations

import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ingestion.adapters import (
    AdapterConfig,
    InvestorRelationsAdapter,
    RssSitemapAdapter,
    SecEdgarAdapter,
    TranscriptCandidateAdapter,
    SerpApiGoogleSearchAdapter,
)
from ingestion.http import RateLimiter, RobotsPolicy, fetch_url, parse_html
from ingestion.query_generation import QueryPlan, generate_queries
from ingestion.relevance import score_relevance
from ingestion.schema import INGESTION_COLUMNS, IngestedItem, IngestionError, utc_now_iso
from ingestion.utils import (
    RESTRICTED_DOMAINS,
    canonicalize_url,
    classify_source_type,
    content_hash,
    domain_from_url,
    infer_company_entity,
    normalize_title,
    stable_doc_id,
)


@dataclass
class IngestionConfig:
    output_dir: str = "ingestion_outputs"
    start_date: str = "2023-01-01"
    end_date: str | None = None
    refresh_mode: str = "monthly"
    relevance_threshold: float = 0.35
    minimum_text_chars: int = 500
    max_queries: int = 80
    max_results_per_query: int = 10
    max_queries_per_run: int = 30
    max_pages_per_query: int = 2
    timeout_seconds: int = 10
    extraction_timeout_seconds: int = 10
    max_workers: int = 5
    max_candidates_to_process: int | None = None
    skip_extraction: bool = False
    rate_limit_seconds: float = 1.0
    user_agent: str = "HBMNarrativeResearchBot/0.1 compliance-aware research prototype; contact=research@example.com"
    enabled_adapters: list[str] = field(default_factory=lambda: ["serpapi_google", "rss_sitemap", "investor_relations", "sec_edgar", "transcript_candidate"])
    feed_urls: list[str] = field(default_factory=list)
    sitemap_urls: list[str] = field(default_factory=list)
    search_results_file: str | None = None
    serpapi_api_key: str | None = None
    verbose: bool = False


class IngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = RateLimiter(config.rate_limit_seconds)
        self.robots = RobotsPolicy(config.user_agent, timeout=config.timeout_seconds)

    def run(self) -> dict[str, Path]:
        query_plan = QueryPlan(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            max_queries=self.config.max_queries,
        )
        queries = generate_queries(query_plan)
        adapter_config = AdapterConfig(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            max_results_per_query=self.config.max_results_per_query,
            max_queries_per_run=min(self.config.max_queries, self.config.max_queries_per_run),
            max_pages_per_query=self.config.max_pages_per_query,
            user_agent=self.config.user_agent,
            rate_limit_seconds=self.config.rate_limit_seconds,
            feed_urls=self.config.feed_urls,
            sitemap_urls=self.config.sitemap_urls,
            search_results_file=self.config.search_results_file,
            serpapi_api_key=self.config.serpapi_api_key,
            verbose=self.config.verbose,
        )

        candidates: list[IngestedItem] = []
        errors: list[IngestionError] = []
        for adapter in self._adapters(adapter_config):
            try:
                if self.config.verbose:
                    print(f"[ingest] running adapter={adapter.name}")
                items, adapter_errors = adapter.discover(queries)
                if self.config.verbose:
                    print(f"[ingest] adapter={adapter.name} candidates={len(items)} errors={len(adapter_errors)}")
                candidates.extend(items)
                errors.extend(adapter_errors)
            except Exception as exc:
                errors.append(IngestionError(utc_now_iso(), adapter.name, "", "", "parse_failed", str(exc)))

        processed, processing_errors = self._process_candidates(candidates)
        errors.extend(processing_errors)
        raw, accepted, rejected = self._split_outputs(processed)
        diagnostics = self._build_diagnostics(raw, accepted, rejected)
        paths = self._write_outputs(raw, accepted, rejected, errors, diagnostics)
        if self.config.verbose:
            self._print_verbose_summary(raw, accepted, rejected)
        return paths

    def _adapters(self, adapter_config: AdapterConfig):
        registry = {
            "serpapi_google": SerpApiGoogleSearchAdapter,
            "rss_sitemap": RssSitemapAdapter,
            "investor_relations": InvestorRelationsAdapter,
            "sec_edgar": SecEdgarAdapter,
            "transcript_candidate": TranscriptCandidateAdapter,
        }
        for name in self.config.enabled_adapters:
            adapter_cls = registry.get(name)
            if adapter_cls:
                yield adapter_cls(adapter_config, self.rate_limiter)

    def _process_candidates(self, candidates: list[IngestedItem]) -> tuple[list[IngestedItem], list[IngestionError]]:
        processed: list[IngestedItem] = []
        errors: list[IngestionError] = []
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        seen_hashes: set[str] = set()
        version_counts: dict[str, int] = {}

        candidates_to_process: list[IngestedItem] = []
        for candidate in candidates:
            try:
                item = self._normalize_candidate(candidate)
                title_key = f"{item.domain}|{normalize_title(item.title)}"
                if item.canonical_url in seen_urls or (item.title and title_key in seen_titles):
                    item.extraction_status = "duplicate"
                    item.text_permission_status = "metadata_only"
                    item.metadata_quality_notes = _append_note(item.metadata_quality_notes, "duplicate canonical URL or title")
                    processed.append(item)
                    continue
                seen_urls.add(item.canonical_url)
                if item.title:
                    seen_titles.add(title_key)
                candidates_to_process.append(item)
            except Exception as exc:
                errors.append(IngestionError(utc_now_iso(), candidate.discovery_method, candidate.query, candidate.url, "parse_failed", str(exc)))

        if self.config.max_candidates_to_process is not None:
            candidates_to_process = candidates_to_process[: self.config.max_candidates_to_process]

        if self.config.skip_extraction:
            for item in candidates_to_process:
                item.access_status = "unknown"
                item.extraction_status = "parse_failed"
                item.text_permission_status = "unknown"
                item.source_type, item.source_type_quality, source_reason = classify_source_type(
                    item.canonical_url,
                    item.domain,
                    item.title,
                    "",
                )
                item.relevance_score, item.relevance_reason = score_relevance(item.title, "", item.source_type, item.query)
                item.metadata_quality_notes = _append_note(item.metadata_quality_notes, "skip_extraction enabled; discovery metadata only")
                self._finalize_metadata_quality(item, source_reason)
                processed.append(item)
            return processed, errors

        total = len(candidates_to_process)
        if self.config.verbose:
            print(f"[extract] starting candidates={total} workers={self.config.max_workers} timeout={self.config.extraction_timeout_seconds}s")

        executor = ThreadPoolExecutor(max_workers=max(1, self.config.max_workers))
        try:
            future_to_item = {
                executor.submit(self._process_single_candidate, deepcopy(item)): item for item in candidates_to_process
            }
            submitted_at = {future: time.monotonic() for future in future_to_item}
            pending = set(future_to_item)
            completed_count = 0
            last_progress = 0

            while pending:
                done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                timed_out = {
                    future
                    for future in pending
                    if time.monotonic() - submitted_at[future] >= self.config.extraction_timeout_seconds
                }
                done |= timed_out
                if not done:
                    continue

                for future in done:
                    if future not in pending:
                        continue
                    pending.remove(future)
                    base_item = future_to_item[future]
                    if future in timed_out and not future.done():
                        future.cancel()
                        item, item_errors = self._timeout_result(base_item)
                    else:
                        try:
                            item, item_errors = future.result(timeout=self.config.extraction_timeout_seconds)
                        except Exception as exc:
                            item = base_item
                            item.access_status = "unknown"
                            item.extraction_status = "parse_failed"
                            item.text_permission_status = "unknown"
                            item_errors = [
                                IngestionError(utc_now_iso(), item.discovery_method, item.query, item.url, "parse_failed", str(exc))
                            ]

                    if item.content_hash and item.content_hash in seen_hashes:
                        item.extraction_status = "duplicate"
                        item.text_permission_status = "metadata_only"
                        item.text = ""
                        item.metadata_quality_notes = _append_note(item.metadata_quality_notes, "duplicate content hash")
                    if item.content_hash:
                        seen_hashes.add(item.content_hash)
                    version_counts[item.canonical_url] = version_counts.get(item.canonical_url, 0) + 1
                    item.version_id = f"v{version_counts[item.canonical_url]}"
                    item.doc_id = stable_doc_id(item.canonical_url, item.title)
                    processed.append(item)
                    errors.extend(item_errors)
                    completed_count += 1

                    should_log = self.config.verbose and (
                        completed_count - last_progress >= 5 or completed_count == total
                    )
                    if should_log:
                        raw, accepted, rejected = self._split_outputs(processed)
                        print(
                            f"[extract] {completed_count}/{total} accepted={len(accepted)} "
                            f"rejected={len(rejected)} errors={len(errors)} domain={item.domain}"
                        )
                        last_progress = completed_count
                    if completed_count % 25 == 0:
                        self._flush_partial(processed, errors)

        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        self._flush_partial(processed, errors)
        return processed, errors

    def _process_single_candidate(self, item: IngestedItem) -> tuple[IngestedItem, list[IngestionError]]:
        errors: list[IngestionError] = []
        if item.domain in RESTRICTED_DOMAINS:
            item.access_status = "paywalled"
            item.robots_allowed = None
            item.extraction_status = "restricted"
            item.text_permission_status = "metadata_only"
            item.source_type, item.source_type_quality, source_reason = classify_source_type(
                item.canonical_url,
                item.domain,
                item.title,
                "",
            )
            item.company_entity = item.company_entity or infer_company_entity(item.title, "", item.canonical_url, item.query)
            item.relevance_score, item.relevance_reason = score_relevance(item.title, "", item.source_type, item.query)
            self._finalize_metadata_quality(item, source_reason)
            errors.append(IngestionError(utc_now_iso(), item.discovery_method, item.query, item.url, "paywalled", "Known restricted domain stored as metadata only."))
            return item, errors

        robots_allowed = self.robots.allowed(item.canonical_url)
        item.robots_allowed = robots_allowed
        if robots_allowed is False:
            item.access_status = "blocked"
            item.extraction_status = "restricted"
            item.text_permission_status = "metadata_only"
            item.source_type, item.source_type_quality, source_reason = classify_source_type(
                item.canonical_url,
                item.domain,
                item.title,
                "",
            )
            self._finalize_metadata_quality(item, source_reason)
            errors.append(IngestionError(utc_now_iso(), item.discovery_method, item.query, item.url, "robots_blocked", "robots.txt disallows fetching this URL."))
            return item, errors

        fetch = fetch_url(
            item.canonical_url,
            user_agent=self.config.user_agent,
            timeout=self.config.timeout_seconds,
            rate_limiter=self.rate_limiter,
        )
        if fetch.error and "timeout" in fetch.error.lower():
            item.access_status = "unknown"
            item.extraction_status = "timeout"
            item.text_permission_status = "unknown"
            errors.append(IngestionError(utc_now_iso(), item.discovery_method, item.query, item.url, "timeout", fetch.error))
        else:
            parsed, access_status, extraction_status = parse_html(fetch)
            item.access_status = access_status
            item.extraction_status = extraction_status
            if parsed.get("canonical_url"):
                item.canonical_url = canonicalize_url(str(parsed["canonical_url"]))
                item.domain = domain_from_url(item.canonical_url)
            if parsed.get("title"):
                item.title = str(parsed["title"])
            if parsed.get("timestamp_published") and not item.timestamp_published:
                item.timestamp_published = str(parsed["timestamp_published"])
                item.timestamp_quality = str(parsed.get("timestamp_quality") or "exact")
            if access_status == "accessible" and extraction_status == "success":
                item.text = str(parsed.get("text") or "")
                item.text_permission_status = "full_text_allowed"
            else:
                item.text = ""
                item.text_permission_status = "metadata_only"
                if access_status in {"paywalled", "blocked"}:
                    errors.append(IngestionError(utc_now_iso(), item.discovery_method, item.query, item.url, access_status, "Restricted content stored as metadata only."))

        item.source_type, item.source_type_quality, source_reason = classify_source_type(
            item.canonical_url,
            item.domain,
            item.title,
            item.text,
        )
        item.company_entity = item.company_entity or infer_company_entity(item.title, item.text, item.canonical_url, item.query)
        item.content_hash = content_hash(item.text)
        score, reason = score_relevance(item.title, item.text, item.source_type, item.query)
        item.relevance_score = score
        item.relevance_reason = reason
        if item.extraction_status == "success" and score < self.config.relevance_threshold:
            item.extraction_status = "irrelevant"
        if not item.timestamp_published and item.extraction_status == "success":
            errors.append(IngestionError(utc_now_iso(), item.discovery_method, item.query, item.url, "missing_timestamp", "Published timestamp was not found."))
        if item.source_type == "general_web_unknown":
            errors.append(IngestionError(utc_now_iso(), item.discovery_method, item.query, item.url, "unknown_source_type", "Could not classify source type beyond general_web_unknown."))
        self._finalize_metadata_quality(item, source_reason)
        return item, errors

    def _timeout_result(self, item: IngestedItem) -> tuple[IngestedItem, list[IngestionError]]:
        item.access_status = "unknown"
        item.extraction_status = "timeout"
        item.text_permission_status = "unknown"
        item.text = ""
        item.source_type, item.source_type_quality, source_reason = classify_source_type(
            item.canonical_url,
            item.domain,
            item.title,
            "",
        )
        self._finalize_metadata_quality(item, source_reason)
        item.metadata_quality_notes = _append_note(
            item.metadata_quality_notes,
            f"full candidate processing exceeded {self.config.extraction_timeout_seconds}s",
        )
        return item, [
            IngestionError(
                utc_now_iso(),
                item.discovery_method,
                item.query,
                item.url,
                "timeout",
                f"Full candidate processing exceeded {self.config.extraction_timeout_seconds}s.",
            )
        ]

    def _normalize_candidate(self, candidate: IngestedItem) -> IngestedItem:
        candidate.canonical_url = canonicalize_url(candidate.canonical_url or candidate.url)
        candidate.domain = domain_from_url(candidate.canonical_url)
        candidate.doc_id = candidate.doc_id or stable_doc_id(candidate.canonical_url, candidate.title)
        candidate.timestamp_discovered = candidate.timestamp_discovered or utc_now_iso()
        candidate.timestamp_modeling_fallback = candidate.timestamp_modeling_fallback or candidate.timestamp_discovered
        if candidate.timestamp_published:
            candidate.timestamp_quality = candidate.timestamp_quality if candidate.timestamp_quality != "missing" else "exact"
        return candidate

    def _finalize_metadata_quality(self, item: IngestedItem, source_reason: str) -> None:
        notes = []
        if not item.timestamp_published:
            item.timestamp_quality = "missing"
            item.timestamp_modeling_fallback = item.timestamp_discovered
            notes.append("timestamp_published missing; timestamp_modeling_fallback uses discovery time")
        elif item.timestamp_quality == "missing":
            item.timestamp_quality = "exact"
        if item.source_type == "general_web_unknown":
            item.source_type_quality = "unknown"
            notes.append("source type unresolved; assigned general_web_unknown")
        if not item.company_entity:
            notes.append("company_entity missing")
        if source_reason:
            notes.append(f"source classification: {source_reason}")
        for note in notes:
            item.metadata_quality_notes = _append_note(item.metadata_quality_notes, note)

    def _split_outputs(self, processed: list[IngestedItem]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raw = pd.DataFrame([item.to_record() for item in processed], columns=INGESTION_COLUMNS)
        if raw.empty:
            raw = pd.DataFrame(columns=INGESTION_COLUMNS)
        text_len = raw["text"].fillna("").astype(str).str.len() if "text" in raw.columns else pd.Series([], dtype=int)
        accepted_mask = (
            (raw["extraction_status"] == "success")
            & (~raw["access_status"].isin(["blocked", "paywalled"]))
            & (raw["text_permission_status"] == "full_text_allowed")
            & (text_len >= self.config.minimum_text_chars)
            & (raw["relevance_score"] >= self.config.relevance_threshold)
        )
        raw["rejection_reason"] = ""
        raw.loc[text_len == 0, "rejection_reason"] = "empty_text"
        raw.loc[(text_len > 0) & (text_len < self.config.minimum_text_chars), "rejection_reason"] = "text_too_short"
        raw.loc[raw["relevance_score"] < self.config.relevance_threshold, "rejection_reason"] = "low_relevance"
        raw.loc[raw["text_permission_status"] == "metadata_only", "rejection_reason"] = "metadata_only"
        raw.loc[raw["extraction_status"].isin(["restricted", "parse_failed", "timeout", "duplicate"]), "rejection_reason"] = raw["extraction_status"]
        raw.loc[raw["access_status"].isin(["blocked", "paywalled"]), "rejection_reason"] = raw["access_status"]
        raw.loc[accepted_mask, "rejection_reason"] = ""
        accepted = raw.loc[accepted_mask].copy()
        rejected = raw.loc[~accepted_mask].copy()
        return raw, accepted, rejected

    def _build_diagnostics(self, raw: pd.DataFrame, accepted: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        def add(metric: str, value: object, group: str = "summary") -> None:
            rows.append({"metric_group": group, "metric": metric, "value": value})

        add("total_raw_candidates", len(raw))
        add("accepted_count", len(accepted))
        add("rejected_count", len(rejected))
        if raw.empty:
            return pd.DataFrame(rows)
        add("restricted_count", int((raw["extraction_status"] == "restricted").sum()))
        add("paywalled_count", int((raw["access_status"] == "paywalled").sum()))
        add("robots_blocked_count", int(((raw["access_status"] == "blocked") & (raw["robots_allowed"] == False)).sum()))
        add("parse_failed_count", int((raw["extraction_status"] == "parse_failed").sum()))
        add("missing_timestamp_count", int((raw["timestamp_quality"] == "missing").sum()))
        add("unknown_source_count", int((raw["source_type"] == "general_web_unknown").sum()))
        for source_type, count in accepted["source_type"].value_counts(dropna=False).items():
            add(str(source_type), int(count), "accepted_by_source_type")
        for reason, count in rejected["rejection_reason"].replace("", "unspecified").value_counts(dropna=False).items():
            add(str(reason), int(count), "rejected_by_reason")
        for status, count in raw["extraction_status"].value_counts(dropna=False).items():
            add(str(status), int(count), "extraction_status_counts")
        return pd.DataFrame(rows)

    def _print_verbose_summary(self, raw: pd.DataFrame, accepted: pd.DataFrame, rejected: pd.DataFrame) -> None:
        print(f"[ingest] raw={len(raw)} accepted={len(accepted)} rejected={len(rejected)}")
        if raw.empty:
            return
        print("[ingest] extraction_status counts:")
        print(raw["extraction_status"].value_counts(dropna=False).to_string())
        print("[ingest] source_type counts:")
        print(raw["source_type"].value_counts(dropna=False).to_string())
        if not rejected.empty:
            print("[ingest] rejection reasons:")
            print(rejected["rejection_reason"].replace("", "unspecified").value_counts(dropna=False).to_string())

    def _flush_partial(self, processed: list[IngestedItem], errors: list[IngestionError]) -> None:
        raw, accepted, rejected = self._split_outputs(processed)
        diagnostics = self._build_diagnostics(raw, accepted, rejected)
        self._write_outputs(raw, accepted, rejected, errors, diagnostics)

    def _write_outputs(
        self,
        raw: pd.DataFrame,
        accepted: pd.DataFrame,
        rejected: pd.DataFrame,
        errors: list[IngestionError],
        diagnostics: pd.DataFrame,
    ) -> dict[str, Path]:
        paths = {
            "raw_candidates": self.output_dir / "raw_candidates.csv",
            "accepted_documents": self.output_dir / "accepted_documents.csv",
            "rejected_documents": self.output_dir / "rejected_documents.csv",
            "ingestion_errors": self.output_dir / "ingestion_errors.csv",
            "acceptance_diagnostics": self.output_dir / "acceptance_diagnostics.csv",
        }
        raw.to_csv(paths["raw_candidates"], index=False)
        accepted.to_csv(paths["accepted_documents"], index=False)
        rejected.to_csv(paths["rejected_documents"], index=False)
        error_frame = pd.DataFrame([error.to_record() for error in errors])
        if error_frame.empty:
            error_frame = pd.DataFrame(columns=["timestamp", "adapter", "query", "url", "error_state", "message"])
        error_frame.to_csv(paths["ingestion_errors"], index=False)
        diagnostics.to_csv(paths["acceptance_diagnostics"], index=False)
        return paths


def _append_note(existing: str, note: str) -> str:
    if not note:
        return existing or ""
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}; {note}"
