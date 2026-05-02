"""Source-specific discovery adapters."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from ingestion.http import DEFAULT_USER_AGENT, RateLimiter, fetch_url, parse_json_response
from ingestion.query_generation import QueryPlan
from ingestion.schema import IngestedItem, IngestionError, utc_now_iso
from ingestion.utils import canonicalize_url, domain_from_url, infer_company_entity, parse_date_like, stable_doc_id


@dataclass
class AdapterConfig:
    start_date: str = "2023-01-01"
    end_date: str | None = None
    max_results_per_query: int = 10
    max_queries_per_run: int = 30
    max_pages_per_query: int = 2
    user_agent: str = DEFAULT_USER_AGENT
    rate_limit_seconds: float = 1.0
    feed_urls: list[str] = field(default_factory=list)
    sitemap_urls: list[str] = field(default_factory=list)
    search_results_file: str | None = None
    serpapi_api_key: str | None = None
    verbose: bool = False
    companies: list[str] = field(default_factory=lambda: QueryPlan().companies)
    sec_user_agent: str | None = None


class BaseAdapter:
    name = "base"

    def __init__(self, config: AdapterConfig, rate_limiter: RateLimiter | None = None) -> None:
        self.config = config
        self.rate_limiter = rate_limiter or RateLimiter(config.rate_limit_seconds)

    def discover(self, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        raise NotImplementedError

    def _candidate(
        self,
        query: str,
        url: str,
        title: str = "",
        published: str | None = None,
        snippet: str = "",
        discovery_method: str | None = None,
    ) -> IngestedItem:
        canonical = canonicalize_url(url)
        return IngestedItem(
            doc_id=stable_doc_id(canonical, title),
            query=query,
            url=url,
            canonical_url=canonical,
            domain=domain_from_url(canonical),
            discovery_method=discovery_method or self.name,
            timestamp_published=published,
            timestamp_discovered=utc_now_iso(),
            title=title or "",
            snippet=snippet or "",
            company_entity=infer_company_entity(title=title, url=url, fallback_query=query),
        )


class SearchAdapter(BaseAdapter):
    """Abstract search-provider adapter.

    Search providers are discovery only. They return URLs and metadata, never
    model-ready full text.
    """

    name = "search"


class SerpApiGoogleSearchAdapter(SearchAdapter):
    """SerpAPI-backed Google search discovery adapter.

    Uses the official `google-search-results` Python client. It runs both
    standard Google search and Google News search (`tbm=nws`) for each query and
    paginates with `start = 0, 10, 20, ...`.
    """

    name = "serpapi_google"

    def discover(self, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        if self.config.search_results_file:
            return self._from_structured_serp_export(self.config.search_results_file)

        api_key = self.config.serpapi_api_key or os.getenv("SERPAPI_KEY")
        if not api_key:
            return [], [
                IngestionError(
                    timestamp=utc_now_iso(),
                    adapter=self.name,
                    query="",
                    url="",
                    error_state="missing_api_key",
                    message="Set SERPAPI_KEY or provide --search-results-file exported from a structured SERP provider.",
                )
            ]

        try:
            from serpapi import GoogleSearch
        except ImportError:
            return [], [
                IngestionError(
                    timestamp=utc_now_iso(),
                    adapter=self.name,
                    query="",
                    url="",
                    error_state="missing_dependency",
                    message="Install the official SerpAPI client with: pip install google-search-results",
                )
            ]

        items: list[IngestedItem] = []
        errors: list[IngestionError] = []
        limited_queries = queries[: self.config.max_queries_per_run]
        page_starts = [page * self.config.max_results_per_query for page in range(self.config.max_pages_per_query)]
        for query in limited_queries:
            if self.config.verbose:
                print(f"[serpapi] query={query}")
            dated_query = _query_with_date_range(query, self.config.start_date, self.config.end_date)
            for search_mode in ["organic", "news"]:
                for start in page_starts:
                    params = {
                        "engine": "google",
                        "q": dated_query,
                        "api_key": api_key,
                        "hl": "en",
                        "gl": "us",
                        "num": self.config.max_results_per_query,
                        "start": start,
                    }
                    if search_mode == "news":
                        params["tbm"] = "nws"
                    try:
                        self.rate_limiter.wait("https://serpapi.com/search")
                        results = GoogleSearch(params).get_dict()
                    except Exception as exc:
                        errors.append(IngestionError(utc_now_iso(), self.name, query, "", "api_failure", str(exc)))
                        continue

                    api_error = results.get("error") or results.get("serpapi_error")
                    if api_error:
                        error_state = "quota_exceeded" if "quota" in str(api_error).lower() else "api_failure"
                        errors.append(IngestionError(utc_now_iso(), self.name, query, "", error_state, str(api_error)))
                        continue

                    extracted = self._items_from_serpapi_results(query, results, search_mode)
                    if self.config.verbose:
                        print(f"[serpapi] query={query!r} mode={search_mode} start={start} candidates={len(extracted)}")
                    if not extracted:
                        errors.append(
                            IngestionError(
                                utc_now_iso(),
                                self.name,
                                query,
                                "",
                                "empty_results",
                                f"No {search_mode} results for start={start}.",
                            )
                        )
                    items.extend(extracted)
        return _filter_search_items(items), errors

    def _from_structured_serp_export(self, path: str) -> tuple[list[IngestedItem], list[IngestionError]]:
        source = Path(path)
        if not source.exists():
            return [], [IngestionError(utc_now_iso(), self.name, "", path, "inaccessible", "Structured SERP export file not found.")]
        if source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
        elif source.suffix.lower() in {".jsonl", ".ndjson"}:
            frame = pd.read_json(source, lines=True)
        else:
            frame = pd.read_json(source)
        items = []
        for row in frame.to_dict(orient="records"):
            url = str(row.get("url") or row.get("link") or "")
            if not url:
                continue
            items.append(
                self._candidate(
                    query=str(row.get("query") or ""),
                    url=url,
                    title=str(row.get("title") or row.get("name") or ""),
                    published=parse_date_like(row.get("timestamp_published") or row.get("date")),
                    snippet=str(row.get("snippet") or ""),
                    discovery_method=str(row.get("discovery_method") or self.name),
                )
            )
        filtered = _filter_search_items(items)
        if self.config.verbose:
            print(f"[serpapi] structured export rows={len(items)} filtered_candidates={len(filtered)}")
        return filtered, []

    def _items_from_serpapi_results(
        self,
        query: str,
        results: dict,
        search_mode: str,
    ) -> list[IngestedItem]:
        raw_results = []
        if search_mode == "news":
            raw_results.extend(results.get("news_results") or [])
            raw_results.extend(results.get("organic_results") or [])
        else:
            raw_results.extend(results.get("organic_results") or [])
        items = []
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            url = result.get("link") or result.get("url")
            title = result.get("title") or result.get("source") or ""
            snippet = result.get("snippet") or result.get("description") or ""
            if not url:
                continue
            item = self._candidate(
                query=query,
                url=url,
                title=title,
                published=parse_date_like(result.get("date") or result.get("published_date")),
                snippet=snippet,
                discovery_method=f"{self.name}_{search_mode}",
            )
            items.append(item)
        return items


class RssSitemapAdapter(BaseAdapter):
    name = "rss_sitemap"

    def discover(self, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        items: list[IngestedItem] = []
        errors: list[IngestionError] = []
        for feed_url in self.config.feed_urls:
            feed_items, feed_errors = self._discover_feed(feed_url, queries)
            items.extend(feed_items)
            errors.extend(feed_errors)
        for sitemap_url in self.config.sitemap_urls:
            sitemap_items, sitemap_errors = self._discover_sitemap(sitemap_url, queries)
            items.extend(sitemap_items)
            errors.extend(sitemap_errors)
        return items, errors

    def _discover_feed(self, feed_url: str, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        fetch = fetch_url(feed_url, user_agent=self.config.user_agent, rate_limiter=self.rate_limiter)
        if fetch.error or not fetch.text:
            return [], [IngestionError(utc_now_iso(), self.name, "", feed_url, "inaccessible", fetch.error or "empty feed")]
        try:
            root = ET.fromstring(fetch.text)
        except ET.ParseError as exc:
            return [], [IngestionError(utc_now_iso(), self.name, "", feed_url, "parse_failed", str(exc))]
        items = []
        for node in root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title = _child_text(node, ["title", "{http://www.w3.org/2005/Atom}title"])
            link = _child_text(node, ["link"])
            if not link:
                link_node = node.find("{http://www.w3.org/2005/Atom}link")
                link = link_node.attrib.get("href", "") if link_node is not None else ""
            published = parse_date_like(_child_text(node, ["pubDate", "published", "{http://www.w3.org/2005/Atom}published", "{http://www.w3.org/2005/Atom}updated"]))
            if published and not _date_in_range(published, self.config.start_date, self.config.end_date):
                continue
            matched_query = _best_query_match(title, link, queries)
            if link and matched_query:
                items.append(self._candidate(matched_query, link, title, published))
        return items, []

    def _discover_sitemap(self, sitemap_url: str, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        fetch = fetch_url(sitemap_url, user_agent=self.config.user_agent, rate_limiter=self.rate_limiter)
        if fetch.error or not fetch.text:
            return [], [IngestionError(utc_now_iso(), self.name, "", sitemap_url, "inaccessible", fetch.error or "empty sitemap")]
        try:
            root = ET.fromstring(fetch.text)
        except ET.ParseError as exc:
            return [], [IngestionError(utc_now_iso(), self.name, "", sitemap_url, "parse_failed", str(exc))]
        items = []
        for url_node in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = _child_text(url_node, ["{http://www.sitemaps.org/schemas/sitemap/0.9}loc"])
            lastmod = parse_date_like(_child_text(url_node, ["{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod"]))
            if lastmod and not _date_in_range(lastmod, self.config.start_date, self.config.end_date):
                continue
            matched_query = _best_query_match("", loc, queries)
            if loc and matched_query:
                items.append(self._candidate(matched_query, loc, "", lastmod))
        return items, []


class InvestorRelationsAdapter(BaseAdapter):
    name = "investor_relations"

    def discover(self, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        ir_queries = []
        for company in self.config.companies:
            ir_queries.extend(
                [
                    f"{company} HBM investor relations quarterly results",
                    f"{company} HBM earnings call prepared remarks",
                    f"{company} High Bandwidth Memory supply constraint investor relations",
                ]
            )
        search = SerpApiGoogleSearchAdapter(self.config, self.rate_limiter)
        items, errors = search.discover(list(dict.fromkeys(ir_queries)))
        for item in items:
            item.discovery_method = self.name
            if item.source_type == "unknown":
                item.source_type = "company_ir"
        return items, errors


class SecEdgarAdapter(BaseAdapter):
    name = "sec_edgar"

    CIKS = {
        "NVIDIA": "0001045810",
        "AMD": "0000002488",
        "Micron": "0000723125",
        "Broadcom": "0001730168",
        "Marvell": "0001835632",
        "ASML": "0000937966",
        "Applied Materials": "0000006951",
    }

    def discover(self, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        user_agent = self.config.sec_user_agent or self.config.user_agent
        items: list[IngestedItem] = []
        errors: list[IngestionError] = []
        for company in self.config.companies:
            cik = self.CIKS.get(company)
            if not cik:
                continue
            submissions_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            fetch = fetch_url(submissions_url, user_agent=user_agent, rate_limiter=self.rate_limiter)
            if fetch.error or fetch.status_code != 200:
                errors.append(IngestionError(utc_now_iso(), self.name, company, submissions_url, "inaccessible", fetch.error or f"status {fetch.status_code}"))
                continue
            try:
                data = parse_json_response(fetch)
                recent = data.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                accessions = recent.get("accessionNumber", [])
                docs = recent.get("primaryDocument", [])
                for form, filing_date, accession, primary_doc in zip(forms, dates, accessions, docs, strict=False):
                    if form not in {"10-K", "10-Q", "8-K", "6-K", "20-F"}:
                        continue
                    if not _date_in_range(filing_date, self.config.start_date, self.config.end_date):
                        continue
                    acc_no_dash = accession.replace("-", "")
                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dash}/{primary_doc}"
                    title = f"{company} {form} filing {filing_date}"
                    item = self._candidate("SEC EDGAR HBM supply constraint filings", filing_url, title, filing_date)
                    item.source_type = "sec_filing"
                    item.company_entity = company
                    items.append(item)
            except Exception as exc:
                errors.append(IngestionError(utc_now_iso(), self.name, company, submissions_url, "parse_failed", str(exc)))
        return items, errors


class TranscriptCandidateAdapter(BaseAdapter):
    name = "transcript_candidate"

    def discover(self, queries: list[str]) -> tuple[list[IngestedItem], list[IngestionError]]:
        transcript_queries = []
        for company in self.config.companies:
            transcript_queries.extend(
                [
                    f"{company} HBM shortage earnings call transcript",
                    f"{company} High Bandwidth Memory earnings call transcript",
                    f"{company} supply constraint earnings call transcript",
                ]
            )
        search = SerpApiGoogleSearchAdapter(self.config, self.rate_limiter)
        items, errors = search.discover(list(dict.fromkeys(transcript_queries)))
        for item in items:
            item.discovery_method = self.name
            item.source_type = "earnings_call_transcript"
        return items, errors


def _child_text(node, names: list[str]) -> str:
    for name in names:
        child = node.find(name)
        if child is not None and child.text:
            return child.text.strip()
    return ""


def _best_query_match(title: str, url: str, queries: list[str]) -> str:
    haystack = f"{title} {url}".lower()
    for query in queries:
        tokens = [token for token in query.lower().split() if len(token) > 2]
        if any(token in haystack for token in tokens[:4]):
            return query
    return ""


def _date_in_range(value: str, start_date: str, end_date: str | None) -> bool:
    parsed = parse_date_like(value)
    if not parsed:
        return True
    current = date.fromisoformat(parsed)
    if start_date and current < date.fromisoformat(start_date):
        return False
    if end_date and current > date.fromisoformat(end_date):
        return False
    return True


def _query_with_date_range(query: str, start_date: str, end_date: str | None) -> str:
    parts = [query]
    if start_date:
        parts.append(f"after:{start_date}")
    if end_date:
        parts.append(f"before:{end_date}")
    return " ".join(parts)


def _filter_search_items(items: list[IngestedItem]) -> list[IngestedItem]:
    """Deduplicate and lightly filter SERP discoveries before page extraction."""
    seen_urls: set[str] = set()
    filtered: list[IngestedItem] = []
    low_quality_domains = {
        "google.com",
        "bing.com",
        "youtube.com",
        "youtu.be",
        "facebook.com",
        "x.com",
        "twitter.com",
        "reddit.com",
    }
    for item in items:
        if not item.url:
            continue
        canonical = canonicalize_url(item.canonical_url or item.url)
        domain = domain_from_url(canonical)
        if canonical in seen_urls:
            continue
        if domain in low_quality_domains:
            continue
        if not _is_probably_english(f"{item.title} {item.snippet}"):
            continue
        item.canonical_url = canonical
        item.domain = domain
        seen_urls.add(canonical)
        filtered.append(item)
    return filtered


def _is_probably_english(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return True
    ascii_chars = sum(1 for char in sample if ord(char) < 128)
    ascii_ratio = ascii_chars / max(1, len(sample))
    common_english_markers = {"the", "and", "for", "with", "supply", "memory", "shortage", "earnings"}
    tokens = {token.lower().strip(".,:;()[]") for token in sample.split()}
    return ascii_ratio >= 0.85 or bool(tokens & common_english_markers)
