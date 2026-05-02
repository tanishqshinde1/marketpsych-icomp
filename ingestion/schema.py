"""Schemas and controlled vocabularies for ingestion outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


INGESTION_COLUMNS = [
    "doc_id",
    "query",
    "url",
    "canonical_url",
    "domain",
    "source_type",
    "discovery_method",
    "timestamp_published",
    "timestamp_quality",
    "timestamp_modeling_fallback",
    "timestamp_discovered",
    "title",
    "snippet",
    "company_entity",
    "access_status",
    "robots_allowed",
    "extraction_status",
    "text_permission_status",
    "text",
    "relevance_score",
    "relevance_reason",
    "content_hash",
    "version_id",
    "source_type_quality",
    "metadata_quality_notes",
]

ACCESS_STATUSES = {
    "accessible",
    "blocked",
    "paywalled",
    "manual_review",
    "unknown",
}

EXTRACTION_STATUSES = {
    "success",
    "parse_failed",
    "timeout",
    "duplicate",
    "restricted",
    "irrelevant",
}

TEXT_PERMISSION_STATUSES = {
    "full_text_allowed",
    "metadata_only",
    "unknown",
}

SOURCE_TYPES = {
    "specialist_media",
    "earnings_call_transcript",
    "company_ir",
    "sec_filing",
    "supply_chain_vendor",
    "general_news",
    "analyst_or_institutional_commentary",
    "social_high_frequency_reserved",
    "general_web_unknown",
    "unknown",
}


@dataclass
class IngestedItem:
    doc_id: str = ""
    query: str = ""
    url: str = ""
    canonical_url: str = ""
    domain: str = ""
    source_type: str = "unknown"
    discovery_method: str = ""
    timestamp_published: str | None = None
    timestamp_quality: str = "missing"
    timestamp_modeling_fallback: str | None = None
    timestamp_discovered: str = ""
    title: str = ""
    snippet: str = ""
    company_entity: str = ""
    access_status: str = "unknown"
    robots_allowed: bool | None = None
    extraction_status: str = "parse_failed"
    text_permission_status: str = "unknown"
    text: str = ""
    relevance_score: float = 0.0
    relevance_reason: str = ""
    content_hash: str = ""
    version_id: str = "v1"
    source_type_quality: str = "unknown"
    metadata_quality_notes: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp_discovered:
            self.timestamp_discovered = utc_now_iso()
        if self.timestamp_published and self.timestamp_quality == "missing":
            self.timestamp_quality = "exact"
        if not self.timestamp_modeling_fallback:
            self.timestamp_modeling_fallback = self.timestamp_discovered

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for col in INGESTION_COLUMNS:
            record.setdefault(col, None)
        return {col: record[col] for col in INGESTION_COLUMNS}


@dataclass
class IngestionError:
    timestamp: str
    adapter: str
    query: str
    url: str
    error_state: str
    message: str

    def to_record(self) -> dict[str, str]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
