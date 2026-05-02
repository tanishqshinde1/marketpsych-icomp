"""Relevance scoring for discovered candidates."""

from __future__ import annotations

import re

from text_preprocessing import CANONICAL_NARRATIVE, NARRATIVE_KEYWORDS


SOURCE_TYPE_PRIORS = {
    "specialist_media": 0.10,
    "earnings_call_transcript": 0.10,
    "company_ir": 0.08,
    "sec_filing": 0.05,
    "supply_chain_vendor": 0.08,
    "general_news": 0.03,
    "analyst_or_institutional_commentary": 0.06,
    "general_web_unknown": 0.0,
    "unknown": 0.0,
}


def score_relevance(title: str, text: str, source_type: str, query: str = "") -> tuple[float, str]:
    haystack = f"{title} {text} {query}".lower()
    keyword_hits = [kw for kw in NARRATIVE_KEYWORDS if kw.lower() in haystack]
    core_hits = [
        term
        for term in ["hbm", "high bandwidth memory", "cowos", "advanced packaging", "ai accelerator", "gpu"]
        if term in haystack
    ]
    bottleneck_hits = [
        term
        for term in ["shortage", "constraint", "bottleneck", "allocation", "capacity", "lead time", "supply"]
        if term in haystack
    ]
    economic_hits = [
        term
        for term in ["pricing", "margin", "capex", "revenue", "guidance", "shipment", "production", "demand"]
        if term in haystack
    ]
    canonical_overlap = _token_overlap(CANONICAL_NARRATIVE, f"{title} {text}")
    score = 0.0
    score += min(0.35, 0.07 * len(keyword_hits))
    score += min(0.25, 0.08 * len(core_hits))
    score += min(0.20, 0.05 * len(bottleneck_hits))
    score += min(0.10, 0.025 * len(economic_hits))
    score += min(0.10, canonical_overlap * 0.25)
    score += SOURCE_TYPE_PRIORS.get(source_type, 0.0)
    score = max(0.0, min(1.0, score))
    reason_parts = []
    if keyword_hits:
        reason_parts.append(f"keyword hits: {', '.join(keyword_hits[:5])}")
    if core_hits and bottleneck_hits:
        reason_parts.append("contains memory/AI and bottleneck language")
    if economic_hits:
        reason_parts.append(f"economic terms: {', '.join(economic_hits[:4])}")
    if source_type != "unknown":
        reason_parts.append(f"source prior: {source_type}")
    if not reason_parts:
        reason_parts.append("weak narrative evidence")
    return score, "; ".join(reason_parts)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)
