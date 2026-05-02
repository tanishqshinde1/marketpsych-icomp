"""Utility functions for compliance-aware discovery and normalization."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from html import unescape
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


RESTRICTED_DOMAINS = {
    "seekingalpha.com",
    "www.seekingalpha.com",
    "wsj.com",
    "www.wsj.com",
    "ft.com",
    "www.ft.com",
    "nytimes.com",
    "www.nytimes.com",
    "economist.com",
    "www.economist.com",
}

GENERAL_NEWS_DOMAINS = {
    "reuters.com",
    "www.reuters.com",
    "cnbc.com",
    "www.cnbc.com",
    "wsj.com",
    "ft.com",
    "economist.com",
    "bloomberg.com",
    "businesskorea.co.kr",
    "marketwatch.com",
    "barrons.com",
    "forbes.com",
    "fortune.com",
    "businessinsider.com",
}

SPECIALIST_DOMAINS = {
    "semianalysis.com",
    "newsletter.semianalysis.com",
    "trendforce.com",
    "tomshardware.com",
    "anandtech.com",
    "eetimes.com",
    "semiengineering.com",
    "datacenterknowledge.com",
    "thelec.net",
    "digitimes.com",
    "techspot.com",
    "wccftech.com",
}

SUPPLY_CHAIN_VENDOR_DOMAINS = {
    "microchipusa.com",
    "everstream.ai",
    "oemsecrets.com",
    "sourcengine.com",
}

IR_HINTS = ["investor relations", "/ir/", "/investors/", "quarterly results", "earnings release", "prepared remarks", "presentation", "financial-results"]
TRANSCRIPT_HINTS = ["earnings call transcript", "q1 earnings call", "q2 earnings call", "q3 earnings call", "q4 earnings call", "earnings transcript", "conference call transcript"]
SEC_HINTS = ["sec.gov", "edgar", "10-k", "10-q", "8-k", "ex-99"]
ANALYST_HINTS = ["morgan stanley", "goldman", "jpmorgan", "ubs", "barclays", "research note"]

COMPANY_PATTERNS = {
    "NVIDIA": ["nvidia", "nvda"],
    "AMD": ["advanced micro devices", " amd ", "amd,"],
    "Micron": ["micron", " mu "],
    "SK Hynix": ["sk hynix", "hynix"],
    "Samsung": ["samsung"],
    "TSMC": ["tsmc", "taiwan semiconductor"],
    "Broadcom": ["broadcom", "avgo"],
    "Marvell": ["marvell", "mrvl"],
    "ASML": ["asml"],
    "Applied Materials": ["applied materials", "amat"],
}


def canonicalize_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = re.sub(r"/+$", "", parts.path or "/")
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
        and k.lower()
        not in {
            "fbclid",
            "gclid",
            "mc_cid",
            "mc_eid",
            "cmpid",
            "ref",
        }
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def domain_from_url(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def normalize_title(title: str) -> str:
    title = unescape(title or "").lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def content_hash(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_doc_id(canonical_url: str, title: str = "") -> str:
    raw = f"{canonical_url}|{normalize_title(title)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def classify_source_type(
    url: str,
    domain: str | None = None,
    title: str = "",
    text: str | None = None,
) -> tuple[str, str, str]:
    """Classify source type with quality and reason.

    Returns `(source_type, source_type_quality, reason)`.
    """
    domain = (domain or domain_from_url(url)).lower().removeprefix("www.")
    haystack = f"{url} {domain} {title} {(text or '')[:1000]}".lower()
    path_haystack = f"{url} {title}".lower()

    if domain == "sec.gov" or domain.endswith(".sec.gov"):
        return "sec_filing", "domain_based", "SEC domain"
    if any(hint in haystack for hint in SEC_HINTS):
        return "sec_filing", "url_pattern_based", "SEC filing pattern"

    if any(hint in path_haystack for hint in TRANSCRIPT_HINTS):
        return "earnings_call_transcript", "url_pattern_based", "earnings transcript phrase in URL/title"
    if domain in {"seekingalpha.com", "fool.com", "marketscreener.com"} and any(
        token in path_haystack for token in ["transcript", "earnings-call", "earnings call", "conference-call"]
    ):
        return "earnings_call_transcript", "domain_based", "transcript provider domain and transcript pattern"

    if any(hint in haystack for hint in IR_HINTS):
        return "company_ir", "url_pattern_based", "investor relations or earnings release pattern"

    if domain in SPECIALIST_DOMAINS or any(domain.endswith(f".{known}") for known in SPECIALIST_DOMAINS):
        return "specialist_media", "domain_based", f"specialist media domain: {domain}"
    if any(hint in haystack for hint in ["semiconductor trade press", "semiconductor engineering", "semi analysis", "trendforce"]):
        return "specialist_media", "rule_based", "specialist semiconductor language"

    if domain in GENERAL_NEWS_DOMAINS or any(domain.endswith(f".{known}") for known in GENERAL_NEWS_DOMAINS):
        return "general_news", "domain_based", f"general news domain: {domain}"
    if any(news in domain for news in ["news", "business", "marketwatch", "barrons"]):
        return "general_news", "domain_based", "news/business domain pattern"

    if domain in SUPPLY_CHAIN_VENDOR_DOMAINS or any(domain.endswith(f".{known}") for known in SUPPLY_CHAIN_VENDOR_DOMAINS):
        return "supply_chain_vendor", "domain_based", f"supply chain/vendor domain: {domain}"
    if any(
        hint in haystack
        for hint in [
            "component distributor",
            "electronic components",
            "supply chain risk",
            "component marketplace",
            "semiconductor vendor",
            "authorized distributor",
        ]
    ):
        return "supply_chain_vendor", "rule_based", "supply-chain or component-marketplace language"

    if any(hint in haystack for hint in ANALYST_HINTS):
        return "analyst_or_institutional_commentary", "rule_based", "analyst/institutional language"

    return "general_web_unknown", "unknown", "no source classification rule matched"


def infer_company_entity(title: str = "", text: str = "", url: str = "", fallback_query: str = "") -> str:
    haystack = f" {title} {text[:1000]} {url} {fallback_query} ".lower()
    hits = []
    for company, patterns in COMPANY_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            hits.append(company)
    return "; ".join(dict.fromkeys(hits))


def parse_date_like(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    candidates = [
        text,
        text.replace("Z", "+00:00"),
    ]
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except ValueError:
            pass
    match = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    month_names = (
        "January|February|March|April|May|June|July|August|September|October|November|December|"
        "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    )
    match = re.search(rf"\b({month_names})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b", text, flags=re.IGNORECASE)
    if match:
        month, day, year = match.groups()
        return _date_from_month_name(year, month, day)
    match = re.search(rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b", text, flags=re.IGNORECASE)
    if match:
        day, month, year = match.groups()
        return _date_from_month_name(year, month, day)
    return None


def extract_published_date(text: str = "", url: str = "") -> tuple[str | None, str, str]:
    """Infer a publication date from visible text or URL patterns."""
    visible = parse_date_like((text or "")[:5000])
    if visible:
        return visible, "inferred", "date inferred from visible page text"
    url_match = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|-|$)", url)
    if url_match:
        year, month, day = url_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "inferred", "date inferred from URL path"
    url_match = re.search(r"/(20\d{2})-(\d{1,2})-(\d{1,2})(?:/|-|$)", url)
    if url_match:
        year, month, day = url_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "inferred", "date inferred from URL path"
    return None, "missing", "published timestamp not found"


def _date_from_month_name(year: str, month: str, day: str) -> str | None:
    try:
        parsed = datetime.strptime(f"{month} {int(day)} {year}", "%B %d %Y")
    except ValueError:
        try:
            parsed = datetime.strptime(f"{month} {int(day)} {year}", "%b %d %Y")
        except ValueError:
            return None
    return parsed.date().isoformat()


def looks_paywalled_or_blocked(html: str, status_code: int | None, url: str) -> tuple[str, str]:
    domain = domain_from_url(url)
    lower = (html or "").lower()
    if status_code in {401, 403}:
        return "blocked", "restricted"
    if domain in RESTRICTED_DOMAINS:
        return "paywalled", "restricted"
    paywall_terms = [
        "subscribe to continue",
        "sign in to continue",
        "already a subscriber",
        "create an account",
        "to continue reading",
        "subscription required",
        "enable javascript and cookies",
    ]
    if any(term in lower for term in paywall_terms):
        return "paywalled", "restricted"
    if status_code and status_code >= 400:
        return "blocked", "parse_failed"
    return "accessible", "parse_failed"
