"""HTTP, robots, and HTML extraction helpers."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib import error, request, robotparser
from urllib.parse import urljoin, urlsplit

from ingestion.schema import utc_now_iso
from ingestion.utils import extract_published_date, looks_paywalled_or_blocked, parse_date_like


DEFAULT_USER_AGENT = "HBMNarrativeResearchBot/0.1 compliance-aware research prototype; contact=research@example.com"


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int | None
    content_type: str
    text: str
    error: str = ""


class RateLimiter:
    def __init__(self, delay_seconds: float = 1.0) -> None:
        self.delay_seconds = delay_seconds
        self._last_call_by_domain: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        domain = urlsplit(url).netloc.lower()
        with self._lock:
            now = time.monotonic()
            last = self._last_call_by_domain.get(domain)
            if last is not None:
                sleep_for = self.delay_seconds - (now - last)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._last_call_by_domain[domain] = time.monotonic()


class RobotsPolicy:
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: int = 8) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[str, robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()

    def allowed(self, url: str) -> bool | None:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return None
        root = f"{parts.scheme}://{parts.netloc}"
        with self._lock:
            parser = self._cache.get(root)
            if root not in self._cache:
                parser = robotparser.RobotFileParser()
                parser.set_url(urljoin(root, "/robots.txt"))
                try:
                    robots_req = request.Request(
                        urljoin(root, "/robots.txt"),
                        headers={"User-Agent": self.user_agent},
                    )
                    with request.urlopen(robots_req, timeout=self.timeout) as resp:
                        lines = resp.read(500_000).decode("utf-8", errors="replace").splitlines()
                    parser.parse(lines)
                    self._cache[root] = parser
                except Exception:
                    self._cache[root] = None
                    parser = None
        if parser is None:
            return None
        try:
            return bool(parser.can_fetch(self.user_agent, url))
        except Exception:
            return None


def fetch_url(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = 15,
    rate_limiter: RateLimiter | None = None,
) -> FetchResult:
    if rate_limiter:
        rate_limiter.wait(url)
    req = request.Request(url, headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(2_000_000)
            charset = resp.headers.get_content_charset() or "utf-8"
            return FetchResult(
                url=url,
                final_url=resp.geturl(),
                status_code=getattr(resp, "status", None),
                content_type=resp.headers.get("Content-Type", ""),
                text=raw.decode(charset, errors="replace"),
            )
    except error.HTTPError as exc:
        body = exc.read(200_000).decode("utf-8", errors="replace")
        return FetchResult(url=url, final_url=url, status_code=exc.code, content_type=exc.headers.get("Content-Type", ""), text=body, error=str(exc))
    except TimeoutError as exc:
        return FetchResult(url=url, final_url=url, status_code=None, content_type="", text="", error=f"timeout: {exc}")
    except Exception as exc:
        return FetchResult(url=url, final_url=url, status_code=None, content_type="", text="", error=str(exc))


class MetadataTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.meta: dict[str, str] = {}
        self.canonical_url = ""
        self._in_title = False
        self._skip_depth = 0
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "meta":
            key = attrs_dict.get("property") or attrs_dict.get("name") or attrs_dict.get("itemprop")
            value = attrs_dict.get("content")
            if key and value:
                self.meta[key.lower()] = value
        if tag == "link" and attrs_dict.get("rel", "").lower() == "canonical":
            self.canonical_url = attrs_dict.get("href", "")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._in_title:
            self.title += data
        if self._skip_depth == 0:
            clean = re.sub(r"\s+", " ", data).strip()
            if len(clean) > 2:
                self._text_chunks.append(clean)

    def parsed(self) -> dict[str, str | None]:
        title = self.meta.get("og:title") or self.meta.get("twitter:title") or self.title
        published = (
            self.meta.get("article:published_time")
            or self.meta.get("datepublished")
            or self.meta.get("datemodified")
            or self.meta.get("og:published_time")
            or self.meta.get("date")
            or self.meta.get("pubdate")
            or self.meta.get("publishdate")
            or self.meta.get("dc.date")
        )
        text = " ".join(self._text_chunks)
        text = unescape(re.sub(r"\s+", " ", text)).strip()
        parsed_published = parse_date_like(published)
        timestamp_quality = "exact" if parsed_published else "missing"
        if not parsed_published:
            parsed_published, timestamp_quality, _ = extract_published_date(text)
        return {
            "title": unescape(re.sub(r"\s+", " ", title or "")).strip(),
            "timestamp_published": parsed_published,
            "timestamp_quality": timestamp_quality,
            "canonical_url": self.canonical_url,
            "text": text,
        }


def parse_html(fetch: FetchResult) -> tuple[dict[str, str | None], str, str]:
    access_status, extraction_status = looks_paywalled_or_blocked(fetch.text, fetch.status_code, fetch.final_url)
    if access_status != "accessible":
        return {"title": "", "timestamp_published": None, "canonical_url": "", "text": ""}, access_status, extraction_status
    if "html" not in fetch.content_type.lower() and "<html" not in fetch.text[:500].lower():
        return {"title": "", "timestamp_published": None, "canonical_url": "", "text": ""}, access_status, "parse_failed"
    parser = MetadataTextParser()
    try:
        parser.feed(fetch.text)
        parsed = parser.parsed()
    except Exception:
        return {"title": "", "timestamp_published": None, "canonical_url": "", "text": ""}, access_status, "parse_failed"
    if not parsed.get("timestamp_published"):
        structured_date = _extract_json_ld_date(fetch.text)
        if structured_date:
            parsed["timestamp_published"] = structured_date
            parsed["timestamp_quality"] = "exact"
    if not parsed.get("timestamp_published"):
        inferred_date, quality, _ = extract_published_date(str(parsed.get("text") or ""), fetch.final_url)
        parsed["timestamp_published"] = inferred_date
        parsed["timestamp_quality"] = quality
    if not parsed.get("text"):
        return parsed, access_status, "parse_failed"
    return parsed, access_status, "success"


def parse_json_response(fetch: FetchResult) -> dict:
    if not fetch.text:
        return {}
    return json.loads(fetch.text)


def _extract_json_ld_date(html: str) -> str | None:
    for pattern in [r'"datePublished"\s*:\s*"([^"]+)"', r'"dateModified"\s*:\s*"([^"]+)"']:
        match = re.search(pattern, html or "", flags=re.IGNORECASE)
        if match:
            parsed = parse_date_like(match.group(1))
            if parsed:
                return parsed
    return None
