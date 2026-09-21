from __future__ import annotations

import html
import ipaddress
import logging
import re
import urllib.request
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# Shared HTTP User-Agent (also used by retrieval/sources.py and retrieval/fulltext.py so
# it is defined once, here, since sources.py already imports this module).
USER_AGENT = "GraphHypoth/0.1 (+https://github.com/Mingxue-Xu/graph-hypoth)"

_ARXIV_TITLE_PREFIX_RE = re.compile(
    r"^\[(?:arxiv:)?(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?\]\s+",
    re.IGNORECASE,
)
_GENERIC_EVIDENCE_TITLE_RE = re.compile(
    r"^exa\s+result\s+\d+(?:\s*[-·]\s*[\w\s]+)?$",
    re.IGNORECASE,
)
_PDF_FILENAME_TITLE_RE = re.compile(r"^[^/\\]+\.pdf$", re.IGNORECASE)
_HASHLIKE_TITLE_RE = re.compile(r"^[a-f0-9]{16,}$", re.IGNORECASE)
# Shared tag-strip regex (also used by retrieval/sources.py and retrieval/fulltext.py
# via strip_markup(), so it is defined once, here).
_TAG_RE = re.compile(r"<[^>]+>")
# Shared with retrieval/ledger.py (which previously defined byte-identical copies of
# these two patterns).
METADATA_HEADING_RE = re.compile(r"^(?:keywords?|index\s+terms?)\b\s*[:—-]?", re.I)
METADATA_TO_SECTION_RE = re.compile(
    r"^\s*(?:keywords?|index\s+terms?)\b\s*[:—-]?.*?"
    r"(?:\b(?:\d+|[ivxlcdm]+)[.)]\s+[A-Z][A-Za-z ]+\s+)",
    re.I | re.S,
)


def strip_arxiv_title_prefix(title: str) -> str:
    return _ARXIV_TITLE_PREFIX_RE.sub("", title, count=1).strip()


def strip_markup(value: Any) -> str | None:
    """Strip HTML/XML tags (-> space) and unescape entities, then collapse whitespace.

    Shared by retrieval/sources.py (Crossref abstract) and retrieval/fulltext.py
    (JATS abstract/body). Not used by retrieval/ledger.py's title normalizer,
    which strips tags to "" (no space) instead — a genuinely different, local
    behavior, not just a lowercasing variant.
    """
    if value is None:
        return None
    text = _TAG_RE.sub(" ", str(value))
    text = html.unescape(text)
    text = " ".join(text.split())
    return text or None


def looks_like_placeholder_title(title: str) -> bool:
    lowered = title.lower().strip()
    if lowered in {
        "untitled",
        "(untitled)",
        "pdf",
        "document",
        "research paper",
        "article",
        "manuscript",
        "submission",
        "main",
        "output",
        "openreview",
        "arxiv",
    }:
        return True
    if _GENERIC_EVIDENCE_TITLE_RE.fullmatch(title):
        return True
    if lowered.startswith(("http://", "https://")):
        return True
    if _PDF_FILENAME_TITLE_RE.fullmatch(title):
        return True
    return bool(_HASHLIKE_TITLE_RE.fullmatch(title))


def real_paper_title(value: Any) -> str | None:
    if value is None:
        return None
    title = " ".join(str(value).split())
    title = strip_arxiv_title_prefix(title)
    if not title or len(title) > 250 or looks_like_placeholder_title(title):
        return None
    return title


def title_from_source_text(value: Any) -> str | None:
    if value is None:
        return None
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    title_lines: list[str] = []
    for line in lines[:10]:
        lowered = line.lower()
        if lowered in {"abstract", "introduction", "keywords"}:
            break
        if lowered.startswith(("arxiv:", "doi:", "http://", "https://")):
            continue
        if "@" in line or re.search(r"\d+\s*,?\s*\*|\d+\s*,?\s*†", line):
            if title_lines:
                break
            continue
        if any(
            marker in lowered
            for marker in (
                "university",
                "institute",
                "department",
                "school of",
            )
        ):
            if title_lines:
                break
            continue
        title_lines.append(line)
        if len(title_lines) >= 2:
            break
    return real_paper_title(" ".join(title_lines))


def looks_like_metadata_or_title_excerpt(
    value: str | None,
    title: str | None,
    *,
    normalize: Callable[[str], str] | None = None,
) -> bool:
    """True if `value` is a keywords/index-terms metadata line, or is (nearly)
    just the paper title reproduced back as an "excerpt".

    Called by retrieval/ledger.py without ``normalize``; the hook has no production
    caller today and is retained as an extension point. The 12/+30 magic
    numbers were consolidated here from duplicate implementations.
    """

    def _clean(text: str | None) -> str:
        text = normalize(text) if normalize else (text or "")
        return re.sub(r"\s+", " ", text).strip()

    def _identity(text: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", "", _clean(text).lower())

    compact = _clean(value)
    if not compact:
        return False
    if METADATA_HEADING_RE.match(compact):
        return True
    normalized = _identity(compact)
    normalized_title = _identity(title)
    if not normalized or len(normalized_title) < 12:
        return False
    if normalized == normalized_title:
        return True
    if normalized in normalized_title:
        return True
    return (
        normalized_title in normalized
        and len(normalized) <= len(normalized_title) + 30
    )


def canonical_source_url(value: str) -> str | None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def source_text_keys(result: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for raw_key in (result.get("id"), result.get("url")):
        key = None if raw_key is None else str(raw_key)
        if not key:
            continue
        keys.append(key)
        canonical_key = canonical_source_url(key)
        if canonical_key and canonical_key not in keys:
            keys.append(canonical_key)
    return keys


def looks_like_pdf_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return parsed.path.lower().endswith(".pdf")


def pdf_url_from_values(*values: str | None) -> str | None:
    for value in values:
        if looks_like_pdf_url(value):
            return value
    return None


def _blocked_fetch_host(url: str) -> bool:
    """Return True when ``url`` must not be fetched (SSRF / non-public target).

    Blocks loopback/private/link-local IP literals, localhost names, non-public
    DNS suffixes (``.internal``, ``.local``, …), hostnames without a dot, and
    hostnames that resolve only to non-public addresses. Aligns with the public
    domain rules used by Codex web retrieval.
    """
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return True
    if not host:
        return True
    lowered = host.lower().rstrip(".")
    if lowered in {"localhost", "localhost.localdomain"} or lowered.endswith(
        (".local", ".localhost", ".internal", ".home.arpa")
    ):
        return True
    if "." not in lowered:
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return _hostname_resolves_non_public(lowered)
    return _ip_is_non_public(address)


def _ip_is_non_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _hostname_resolves_non_public(hostname: str) -> bool:
    """True when DNS resolves to any non-public address.

    Unresolved names are not blocked here (transport fails later); this catches
    public-looking hostnames that point at loopback/private/link-local targets.
    """
    import socket

    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError:
        return False
    for info in infos:
        raw = info[4][0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if _ip_is_non_public(address):
            return True
    return False


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-check each redirect hop with ``_blocked_fetch_host`` before following."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        if _blocked_fetch_host(newurl):
            raise ValueError("redirect target is not allowed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_urlopen(request: urllib.request.Request, timeout: float = 10.0):
    """``urlopen`` that refuses blocked hosts on the initial URL and on redirects."""
    if _blocked_fetch_host(request.full_url):
        raise ValueError("fetch target is not allowed")
    opener = urllib.request.build_opener(_SafeRedirectHandler)
    return opener.open(request, timeout=timeout)


def fetch_pdf_bytes(
    url: str,
    *,
    timeout_seconds: float = 10.0,
    max_bytes: int = 20_000_000,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    if not looks_like_pdf_url(url):
        raise ValueError("url is not an http(s) PDF URL")
    if _blocked_fetch_host(url):
        raise ValueError("PDF title lookup target is not allowed")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"{USER_AGENT} PDF-title-resolver"},
    )
    open_url = opener or safe_urlopen
    with open_url(request, timeout=timeout_seconds) as response:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("PDF title lookup exceeded max bytes")
            chunks.append(chunk)
    data = b"".join(chunks)
    if not data.startswith(b"%PDF"):
        raise ValueError("downloaded content is not a PDF")
    return data


def _metadata_title_matches_url(title: str, source_url: str | None) -> bool:
    if not source_url:
        return False
    path = PurePosixPath(urlsplit(source_url).path)
    candidates = {path.name.lower(), path.stem.lower()}
    normalized_title = title.lower().strip()
    return normalized_title in candidates


def pdf_pages_text(pdf_bytes: bytes, *, max_pages: int) -> list[str]:
    """Extract per-page text from PDF bytes via PyMuPDF (fitz).

    Shared by extract_pdf_title_from_bytes (this module) and
    retrieval/fulltext.py's OA-PDF path. Raises (ImportError if fitz is
    missing, or a fitz error on a malformed PDF) — callers are expected to
    wrap the call in their own try/except and log per their own policy.
    """
    import fitz  # type: ignore

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [doc[index].get_text("text") for index in range(min(max_pages, len(doc)))]


def extract_pdf_title_from_bytes(
    pdf_bytes: bytes,
    *,
    source_url: str | None = None,
) -> str | None:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional extra.
        logger.warning("PyMuPDF unavailable for PDF title lookup: %s", exc)
        return None

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            metadata_title = real_paper_title((doc.metadata or {}).get("title"))
            if metadata_title and not _metadata_title_matches_url(
                metadata_title, source_url
            ):
                return metadata_title
        for page_text in pdf_pages_text(pdf_bytes, max_pages=2):
            title = title_from_source_text(page_text)
            if title:
                return title
    except Exception as exc:
        logger.warning("Failed to extract PDF title from %s: %s", source_url, exc)
    return None


def resolve_pdf_title_from_url(
    url: str,
    *,
    timeout_seconds: float = 10.0,
    max_bytes: int = 20_000_000,
    opener: Callable[..., Any] | None = None,
) -> str | None:
    try:
        pdf_bytes = fetch_pdf_bytes(
            url,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            opener=opener,
        )
    except Exception as exc:
        logger.debug("Failed to fetch PDF title source %s: %s", url, exc)
        return None
    return extract_pdf_title_from_bytes(pdf_bytes, source_url=url)
