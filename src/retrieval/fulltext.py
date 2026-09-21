from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from src.paper_titles import (
    USER_AGENT,
    _blocked_fetch_host,
    fetch_pdf_bytes,
    looks_like_pdf_url,
    pdf_pages_text,
    safe_urlopen,
    strip_markup,
)
from src.retrieval._util import get_attr_or_key as _get

logger = logging.getLogger(__name__)

_BODY_RE = re.compile(r"<(?:\w+:)?body\b[^>]*>(.*?)</(?:\w+:)?body>", re.IGNORECASE | re.DOTALL)
_ABSTRACT_RE = re.compile(
    r"<(?:\w+:)?abstract\b[^>]*>(.*?)</(?:\w+:)?abstract>", re.IGNORECASE | re.DOTALL
)


@dataclass
class _OAInfo:
    kind: str  # "jats" | "pdf"
    url: str


def detect_oa(result: Any) -> _OAInfo | None:
    """Decide whether/where an OA full text can be fetched for a result.

    JATS XML (Europe PMC) is preferred over PDF because it is structured and
    needs no binary parsing. Returns None when the result is not open access or
    no fetchable full-text location is known.
    """
    metadata = getattr(result, "metadata", None) or {}
    if not metadata.get("is_oa"):
        return None
    jats_url = metadata.get("jats_fulltext_url")
    if jats_url:
        return _OAInfo(kind="jats", url=str(jats_url))
    best = metadata.get("best_oa_location")
    pdf_url = metadata.get("oa_url")
    if not pdf_url and isinstance(best, dict):
        pdf_url = best.get("pdf_url")
    if pdf_url and looks_like_pdf_url(str(pdf_url)):
        return _OAInfo(kind="pdf", url=str(pdf_url))
    return None


def fetch_oa_fulltext(
    result: Any,
    config: Any,
    *,
    opener: Callable[..., Any] | None = None,
) -> str | None:
    """Best-effort fetch of OA full text for a result, discarded after parsing.

    Never raises and never persists anything to disk: the bytes are parsed in
    memory and dropped. Returns None when no OA full text is available or the
    fetch/parse fails.
    """
    info = detect_oa(result)
    if info is None:
        return None
    timeout = float(_get(config, "timeout_seconds", 15.0))
    max_bytes = int(_get(config, "max_bytes", 20_000_000))
    max_pages = int(_get(config, "max_pages", 40))
    try:
        if info.kind == "jats":
            return _fetch_jats_text(
                info.url, timeout=timeout, max_bytes=max_bytes, opener=opener
            )
        if info.kind == "pdf":
            pdf_bytes = fetch_pdf_bytes(
                info.url,
                timeout_seconds=timeout,
                max_bytes=max_bytes,
                opener=opener,
            )
            return _pdf_to_text(pdf_bytes, max_pages=max_pages)
    except Exception as exc:  # noqa: BLE001 - enrichment fetch is best-effort.
        logger.debug("OA full-text fetch failed for %s: %s", info.url, exc)
        return None
    return None


def _fetch_jats_text(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    opener: Callable[..., Any] | None,
) -> str | None:
    if _blocked_fetch_host(url):
        raise ValueError("JATS full-text host is not allowed")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    open_url = opener or safe_urlopen
    with open_url(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    # Measure the cap in bytes: a custom str-returning opener would make len()
    # count characters, under-counting multibyte UTF-8 content.
    size_bytes = len(raw) if isinstance(raw, bytes) else len(raw.encode("utf-8"))
    if size_bytes > max_bytes:
        raise ValueError("JATS full text exceeded max bytes")
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    return _jats_to_text(text)


def _jats_to_text(xml: str) -> str | None:
    fragments = [m.group(1) for m in _ABSTRACT_RE.finditer(xml)]
    fragments += [m.group(1) for m in _BODY_RE.finditer(xml)]
    if not fragments:
        # No <abstract>/<body>: do not dump <front>/<back> metadata as "full text".
        return None
    return strip_markup(" ".join(fragments))


def _pdf_to_text(pdf_bytes: bytes, *, max_pages: int) -> str | None:
    try:
        import fitz  # type: ignore  # noqa: F401  # availability probe; PyMuPDF used via pdf_pages_text
    except Exception as exc:  # pragma: no cover - depends on optional extra.
        logger.debug("PyMuPDF unavailable for OA full-text extraction: %s", exc)
        return None
    try:
        pages = pdf_pages_text(pdf_bytes, max_pages=max_pages)
    except Exception as exc:  # pragma: no cover - malformed PDFs.
        logger.debug("Failed to extract OA PDF text: %s", exc)
        return None
    text = " ".join(" ".join(page.split()) for page in pages if page)
    return text or None
