from __future__ import annotations

import urllib.request
from typing import Any

import pytest

from src.paper_titles import (
    USER_AGENT,
    extract_pdf_title_from_bytes,
    fetch_pdf_bytes,
    looks_like_metadata_or_title_excerpt,
    pdf_pages_text,
    pdf_url_from_values,
    real_paper_title,
    resolve_pdf_title_from_url,
    strip_markup,
)


def _pdf_bytes(
    *,
    metadata_title: str | None = None,
    first_page_text: str | None = None,
) -> bytes:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    if first_page_text:
        page.insert_text((72, 72), first_page_text)
    if metadata_title is not None:
        doc.set_metadata({"title": metadata_title})
    try:
        return doc.tobytes()
    finally:
        doc.close()


def _multi_page_pdf_bytes(pages: list[str]) -> bytes:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text)
    try:
        return doc.tobytes()
    finally:
        doc.close()


def test_real_paper_title_removes_arxiv_identifier_prefix() -> None:
    assert (
        real_paper_title("[2407.06645] Entropy Law: The Story Behind Compression")
        == "Entropy Law: The Story Behind Compression"
    )


def test_strip_markup_replaces_tags_with_space_and_unescapes_entities() -> None:
    # Regression fixture pinning the old ``sources._strip_jats_markup`` and
    # ``fulltext._jats_to_text`` tail behavior: tags -> space, then unescape, then
    # whitespace-join (not tag -> "" like ledger._strip_html).
    sample = "Findings<sup>1,2</sup> show &amp; confirm<br/>effects."
    assert strip_markup(sample) == "Findings 1,2 show & confirm effects."


def test_strip_markup_returns_none_for_none_or_empty() -> None:
    assert strip_markup(None) is None
    assert strip_markup("<b></b>") is None


def test_pdf_pages_text_returns_per_page_text_capped_at_max_pages() -> None:
    pdf = _multi_page_pdf_bytes(["Page one body.", "Page two body.", "Page three."])

    pages = pdf_pages_text(pdf, max_pages=2)

    assert pages is not None
    assert len(pages) == 2
    assert "Page one body." in pages[0]
    assert "Page two body." in pages[1]


def test_pdf_url_from_values_accepts_http_pdf_url_only() -> None:
    assert (
        pdf_url_from_values(
            "https://example.test/landing-page",
            "https://openreview.net/pdf/example.pdf?download=1",
        )
        == "https://openreview.net/pdf/example.pdf?download=1"
    )
    assert (
        pdf_url_from_values("file:///tmp/example.pdf", "https://example.test/page")
        is None
    )


def test_extract_pdf_title_prefers_real_metadata_title() -> None:
    pdf = _pdf_bytes(
        metadata_title="Recovered Metadata Paper Title",
        first_page_text="Fallback Page Title\nAbstract\nBody",
    )

    assert extract_pdf_title_from_bytes(
        pdf, source_url="https://example.test/paper.pdf"
    ) == ("Recovered Metadata Paper Title")


def test_extract_pdf_title_falls_back_to_first_page_when_metadata_is_filename() -> None:
    pdf = _pdf_bytes(
        metadata_title="paper",
        first_page_text="Recovered First Page Paper Title\nAbstract\nBody",
    )

    assert extract_pdf_title_from_bytes(
        pdf, source_url="https://example.test/paper.pdf"
    ) == ("Recovered First Page Paper Title")


def test_fetch_pdf_bytes_is_bounded_to_allowed_http_pdf_urls() -> None:
    def fail_opener(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("private PDF URL should be rejected before opening")

    with pytest.raises(ValueError, match="not allowed"):
        fetch_pdf_bytes("http://127.0.0.1/private.pdf", opener=fail_opener)


def test_blocked_fetch_host_rejects_internal_and_local_suffixes() -> None:
    from src.paper_titles import _blocked_fetch_host

    assert _blocked_fetch_host("http://metadata.google.internal/latest") is True
    assert _blocked_fetch_host("http://service.internal/x") is True
    assert _blocked_fetch_host("http://printer.local/x") is True
    assert _blocked_fetch_host("http://nodot/x") is True
    assert _blocked_fetch_host("https://example.com/paper.pdf") is False


def test_safe_urlopen_refuses_redirect_to_blocked_host() -> None:
    from src.paper_titles import _SafeRedirectHandler, _blocked_fetch_host

    handler = _SafeRedirectHandler()
    request = urllib.request.Request("https://example.com/paper.pdf")

    class _Fp:
        pass

    with pytest.raises(ValueError, match="redirect target is not allowed"):
        handler.redirect_request(
            request,
            _Fp(),
            302,
            "Found",
            {},
            "http://metadata.google.internal/latest",
        )
    assert _blocked_fetch_host("http://metadata.google.internal/latest") is True


def test_fetch_pdf_bytes_reads_pdf_response() -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self._chunks = [b"%PDF-1.7\n", b"body", b""]

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            del size
            return self._chunks.pop(0)

    def opener(request: Any, *, timeout: float) -> FakeResponse:
        assert request.full_url == "https://example.test/paper.pdf"
        assert timeout == 10.0
        # PDF-title-resolver UA must derive from the shared USER_AGENT, not restate it.
        assert request.get_header("User-agent") == f"{USER_AGENT} PDF-title-resolver"
        return FakeResponse()

    assert fetch_pdf_bytes("https://example.test/paper.pdf", opener=opener) == (
        b"%PDF-1.7\nbody"
    )


# Shared behavior checks for ``looks_like_metadata_or_title_excerpt`` as used by the
# retrieval ledger. The expected values pin behavior across consolidation of
# the formerly duplicated private helpers.
_TITLE = "Causal Graphs for Robust Inference"


def test_looks_like_metadata_or_title_excerpt_matches_keywords_heading() -> None:
    assert looks_like_metadata_or_title_excerpt("Keywords: causal inference, graphs", _TITLE)
    assert looks_like_metadata_or_title_excerpt("Index Terms— deep learning", _TITLE)
    # ported from the ledger's regression battery: internal whitespace, unrelated title
    assert looks_like_metadata_or_title_excerpt(
        "  Keywords:   foo bar   ", "Some Title Long Enough To Count Chars"
    )


def test_looks_like_metadata_or_title_excerpt_matches_exact_or_substring_title() -> None:
    assert looks_like_metadata_or_title_excerpt(_TITLE, _TITLE)
    assert looks_like_metadata_or_title_excerpt("Causal Graphs", _TITLE)
    assert looks_like_metadata_or_title_excerpt(
        "Causal graphs for robust inference", "CAUSAL GRAPHS FOR ROBUST INFERENCE"
    )
    # ported from the ledger's regression battery: an evidence-id-prefixed excerpt
    assert looks_like_metadata_or_title_excerpt(
        "ev_003 causal graphs for robust inference", _TITLE
    )


def test_looks_like_metadata_or_title_excerpt_rejects_unrelated_or_much_longer_text() -> None:
    assert not looks_like_metadata_or_title_excerpt(
        "An unrelated excerpt about something else entirely different topic", _TITLE
    )
    assert not looks_like_metadata_or_title_excerpt(
        "Causal Graphs for Robust Inference in Complex Systems with Extra Words "
        "Appended Here Now",
        _TITLE,
    )


def test_looks_like_metadata_or_title_excerpt_rejects_empty_or_short_title() -> None:
    assert not looks_like_metadata_or_title_excerpt("", _TITLE)
    assert not looks_like_metadata_or_title_excerpt(None, _TITLE)
    assert not looks_like_metadata_or_title_excerpt("Some text", "")
    assert not looks_like_metadata_or_title_excerpt("Some text", None)


def test_looks_like_metadata_or_title_excerpt_applies_normalize_hook_before_matching() -> None:
    # Without a normalize hook, markdown emphasis markers are stripped anyway
    # by the alnum-only identity filter, so this passes even with normalize=None.
    assert looks_like_metadata_or_title_excerpt("**Causal** Graphs for _Robust_ Inference", _TITLE)
    # A normalize hook that rewrites text can flip a match that the plain
    # (normalize=None) path would miss.

    def normalize(text: str) -> str:
        return text.replace("CG4RI", "Causal Graphs for Robust Inference")

    assert looks_like_metadata_or_title_excerpt("CG4RI", _TITLE, normalize=normalize)
    assert not looks_like_metadata_or_title_excerpt("CG4RI", _TITLE)


def test_resolve_pdf_title_from_url_suppresses_optional_fetch_warning(caplog) -> None:
    def opener(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("retrieval incomplete: got only 1 out of 2 bytes")

    caplog.set_level("WARNING")

    assert (
        resolve_pdf_title_from_url(
            "https://example.test/paper.pdf",
            opener=opener,
        )
        is None
    )
    assert "retrieval incomplete" not in caplog.text
