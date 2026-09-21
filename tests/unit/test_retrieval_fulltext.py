from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.paper_titles import USER_AGENT
from src.retrieval.fulltext import detect_oa, fetch_oa_fulltext
from src.retrieval.models import SourceResult


def _config(**overrides):
    base = {"enabled": True, "timeout_seconds": 15.0, "max_bytes": 1_000_000, "max_pages": 40}
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self, amount: int | None = None) -> bytes:
        return self._payload


def _opener(payload: bytes, captured: dict | None = None):
    def opener(request, timeout=None):
        if captured is not None:
            captured["request"] = request
        return _FakeHttpResponse(payload)

    return opener


class _FakeStreamingResponse:
    # fetch_pdf_bytes reads in a chunk loop (unlike the single-shot JATS read
    # above), so this fake must go empty after one read like a real socket.
    def __init__(self, payload: bytes) -> None:
        self._chunks = [payload, b""]

    def __enter__(self) -> "_FakeStreamingResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self, amount: int | None = None) -> bytes:
        return self._chunks.pop(0)


def _pdf_opener(payload: bytes):
    def opener(request, timeout=None):
        return _FakeStreamingResponse(payload)

    return opener


def _pdf_bytes(pages: list[str]) -> bytes:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text)
    try:
        return doc.tobytes()
    finally:
        doc.close()


def test_detect_oa_prefers_jats_url() -> None:
    result = SourceResult(
        source="europepmc",
        title="t",
        metadata={"jats_fulltext_url": "https://www.ebi.ac.uk/x/fullTextXML", "is_oa": True},
    )
    info = detect_oa(result)
    assert info is not None and info.kind == "jats"


def test_detect_oa_returns_pdf_when_open_access_pdf_present() -> None:
    result = SourceResult(
        source="openalex",
        title="t",
        metadata={"is_oa": True, "oa_url": "https://example.org/paper.pdf"},
    )
    info = detect_oa(result)
    assert info is not None and info.kind == "pdf"


def test_detect_oa_returns_none_when_not_open_access() -> None:
    result = SourceResult(source="crossref", title="t", metadata={"is_oa": False})
    assert detect_oa(result) is None


def test_fetch_oa_fulltext_extracts_jats_body_text() -> None:
    jats = (
        b"<article><front><title>T</title></front>"
        b"<body><sec><p>Grounded result &amp; method.</p></sec></body></article>"
    )
    result = SourceResult(
        source="europepmc",
        title="t",
        metadata={"jats_fulltext_url": "https://www.ebi.ac.uk/x/fullTextXML", "is_oa": True},
    )

    captured: dict = {}
    text = fetch_oa_fulltext(result, _config(), opener=_opener(jats, captured))

    assert text == "Grounded result & method."  # only body, tags stripped
    # Must share the single USER_AGENT constant (paper_titles.py), not restate it.
    assert captured["request"].get_header("User-agent") == USER_AGENT


def test_fetch_oa_fulltext_extracts_pdf_body_text_across_pages() -> None:
    pdf = _pdf_bytes(["Page one body.", "Page two body."])
    result = SourceResult(
        source="openalex",
        title="t",
        metadata={"is_oa": True, "oa_url": "https://example.org/paper.pdf"},
    )

    text = fetch_oa_fulltext(result, _config(), opener=_pdf_opener(pdf))

    assert text is not None
    assert "Page one body." in text
    assert "Page two body." in text


def test_fetch_oa_fulltext_caps_pdf_pages_at_max_pages() -> None:
    pdf = _pdf_bytes(["Page one body.", "Page two body.", "Page three body."])
    result = SourceResult(
        source="openalex",
        title="t",
        metadata={"is_oa": True, "oa_url": "https://example.org/paper.pdf"},
    )

    text = fetch_oa_fulltext(
        result, _config(max_pages=1), opener=_pdf_opener(pdf)
    )

    assert text is not None
    assert "Page one body." in text
    assert "Page three body." not in text


def test_fetch_oa_fulltext_returns_none_for_non_oa() -> None:
    result = SourceResult(source="crossref", title="t", metadata={"is_oa": False})
    assert fetch_oa_fulltext(result, _config()) is None


def test_fetch_oa_fulltext_is_best_effort_on_transport_error() -> None:
    def failing_opener(request, timeout=None):
        raise OSError("ebi down")

    result = SourceResult(
        source="europepmc",
        title="t",
        metadata={"jats_fulltext_url": "https://www.ebi.ac.uk/x/fullTextXML", "is_oa": True},
    )

    assert fetch_oa_fulltext(result, _config(), opener=failing_opener) is None


def test_fetch_oa_fulltext_enforces_max_bytes() -> None:
    big = b"<body>" + b"x" * 50 + b"</body>"
    result = SourceResult(
        source="europepmc",
        title="t",
        metadata={"jats_fulltext_url": "https://www.ebi.ac.uk/x/fullTextXML", "is_oa": True},
    )

    assert fetch_oa_fulltext(result, _config(max_bytes=4), opener=_opener(big)) is None


def test_fetch_oa_fulltext_includes_abstract_and_body() -> None:
    jats = (
        b"<article><front>"
        b"<abstract><p>Abstract finding here.</p></abstract>"
        b"</front>"
        b"<body><sec><p>Body method &amp; result.</p></sec></body></article>"
    )
    result = SourceResult(
        source="europepmc",
        title="t",
        metadata={"jats_fulltext_url": "https://www.ebi.ac.uk/x/fullTextXML", "is_oa": True},
    )

    text = fetch_oa_fulltext(result, _config(), opener=_opener(jats))

    assert text is not None
    assert "Abstract finding here." in text
    assert "Body method & result." in text
    # abstract precedes body
    assert text.index("Abstract finding here.") < text.index("Body method & result.")


def test_fetch_oa_fulltext_returns_none_when_only_front_metadata() -> None:
    jats = (
        b"<article><front>"
        b"<journal-title>Secret Journal</journal-title>"
        b"<article-title>Leaked Title</article-title>"
        b"<contrib><name>Doe, Jane</name></contrib>"
        b"</front></article>"
    )
    result = SourceResult(
        source="europepmc",
        title="t",
        metadata={"jats_fulltext_url": "https://www.ebi.ac.uk/x/fullTextXML", "is_oa": True},
    )

    assert fetch_oa_fulltext(result, _config(), opener=_opener(jats)) is None


def test_fetch_oa_fulltext_enforces_max_bytes_for_str_opener() -> None:
    # 5 multibyte chars => len()==5 (<= max_bytes char count) but 15 bytes in UTF-8.
    payload = "é" * 5  # each char is 2 bytes in UTF-8 -> 10 bytes
    assert len(payload) <= 6
    assert len(payload.encode("utf-8")) > 6
    result = SourceResult(
        source="europepmc",
        title="t",
        metadata={"jats_fulltext_url": "https://www.ebi.ac.uk/x/fullTextXML", "is_oa": True},
    )

    assert fetch_oa_fulltext(result, _config(max_bytes=6), opener=_opener(payload)) is None
