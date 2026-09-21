from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel


class QuoteVerificationResult(BaseModel):
    candidate_excerpt: str
    match_type: Literal[
        "exact",
        "normalized",
        "normalized_window",
        "not_found",
        "no_source_text",
    ]
    verified_quote: str | None
    verification_status: Literal["accepted", "rejected", "unverified"]


_STITCHED_LOCATOR_RE = re.compile(r"\[\s*\.{3}\s*\]|\s\.{3}\s")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def verify_candidate_excerpt(
    *,
    candidate_excerpt: str,
    source_text: str | None,
) -> QuoteVerificationResult:
    candidate = _clean(candidate_excerpt)
    if not source_text:
        return QuoteVerificationResult(
            candidate_excerpt=candidate,
            match_type="no_source_text",
            verified_quote=None,
            verification_status="unverified",
        )
    if not candidate or _STITCHED_LOCATOR_RE.search(candidate):
        return _rejected(candidate)

    source_without_controls = _CONTROL_RE.sub("", source_text)
    if candidate in source_without_controls:
        return _accepted(candidate, "exact", candidate)

    source = _clean(source_text)
    normalized_candidate = _normalize(candidate)
    normalized_source = _normalize(source)
    if normalized_candidate and normalized_candidate in normalized_source:
        return _accepted(candidate, "normalized", candidate)

    window = _best_sentence_window(candidate, source)
    if window is not None:
        return _accepted(candidate, "normalized_window", window)
    return _rejected(candidate)


def _accepted(
    candidate: str,
    match_type: Literal["exact", "normalized", "normalized_window"],
    quote: str,
) -> QuoteVerificationResult:
    return QuoteVerificationResult(
        candidate_excerpt=candidate,
        match_type=match_type,
        verified_quote=_clean(quote),
        verification_status="accepted",
    )


def _rejected(candidate: str) -> QuoteVerificationResult:
    return QuoteVerificationResult(
        candidate_excerpt=candidate,
        match_type="not_found",
        verified_quote=None,
        verification_status="rejected",
    )


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", _CONTROL_RE.sub("", value)).strip()


def _normalize(value: str) -> str:
    return " ".join(token.lower() for token in _WORD_RE.findall(value))


def _sentences(value: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)", value)
        if match.group(0).strip()
    ]


def _best_sentence_window(candidate: str, source: str) -> str | None:
    candidate_norm = _normalize(candidate)
    if not candidate_norm:
        return None
    sentences = _sentences(source)
    best: tuple[float, int, str] | None = None
    # Single-sentence windows only: multi-sentence joins can assemble candidate terms
    # from unrelated sentences (e.g. "language" + "models learn representations").
    for sentence in sentences:
        window = _clean(sentence)
        window_norm = _normalize(window)
        # Every candidate term must appear in the window so a single swapped
        # content word (text/images, patients/adults, increases/decreases) cannot
        # pass as an accepted paraphrase.
        if _missing_term_count(candidate_norm, window_norm) != 0:
            continue
        ratio = SequenceMatcher(None, candidate_norm, window_norm).ratio()
        shared = _shared_term_count(candidate_norm, window_norm)
        score = ratio + min(shared, 8) * 0.05
        if best is None or score > best[0]:
            best = (score, 0, window)
    if best is None or best[0] < 0.72:
        return None
    return best[2]


def _shared_term_count(left: str, right: str) -> int:
    return len(set(left.split()) & set(right.split()))


def _missing_term_count(left: str, right: str) -> int:
    return len(set(left.split()) - set(right.split()))
