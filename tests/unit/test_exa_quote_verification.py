from __future__ import annotations

from src.retrieval.quote_verification import verify_candidate_excerpt


def test_exact_excerpt_match_is_accepted() -> None:
    result = verify_candidate_excerpt(
        candidate_excerpt="Language models are lossless compressors.",
        source_text="We show that Language models are lossless compressors. More text.",
    )

    assert result.verification_status == "accepted"
    assert result.match_type == "exact"
    assert result.verified_quote == "Language models are lossless compressors."


def test_normalized_spacing_match_is_accepted() -> None:
    result = verify_candidate_excerpt(
        candidate_excerpt="Language models are lossless compressors.",
        source_text="Language\nmodels   are lossless compressors.",
    )

    assert result.verification_status == "accepted"
    assert result.match_type == "normalized"
    assert result.verified_quote == "Language models are lossless compressors."


def test_window_match_returns_source_text_sentence() -> None:
    result = verify_candidate_excerpt(
        candidate_excerpt="models trained on text generalize to image audio compression",
        source_text=(
            "Our models trained on text also generalize to image and audio "
            "compression tasks. The paper reports strong compression."
        ),
    )

    assert result.verification_status == "accepted"
    assert result.match_type == "normalized_window"
    assert result.verified_quote == (
        "Our models trained on text also generalize to image and audio "
        "compression tasks."
    )


def test_window_match_rejects_meaning_changing_paraphrases() -> None:
    cases = [
        (
            "models compress text efficiently today",
            "Vision models compress images efficiently today. Language models are different.",
        ),
        (
            "language models learn representations",
            "Vision models learn representations from pixels. Separate work studies language.",
        ),
        (
            "mortality declined among patients",
            "Mortality declined among adults. Pediatric rates were unchanged.",
        ),
        (
            "treatment increases survival rates substantially",
            "Treatment decreases survival rates substantially in this cohort.",
        ),
    ]
    for candidate, source in cases:
        result = verify_candidate_excerpt(candidate_excerpt=candidate, source_text=source)
        assert result.verification_status == "rejected", candidate
        assert result.match_type == "not_found"
        assert result.verified_quote is None


def test_stitched_locator_is_rejected() -> None:
    result = verify_candidate_excerpt(
        candidate_excerpt="Compression improves [ ... ] over baselines.",
        source_text="Compression improves substantially over baselines.",
    )

    assert result.verification_status == "rejected"
    assert result.match_type == "not_found"
    assert result.verified_quote is None


def test_no_source_text_is_unverified() -> None:
    result = verify_candidate_excerpt(
        candidate_excerpt="Compression improves over baselines.",
        source_text=None,
    )

    assert result.verification_status == "unverified"
    assert result.match_type == "no_source_text"
    assert result.verified_quote is None
