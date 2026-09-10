from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_card_quantile_semantics_match_tutorial_contract() -> None:
    text = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "21 calibrated quantiles" not in text
    assert "not claimed to be calibrated prediction or confidence intervals" in text
    assert "uncalibrated model quantiles" in text


def test_model_card_names_only_the_implemented_evaluation_metrics() -> None:
    text = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    for required in (
        "**MAE**",
        "**RMSE**",
        "**per-quantile pinball loss**",
        "**empirical interval coverage**",
        "CRPS, MASE, and WAPE are **not** currently implemented",
    ):
        assert required in text


def test_readme_documents_split_sample_manifests() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`SHA256SUMS` contains only tutorial artifacts" in text
    assert "actually present in a clean checkout" in text
    assert "`SHA256SUMS.generated` records the canonical digests" in text


def test_release_docs_keep_multitarget_coverage_and_candidate_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "Mode C multi-target forecasting" in readme
    assert "Mode C multi-target forecasting" in registry
    assert "release candidate" in registry
    assert "release-grade" in registry
