from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "chronos_2_forecasting_colab.ipynb"


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def notebook_text() -> tuple[dict, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook, "\n".join(source(cell) for cell in notebook["cells"])


def test_notebook_declares_task_inference_profile_and_spec() -> None:
    notebook, text = notebook_text()
    assert notebook["metadata"]["dimer"] == {
        "notebook_profile": "TASK-INFERENCE",
        "notebook_spec": "1.0",
    }
    assert "**Notebook profile:** `TASK-INFERENCE`" in text
    assert "No training or fine-tuning occurs" in text


def test_learning_contract_and_capability_boundaries_are_explicit() -> None:
    _, text = notebook_text()
    for required in (
        "By the end of this notebook you will be able to:",
        "Chronos-2 supplies the pretrained forecasting model",
        "this repository adds the DIMER-facing configuration",
        "**This notebook does not demonstrate:**",
        "classification",
        "anomaly detection",
        "imputation",
        "production fitness",
    ):
        assert required in text


def test_runtime_provenance_limits_and_uncertainty_are_explained() -> None:
    _, text = notebook_text()
    for required in (
        "runtime_versions",
        "ResourceLimits",
        "Pinned-model context limit: 8192",
        "Pinned-model native prediction length: 1024",
        "477,930,472 bytes",
        "not guaranteed frequentist confidence intervals",
        "no random split",
    ):
        assert required in text


def test_byod_contract_rejects_duplicate_headers_before_pandas_and_states_data_boundary() -> None:
    _, text = notebook_text()
    assert "reject_duplicate_csv_headers(payload)" in text
    assert "frame = read_checked_csv(payload)" in text
    assert "is not sent by this notebook to an external inference service" in text
    assert "do not upload confidential, restricted, personal" in text


def test_forecasting_evaluation_contract_is_explicit() -> None:
    _, text = notebook_text()
    for required in (
        "single chronological tail holdout",
        "**MAE**",
        "**RMSE**",
        "**Pinball loss**",
        "**Empirical interval coverage**",
        "last-value baseline",
        "seasonal-naive baseline",
        "tutorial/sanity evidence",
    ):
        assert required in text


def test_output_contract_and_final_evidence_boundary_are_explicit() -> None:
    _, text = notebook_text()
    for required in (
        "`series_id`",
        "`timestamp`",
        "`target_name`",
        "`prediction`",
        "`q0.1`, `q0.5`, `q0.9`",
        "A successful default run **proves**",
        "It **does not prove**",
        "chronos_forecast.csv",
        "chronos_provenance.json",
        "chronos_evaluation.json",
    ):
        assert required in text


def test_notebook_contains_no_release_placeholders() -> None:
    _, text = notebook_text()
    upper = text.upper()
    for token in ("TODO", "TBD", "FIXME"):
        assert token not in upper
