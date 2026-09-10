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


def code_text() -> str:
    notebook, _ = notebook_text()
    return "\n".join(
        source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )


def test_every_code_cell_is_valid_python() -> None:
    notebook, _ = notebook_text()
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        compile(source(cell), f"{NOTEBOOK}:cell-{index}", "exec")


def test_notebook_declares_task_inference_profile_and_spec() -> None:
    notebook, text = notebook_text()
    assert notebook["metadata"]["dimer"] == {
        "notebook_profile": "TASK-INFERENCE",
        "notebook_spec": "1.0",
    }
    assert "DIMER `TASK-INFERENCE`" in text
    assert "No training or fine-tuning occurs" in text


def test_learning_contract_and_capability_boundaries_are_explicit() -> None:
    _, text = notebook_text()
    for required in (
        "By the end",
        "Chronos-2 supplies the pretrained zero-shot forecasting model",
        "This repository supplies DIMER configuration and validation",
        "Out of scope:",
        "classification",
        "anomaly detection",
        "imputation",
        "production-fitness",
    ):
        assert required in text


def test_runtime_limits_and_uncertainty_are_explained() -> None:
    _, text = notebook_text()
    for required in (
        "runtime_versions",
        "ResourceLimits",
        "Pinned-model context limit: 8192",
        "Pinned-model native prediction length: 1024",
        "~478 MB",
        "not guaranteed frequentist confidence intervals",
        "no random split",
    ):
        assert required in text


def test_byod_duplicate_headers_are_rejected_before_pandas_parse() -> None:
    _, text = notebook_text()
    code = code_text()
    duplicate_guard = code.index("duplicates = sorted")
    pandas_parse = code.index("return pd.read_csv(io.BytesIO(payload))")
    assert duplicate_guard < pandas_parse
    assert "Counter(header)" in code
    assert "is not sent to an external inference service" in text
    assert "Do not upload confidential, restricted, personal" in text


def test_forecasting_evaluation_contract_is_explicit() -> None:
    _, text = notebook_text()
    for required in (
        "chronological holdout",
        "**MAE**",
        "**RMSE**",
        "**pinball loss**",
        "**empirical interval coverage**",
        "tutorial/sanity metrics",
        "last_value_evaluation",
    ):
        assert required in text


def test_short_hourly_byod_cannot_fail_only_for_seasonal_baseline() -> None:
    code = code_text()
    assert "minimum_history >= 24" in code
    assert "if can_use_daily_seasonal:" in code
    assert "seasonal_evaluation = None" in code


def test_primary_multitarget_mode_is_executed_and_checked() -> None:
    _, text = notebook_text()
    code = code_text()
    assert "Primary Mode C" in text
    assert 'target=["target", "target_aux"]' in code
    assert "mode_c_result = forecast(" in code
    assert 'mode_c_result.forecast["target_name"]' in code
    assert "chronos_multitarget_forecast.csv" in code


def test_visualization_is_executable_and_machine_outputs_remain_authoritative() -> None:
    _, text = notebook_text()
    code = code_text()
    assert "Visualize the held-out forecast" in text
    assert "chronos_forecast.svg" in code
    assert "from IPython.display import SVG, display" in code
    assert "machine-readable exports remain authoritative" in text


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
