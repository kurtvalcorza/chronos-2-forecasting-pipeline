from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "chronos_2_forecasting_colab.ipynb"


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def test_tutorial_notebook_is_valid_pure_python_and_has_no_hidden_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells, start=1):
        text = source(cell)
        compile(text, f"tutorial-cell-{index}", "exec")
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        for line in text.splitlines():
            assert not line.lstrip().startswith(("!", "%"))


def test_tutorial_covers_the_live_release_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join(source(cell) for cell in notebook["cells"])

    for required in (
        "USE_BYOD",
        "chronological_holdout",
        "load_pinned_model",
        "forecast(split.history",
        "evaluate_forecast",
        "last_value_baseline",
        "seasonal_naive_baseline",
        "chronos_forecast.csv",
        "chronos_provenance.json",
        "chronos_evaluation.json",
        "RUN_COVARIATE_DEMO",
    ):
        assert required in text
