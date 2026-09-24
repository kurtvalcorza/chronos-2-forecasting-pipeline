# ruff: noqa: E501,I001
"""Static contract tests for the multi-model forecasting workshop."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb"

def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

def test_generator_parity():
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "build_multimodel_forecasting_workshop.py"), "--check"],
        cwd=REPO,
        check=True,
    )

def test_notebook_declares_spec_2_1_workshop():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "TASK-INFERENCE"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["clean_runtime_evidence"] == "pending"

def test_notebook_has_required_models_and_forms():
    source = "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])
    for literal in [
        "NX-AI/TiRex-2",
        "amazon/chronos-2",
        "Datadog/Toto-2.0-2.5B",
        'USE_BYOD = False  # @param',
        'BYOD_PATH = ""  # @param',
        'RUN_TIREX = True  # @param',
        'RUN_CHRONOS = True  # @param',
        'RUN_TOTO = True  # @param',
        'HORIZON = 48  # @param',
        "seasonal_naive",
        "selection_partition",
        "multimodel_forecasting_predictions.csv",
        "multimodel_forecasting_provenance.json",
    ]:
        assert literal in source

def test_notebook_avoids_runtime_repo_dependencies():
    source = "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])
    forbidden = [
        "git clone ",
        "pip install -e",
        "raw.githubusercontent.com/kurtvalcorza",
        "dimer-backend",
        "localhost:",
    ]
    for literal in forbidden:
        assert literal not in source

def test_committed_notebook_is_clean():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
