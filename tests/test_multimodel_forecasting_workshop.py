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
    assert meta["default_tier"] == "STANDARD"

def test_notebook_has_required_models_and_forms():
    source = "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])
    for literal in [
        "NX-AI/TiRex-2",
        "amazon/chronos-2",
        "Datadog/Toto-2.0-2.5B",
        'WORKSHOP_TIER = "STANDARD"  # @param ["STANDARD", "FULL"]',
        "USE_BYOD = False            # @param",
        'BYOD_CSV_PATH = ""          # @param',
        "VALIDATION_HORIZON = 24     # @param",
        "TEST_HORIZON = 24           # @param",
        "SEASON_LENGTH = 24          # @param",
        "6f12f1475411aaab13a2de860ba50f008bd7185a9266f99147c244db4a5de8b3",
        "seasonal_naive",
        "frozen_experiment.json",
        "experiment_manifest.json",
        "workshop_summary.json",
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


def test_canonical_sample_digest_matches_generated_fixture():
    sums = (REPO / "examples" / "sample-data" / "SHA256SUMS.generated").read_text(encoding="utf-8")
    digests = dict(reversed(line.split()) for line in sums.splitlines() if line.strip())
    source = "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])
    assert f'CANONICAL_SAMPLE_SHA256 = "{digests["chronos_multi_series.csv"]}"' in source


def test_embedded_open_meteo_sample_matches_the_committed_byod_file():
    cells = load()["cells"]
    source = next(
        "".join(cell["source"]) for cell in cells if "def open_meteo_ph_bytes" in "".join(cell["source"])
    )
    definitions = source[source.index("OPEN_METEO_PH_SHA256 ="):source.index("def read_checked_csv")]
    import pandas as pd

    namespace = {"pd": pd}
    exec(definitions, namespace)
    sample_dir = REPO / "examples" / "byod-data" / "open-meteo-ph-temperature"
    committed = (sample_dir / "openmeteo_ph_hourly_temperature.csv").read_bytes()
    manifest_digest = (sample_dir / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
    assert namespace["open_meteo_ph_bytes"]() == committed
    assert namespace["OPEN_METEO_PH_SHA256"] == manifest_digest
    assert 'SAMPLE_DATASET = "SYNTHETIC"  # @param ["SYNTHETIC", "OPEN_METEO_PH"]' in "\n".join(
        "".join(cell["source"]) for cell in cells
    )
