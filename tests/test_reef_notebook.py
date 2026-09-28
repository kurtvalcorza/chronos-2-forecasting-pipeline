"""Integration checks for carried code and real NOAA data; no GPU/model evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "reef_builder", ROOT / "tools/build_reef_capstone_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def carried():
    nb = load_builder().build()
    space = {"__name__": "reef_contract_test"}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, cell["id"], "exec")
            if source.startswith("# Infrastructure: reef_"):
                exec(source, space)
    source = "from __future__ import annotations\n" + "".join(
        space[n] for n in ["CORE_SOURCE", "MODEL_SOURCE", "RUNTIME_SOURCE"]
    )
    exec(compile(source, "carried_reef.py", "exec"), space)
    return space


@pytest.fixture
def data_root(tmp_path, carried):
    import shutil

    data = ROOT / "tutorials/data/reef"
    manifest = json.loads((data / "manifest.json").read_text())
    shutil.copy(data / "manifest.json", tmp_path / "dataset_manifest.json")
    shutil.copy(data / manifest["archive"]["filename"], tmp_path)
    shutil.copy(ROOT / "tools/reef_model_manifest.json", tmp_path / "model_manifest.json")
    (tmp_path / "source_identity.json").write_text("{}")
    carried["reef_prepare"](tmp_path)
    return tmp_path


def test_generated_notebook_is_deterministic_and_valid():
    import nbformat

    nb = load_builder().build()
    nbformat.validate(nbformat.from_dict(nb))
    committed = json.loads(
        (ROOT / "tutorials/DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb").read_text(
            encoding="utf-8"
        )
    )
    assert nb == committed
    assert nb == load_builder().build()
    assert all(c.get("outputs", []) == [] for c in nb["cells"])
    all_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for forbidden in (
        "git clone",
        "pip install -e",
        "raw.githubusercontent.com",
        "import chronos2_pipeline",
    ):
        assert forbidden not in all_code


def test_real_snapshot_counts_and_calendar(data_root, carried):
    origins = pd.read_csv(data_root / "origin_manifest.csv")
    counts = origins[origins.eligible].groupby("split").size().to_dict()
    assert counts == {"test": 184, "validation": 115}
    audit = json.loads((data_root / "dataset_audit.json").read_text())
    assert sorted(r["missing_days"] for r in audit["regions"]) == [0, 0, 0, 25, 25]
    assert all(r["dhw_max_error"] < 1e-4 for r in audit["regions"])


def test_corrupt_archive_refused(data_root, carried):
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    path = data_root / manifest["archive"]["filename"]
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="integrity"):
        carried["reef_load_data"](data_root)


def test_quantile_adapter_contract(carried):
    class Tensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.tile([0.1, 0.2, 0.3], (1, 28, 1))

    class Model:
        def predict_quantiles(self, inputs, **kwargs):
            assert len(inputs) == 1 and kwargs["context_length"] == 365
            return [Tensor()], None

    result = carried["reef_predict"](Model(), np.ones(365))
    assert result.shape == (28, 3)


def test_real_baselines_and_serialized_evaluation(data_root, carried):
    carried["reef_experiment"](data_root, "baselines")
    frame = carried["reef_records"](data_root / "baselines_forecasts.csv")
    assert len(frame) == 299 * 2 * 28
    report = carried["evaluate_forecasts"](frame, expected_arms=["persistence", "seasonal"])
    json.dumps(report, allow_nan=False)
    assert len(report["macro"]) > 0
    with pytest.raises(ValueError, match="arm"):
        carried["evaluate_forecasts"](frame, expected_arms=["persistence", "seasonal", "chronos"])


def test_optional_byod_program_compiles():
    nb = load_builder().build()
    source = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "".join(c["source"]).startswith("USE_BYOD")
    )
    # Compile nested program by reading its AST literal, without executing optional model work.
    import ast

    tree = ast.parse(source)
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "pd.read_csv(source)" in n.value
    ]
    assert len(literals) == 1
    compile(literals[0], "byod.py", "exec")


def test_export_reload_integration_with_explicit_model_double(data_root, carried, monkeypatch):
    """Exercise serialization and charts; this is deliberately not real model evidence."""

    def fake_predict(model, history, horizon=28):
        value = float(np.asarray(history)[-1])
        return np.tile([value - 0.1, value, value + 0.1], (horizon, 1))

    monkeypatch.setitem(carried, "reef_load_model", lambda *args: object())
    monkeypatch.setitem(carried, "reef_predict", fake_predict)
    carried["reef_experiment"](data_root, "baselines")
    panel = carried["reef_load_data"](data_root)
    origins = carried["origin_manifest"](panel)
    for stage, split, arm in [
        ("validation", "validation", "chronos"),
        ("activity", "validation", "chronos_180"),
        ("test", "test", "chronos"),
    ]:
        records = []
        subset = origins[origins.eligible & (origins.split == split)]
        for row in subset.itertuples():
            history = carried["context_at"](panel, row.region_id, row.origin)
            q = fake_predict(None, history)
            records.append(
                carried["forecast_frame"](
                    panel, row.region_id, row.origin, arm, q[:, 1], split, quantiles=q
                )
            )
        pd.concat(records, ignore_index=True).to_csv(
            data_root / f"{stage}_forecasts.csv", index=False
        )
        (data_root / f"{stage}_summary.json").write_text(
            json.dumps(
                {"complete": True, "seconds": 1, "origins": len(subset), "test_double": True}
            )
        )
        if stage == "validation":
            (data_root / "validation_parity.json").write_text(
                json.dumps({"history": history.tolist(), "quantiles": q.tolist()})
            )
    (data_root / "setup_summary.json").write_text('{"seconds": 1}')
    carried["reef_lock"](data_root)
    carried["reef_score"](data_root)
    carried["reef_future"](data_root)
    carried["reef_reload"](data_root)
    carried["reef_report"](data_root)
    summary = json.loads((data_root / "run_summary.json").read_text())
    assert summary["reload"]["metrics_reload_parity"]
    assert (data_root / "reef_evidence.zip").stat().st_size > 0
    assert len(json.loads((data_root / "outlook_exclusions.json").read_text())) == 2
    # Changing an exported metric must be detected by an independent recomputation.
    (data_root / "metrics.json").write_text("{}")
    with pytest.raises(AssertionError, match="metric reconstruction"):
        carried["reef_reload"](data_root)


def test_budget_refuses_test_unlock(data_root, carried):
    for stage in ["validation", "activity", "baselines"]:
        (data_root / f"{stage}_summary.json").write_text(
            json.dumps({"complete": True, "seconds": 1801, "origins": 115})
        )
    (data_root / "setup_summary.json").write_text('{"seconds": 0}')
    with pytest.raises(RuntimeError, match="resource budget"):
        carried["reef_lock"](data_root)
    assert not (data_root / "experiment_lock.json").exists()
