"""CPU runtime checks; foundation-model calls are explicit synthetic doubles."""

import copy
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dengue_runtime as runtime  # noqa: E402


@pytest.fixture
def rows():
    return [
        {
            "year": 2010 + i // 52,
            "block": i % 52 + 1,
            "cases": float(10 + i % 52),
            "rain": float(i % 11),
            "temp": 27.0,
        }
        for i in range(16 * 52)
    ]


@pytest.fixture
def root(tmp_path, rows, monkeypatch):
    (tmp_path / "results").mkdir()
    runtime.write(tmp_path / "data.json", rows)
    runtime.write(tmp_path / "dataset_audit.json", {"scope": runtime.SCOPE})
    runtime.write(tmp_path / "model_manifest.json", {"synthetic_test": True})
    runtime.write(tmp_path / "source.json", {"synthetic_test": True})
    (tmp_path / "DATA_LICENSE.md").write_text("Synthetic fixture; no source data.")
    monkeypatch.setattr(runtime.data, "acquire", lambda root: runtime.read(root / "data.json"))
    monkeypatch.setattr(runtime.models, "load_chronos", lambda *args: object())
    monkeypatch.setattr(runtime.models, "load_mitra", lambda *args: object())
    monkeypatch.setattr(
        runtime.models,
        "chronos_predict",
        lambda model, history, h: np.tile([history[-1] - 1, history[-1], history[-1] + 1], (h, 1)),
    )
    monkeypatch.setattr(
        runtime.models,
        "mitra_predict",
        lambda model, X, y, query: np.repeat(np.mean(y), len(query)),
    )
    monkeypatch.setattr(runtime, "free_gpu", lambda: None)
    return tmp_path


def test_ridge_constant_target_and_safe_state_roundtrip(tmp_path):
    X = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
    y = np.log1p([7, 7, 7])
    query = [1000, 1]
    pred, state = runtime.fit_ridge(X, y, query)
    assert pred[0] == pytest.approx(np.log(8))
    assert state["mean"] == [2, 1]
    assert state["scale"][1] == 1
    runtime.write(tmp_path / "state.json", state)
    np.testing.assert_array_equal(
        pred, runtime.ridge_predict(runtime.read(tmp_path / "state.json"), query)
    )


def test_simple_delay_and_future_labels(rows):
    pred, state = runtime.simple(rows, [len(rows) - 1], delay=2, save=True)
    assert len(pred) == 12 and len(state) == 4
    for p in pred:
        assert p["reference"] is None
        assert p["target_key"].startswith("2026-B")
        assert p["cutoff"] == len(rows) - 3
        if p["system"] == "persistence":
            assert p["prediction"] == rows[-3]["cases"]


def test_scores_known_answer_and_missing_horizon():
    predictions = [
        {"system": name, "horizon": h, "reference": 10.0, "prediction": prediction}
        for name, prediction in [("seasonal", 12.0), ("ridge", 11.0)]
        for h in range(1, 5)
    ]
    scores = runtime.scores(predictions)
    assert scores["ridge"]["mae"] == 1
    assert scores["ridge"]["seasonal_skill"] == 0.5
    with pytest.raises(ValueError):
        runtime.scores(predictions[:-1])


def test_cohort_missing_and_duplicate_refused(rows):
    predictions = [
        runtime.prediction(rows, system, 700, h, 0, 10.0)
        for system in runtime.SYSTEMS
        for h in range(1, 5)
    ]
    runtime.assert_cohort(predictions, [700])
    with pytest.raises(ValueError):
        runtime.assert_cohort(predictions[:-1], [700])
    predictions[-1] = copy.deepcopy(predictions[-2])
    with pytest.raises(ValueError):
        runtime.assert_cohort(predictions, [700])


def test_lock_rejects_source_change(root):
    runtime.prepare(root)
    runtime.write(root / "results" / "selection.json", {"window": 52})
    runtime.lock(root)
    assert runtime.selected_window(root) == 52
    runtime.write(root / "source.json", {"changed": True})
    with pytest.raises(ValueError, match="source changed"):
        runtime.selected_window(root)


def test_synthetic_all_stages_reload_and_report(root, monkeypatch):
    # These are function-level integration checks, not hosted execution evidence.
    for name in runtime.STAGES[:7]:
        getattr(runtime, name)(root)
    out = root / "results"
    plan = runtime.read(out / "plan.json")
    assert len(plan["validation"]) == len(plan["test"]) == 26
    assert runtime.read(out / "selection.json")["window"] in (52, 104)
    predictions = runtime.read(out / "test_predictions.json")
    assert len(predictions) == 624
    runtime.assert_cohort(predictions, plan["test"])
    assert all(p["reference"] is None for p in runtime.read(out / "future_predictions.json"))
    with pytest.raises(ValueError, match="fresh process"):
        runtime.reload(root)
    manifest = runtime.read(out / "artifact_manifest.json")
    monkeypatch.setattr(runtime.os, "getpid", lambda: manifest["creator_pid"] + 1)
    runtime.reload(root)
    assert runtime.read(out / "verification.json")["predictions_checked"] == 32
    # Receipt plumbing is separately tested via main. No hosted receipts fabricated here.
    for stage in runtime.STAGES[:-1]:
        runtime.write(out / f"receipt_{stage}.json", {"synthetic_test": True})
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "synthetic-test")
    runtime.report(root)
    assert (out / "results.zip").stat().st_size > 0
    csv = out / "predictions.csv"
    text = csv.read_text()
    for field, value in [
        ("prediction", "9999"),
        ("reference", "9999"),
        ("system", "wrong_system"),
        ("target_key", "1900-B01"),
    ]:
        lines = text.splitlines()
        header, row = lines[0].split(","), lines[1].split(",")
        row[header.index(field)] = value
        lines[1] = ",".join(row)
        csv.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="CSV"):
            runtime.report(root)
    csv.write_text(text)


def test_artifact_tamper_refused(root, monkeypatch):
    runtime.prepare(root)
    runtime.write(root / "results" / "selection.json", {"window": 52})
    runtime.lock(root)
    runtime.future(root)
    out = root / "results"
    manifest = runtime.read(out / "artifact_manifest.json")
    monkeypatch.setattr(runtime.os, "getpid", lambda: manifest["creator_pid"] + 1)
    path = out / "artifact_future_ridge.json"
    state = runtime.read(path)
    state["1"]["state"]["coef"][0] += 1
    runtime.write(path, state)
    with pytest.raises(ValueError, match="hash mismatch"):
        runtime.reload(root)


def test_main_receipt_refuses_changed_plan(root, monkeypatch):
    monkeypatch.setattr(runtime, "figures", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["dengue_runtime", "--root", str(root), "--stage", "prepare"])
    runtime.main()
    out = root / "results"
    plan = runtime.read(out / "plan.json")
    plan["test"] = plan["test"][:-1]
    runtime.write(out / "plan.json", plan)
    monkeypatch.setattr(
        sys, "argv", ["dengue_runtime", "--root", str(root), "--stage", "baselines"]
    )
    with pytest.raises(ValueError, match="Changed stage output"):
        runtime.main()


@pytest.mark.parametrize(
    "field,value", [("reference", 9999), ("target_key", "1900-B01"), ("cutoff", 699), ("delay", 2)]
)
def test_cohort_rejects_cross_system_reference_and_timing_tamper(rows, field, value):
    predictions = [
        runtime.prediction(rows, system, 700, h, 0, 10.0)
        for system in runtime.SYSTEMS
        for h in range(1, 5)
    ]
    predictions[-1][field] = value
    with pytest.raises(ValueError, match="disagreement"):
        runtime.assert_cohort(predictions, [700], rows)


def test_cohort_rejects_consistent_but_false_references(rows):
    predictions = [
        runtime.prediction(rows, system, 700, h, 0, 10.0)
        for system in runtime.SYSTEMS
        for h in range(1, 5)
    ]
    for p in predictions:
        p["reference"] = 9999
    with pytest.raises(ValueError, match="match source"):
        runtime.assert_cohort(predictions, [700], rows)


def test_identity_rejects_changed_embedded_file(root):
    file = root / "embedded.py"
    file.write_text("original")
    runtime.write(root / "source.json", {"files": {file.name: runtime.sha(file)}})
    runtime.identity(root)
    file.write_text("changed")
    with pytest.raises(ValueError, match="Embedded source changed"):
        runtime.identity(root)


def test_identity_rejects_changed_audited_data(root):
    runtime.write(
        root / "dataset_audit.json", {"derived_data_sha256": runtime.sha(root / "data.json")}
    )
    runtime.identity(root)
    records = runtime.read(root / "data.json")
    records[0]["cases"] += 1
    runtime.write(root / "data.json", records)
    with pytest.raises(ValueError, match="audited source"):
        runtime.identity(root)


def test_byod_rejects_extra_personal_column(root, rows):
    path = root / "byod.csv"
    runtime.table(path, [dict(r, patient_name="synthetic") for r in rows])
    runtime.write(
        root / "run_config.json",
        {
            "byod_csv": str(path),
            "source_blocks_confirmed": True,
            "area": "Synthetic test area",
            "source_citation": "Synthetic fixture",
        },
    )
    with pytest.raises(ValueError, match="extra personal fields"):
        runtime.prepare(root)


def test_byod_rejects_five_years(root, rows):
    path = root / "byod.csv"
    runtime.table(path, rows[: 5 * 52])
    runtime.write(
        root / "run_config.json",
        {
            "byod_csv": str(path),
            "source_blocks_confirmed": True,
            "area": "Synthetic test area",
            "source_citation": "Synthetic fixture",
        },
    )
    with pytest.raises(ValueError, match="six complete"):
        runtime.prepare(root)


def test_byod_requires_source_attestation(root, rows):
    path = root / "byod.csv"
    runtime.table(path, rows)
    runtime.write(root / "run_config.json", {"byod_csv": str(path)})
    with pytest.raises(ValueError, match="confirmation"):
        runtime.prepare(root)


def test_byod_six_years_preserves_fixed_cohorts(root, rows):
    path = root / "byod.csv"
    runtime.table(path, rows[: 6 * 52])
    runtime.write(
        root / "run_config.json",
        {
            "byod_csv": str(path),
            "source_blocks_confirmed": True,
            "area": "Synthetic test area",
            "source_citation": "Synthetic fixture",
        },
    )
    runtime.prepare(root)
    plan = runtime.read(root / "results" / "plan.json")
    assert len(plan["validation"]) == len(plan["test"]) == 26
    assert plan["validation_years"] == [2012, 2013]
