"""Regressions for the PR #21 notebook review (R1-R6, V1); CPU only, explicit model double.

Model outputs here come from a deterministic stand-in, never from Chronos. These tests
exercise validation, serialization and presentation boundaries, not forecast quality.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "reef_builder_fixes", ROOT / "tools/build_reef_capstone_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_predict(model, history, horizon=28):
    """Persistence double; above 1 degC its q10 crosses above q50, as Chronos-2 can emit."""
    value = float(np.asarray(history)[-1])
    low = value + 0.05 if value > 1 else value - 0.1
    return np.tile([low, value, value + 0.1], (horizon, 1))


@pytest.fixture(scope="module")
def notebook():
    return _builder().build()


@pytest.fixture(scope="module")
def ns(notebook):
    """Carried notebook source with the model loader/predictor replaced by a test double."""
    space = {"__name__": "reef_review_double"}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("# Infrastructure: reef_"):
            exec(source, space)
    source = "from __future__ import annotations\n" + "".join(
        space[n] for n in ["CORE_SOURCE", "MODEL_SOURCE", "RUNTIME_SOURCE"]
    )
    exec(compile(source, "carried_reef.py", "exec"), space)
    space["reef_load_model"] = lambda *args: object()
    space["reef_predict"] = fake_predict
    return space


def _write_model_stage(ns, root, panel, plan, stage, split, arm, length=365):
    records = []
    subset = plan[plan.split == split]
    for row in subset.itertuples():
        history = ns["context_at"](panel, row.region_id, row.origin, length=length)
        q = fake_predict(None, history)
        records.append(
            ns["forecast_frame"](panel, row.region_id, row.origin, arm, q[:, 1], split, quantiles=q)
        )
    pd.concat(records, ignore_index=True).to_csv(root / f"{stage}_forecasts.csv", index=False)
    summary = {"complete": True, "seconds": 1, "origins": len(subset), "test_double": True}
    (root / f"{stage}_summary.json").write_text(json.dumps(summary))
    if stage == "validation":
        (root / "validation_parity.json").write_text(
            json.dumps({"history": history.tolist(), "quantiles": q.tolist()})
        )


@pytest.fixture(scope="module")
def scored(tmp_path_factory, ns):
    """Complete prepared/locked/scored run on the real NOAA snapshot with a model double."""
    root = tmp_path_factory.mktemp("reef_scored")
    data = ROOT / "tutorials/data/reef"
    manifest = json.loads((data / "manifest.json").read_text())
    shutil.copy(data / "manifest.json", root / "dataset_manifest.json")
    shutil.copy(data / manifest["archive"]["filename"], root)
    shutil.copy(ROOT / "tools/reef_model_manifest.json", root / "model_manifest.json")
    (root / "source_identity.json").write_text('{"runner_sha256": "test-double"}')
    (root / "setup_summary.json").write_text('{"seconds": 1}')
    ns["reef_prepare"](root)
    ns["reef_experiment"](root, "baselines")
    panel, plan = ns["reef_load_data"](root), ns["reef_plan"](root)
    _write_model_stage(ns, root, panel, plan, "validation", "validation", "chronos")
    _write_model_stage(ns, root, panel, plan, "activity", "validation", "chronos_180", 180)
    ns["reef_compare"](root)
    ns["reef_lock"](root)
    ns["reef_experiment"](root, "test")
    ns["reef_score"](root)
    ns["reef_future"](root)
    return root


@pytest.fixture
def run(scored, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(scored, target)
    return target


def _edit(root, stage, change):
    path = root / f"{stage}_forecasts.csv"
    frame = pd.read_csv(path)
    change(frame).to_csv(path, index=False)


# R1 -- runtime entry path ---------------------------------------------------------------


def test_setup_provisions_managed_python_instead_of_rejecting_kernel(notebook):
    setup = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "REQUIREMENTS =" in "".join(c["source"])
    )
    assert "sys.version_info" not in setup
    assert 'TARGET_PYTHON = "3.12.13"' in setup
    assert '"--managed-python"' in setup and '"--python", TARGET_PYTHON' in setup
    assert '"--python", sys.executable' not in setup
    assert "environment_python() != TARGET_PYTHON" in setup
    for key in ("kernel_python", "environment_python", "environment_reused"):
        assert f'"{key}"' in setup
    opening = "".join(notebook["cells"][0]["source"])
    assert "Any current Colab Python version works" in opening


# R2 -- exact forecast grid ----------------------------------------------------------------


@pytest.mark.parametrize(
    "stage,change,message",
    [
        ("test", lambda f: f[f.region_id != "central"], "Missing planned forecasts"),
        ("test", lambda f: f.iloc[:-1], "Missing planned forecasts"),
        (
            "test",
            lambda f: f.assign(target_date=(pd.to_datetime(f.target_date) + pd.Timedelta(days=17))),
            "target_date",
        ),
        ("test", lambda f: f.assign(actual_hotspot_c=f.actual_hotspot_c + 0.5), "frozen source"),
        ("test", lambda f: f.assign(q10=np.nan), "quantiles"),
        ("test", lambda f: f.drop(columns=["raw_q90"]), "required columns"),
        ("test", lambda f: pd.concat([f, f.iloc[:1]]), "Duplicate"),
        ("test", lambda f: f.assign(arm="chronos_v2"), "Missing planned forecasts"),
        ("baselines", lambda f: f[f.arm != "seasonal"], "Missing planned forecasts"),
    ],
)
def test_incomplete_or_misidentified_outputs_refused_before_scores(run, ns, stage, change, message):
    _edit(run, stage, change)
    with pytest.raises(ValueError, match=message):
        ns["reef_metrics"](run)


def test_origin_missing_from_every_arm_is_refused(run, ns):
    """Review fault: a region-origin removed from all arms used to shrink every cohort."""
    drop = ("central", "2024-01-15")

    def remove(frame):
        return frame[~((frame.region_id == drop[0]) & (frame.origin.str[:10] == drop[1]))]

    for stage in ("baselines", "test"):
        _edit(run, stage, remove)
    with pytest.raises(ValueError, match="Missing planned forecasts"):
        ns["reef_metrics"](run)


def test_arm_missing_from_one_region_refused_even_without_plan(ns, scored):
    frame = ns["reef_records"](scored / "test_forecasts.csv")
    base = ns["reef_records"](scored / "baselines_forecasts.csv")
    joined = pd.concat([base, frame[frame.region_id != "central"]], ignore_index=True)
    with pytest.raises(ValueError, match="identical comparison origins"):
        ns["evaluate_forecasts"](joined[joined.split == "test"])


def test_row_permutation_realigns_and_support_matches_frozen_plan(run, ns):
    before = json.loads((run / "metrics.json").read_text())
    _edit(run, "test", lambda f: f.sample(frac=1, random_state=3))
    assert ns["reef_metrics"](run) == before
    macro = pd.DataFrame(before["all"]["macro"])
    chronos = macro[(macro.split == "test") & (macro.arm == "chronos") & (macro.horizon == 14)]
    support = chronos.set_index("period")[["regions", "origins", "days"]].to_dict("index")
    assert support["full"] == {"regions": 5, "origins": 184, "days": 184 * 14}
    assert support["2024"] == {"regions": 5, "origins": 115, "days": 115 * 14}
    assert support["2025"] == {"regions": 3, "origins": 69, "days": 69 * 14}
    validation = macro[(macro.split == "validation") & (macro.horizon == 14)]
    assert set(validation.origins) == {115}


def test_reload_refuses_shifted_dates(run, ns):
    ns["reef_reload"](run)
    assert json.loads((run / "reload_verification.json").read_text())["metrics_reload_parity"]
    _edit(
        run,
        "test",
        lambda f: f.assign(target_date=(pd.to_datetime(f.target_date) + pd.Timedelta(days=17))),
    )
    with pytest.raises(ValueError, match="target_date"):
        ns["reef_reload"](run)


# R3 -- learner-facing diagnostics ---------------------------------------------------------


def test_scoring_prints_support_uncertainty_event_and_paired_tables(scored):
    text = (scored / "results_tables.txt").read_text()
    for header in ("A. ", "B. ", "C. ", "D. ", "E. ", "F. "):
        assert f"\n{header}" in "\n" + text
    for phrase in (
        "Common 2024",
        "2025 (3 regions)",
        "coverage_10_90",
        "width_10_90",
        "pinball_q50",
        "high_stress_mae",
        "new_exceedance",
        "chronos_better",
        "chronos_worse",
        "origins",
    ):
        assert phrase in text
    assert "184" in text and "115" in text and " 69" in text
    assert "..." not in text  # printed with to_string, never truncated


def test_event_table_distinguishes_unavailable_cases(ns):
    base = dict(split="test", period="full", region_id="r", arm="chronos", origins=1, days=1)
    records = [
        dict(base, horizon=14, threshold=8.0, subset="all", n=0, tp=0, fp=0, fn=0, tn=0),
        dict(base, horizon=14, threshold=4.0, subset="all", n=5, tp=0, fp=0, fn=0, tn=5),
        dict(base, horizon=28, threshold=4.0, subset="all", n=5, tp=0, fp=2, fn=0, tn=3),
        dict(base, horizon=28, threshold=8.0, subset="all", n=5, tp=1, fp=1, fn=1, tn=2),
    ]
    table = ns["reef_event_table"](records).set_index(["horizon", "threshold"])
    four, eight = "4 °C-wk", "8 °C-wk"
    assert table.loc[(14, eight), "note"] == "no observations"
    assert table.loc[(14, four), "note"] == "no observed or predicted positives"
    assert np.isnan(table.loc[(14, four), "recall"])
    assert table.loc[(28, four), "note"].startswith("no observed positives")
    assert table.loc[(28, four), "precision"] == 0
    assert table.loc[(28, eight), "note"] == "" and table.loc[(28, eight), "recall"] == 0.5


def test_validation_comparison_and_context_trace(scored):
    text = (scored / "validation_tables.txt").read_text()
    assert "chronos_180" in text and "180 minus 365" in text
    assert "Trace a valid context" in text and "Trace a rejected origin" in text
    assert "incomplete_365_day_context" in text
    assert "test set is still unopened" in text


def test_stage_handoffs_surface_tables(notebook):
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    activity = next(c for c in code if c.startswith('run_stage("activity")'))
    assert activity.splitlines() == [
        'run_stage("activity")',
        'run_stage("compare")',
        'run_stage("lock")',
    ]
    display = next(c for c in code if "Read reports" in c)
    for column in ("Regions", "Origins", "Days"):
        assert column in display
    text = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert "(table A)" in text and "(table E" in text and "(table C)" in text


# R4 -- lock binding and invalidation -------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "validation_forecasts.csv",
        "activity_forecasts.csv",
        "model_manifest.json",
        "seasonal_reference.csv",
    ],
)
def test_test_stage_refuses_changed_locked_inputs(run, ns, name):
    path = run / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after validation"):
        ns["reef_experiment"](run, "test")


def test_restarting_a_stage_invalidates_downstream_records(run, ns):
    for name in ("experiment_lock.json", "test_forecasts.csv", "metrics.json", "outlook.csv"):
        assert (run / name).exists()
    ns["reef_invalidate"](run, "validation")
    for name in (
        "validation_forecasts.csv",
        "experiment_lock.json",
        "test_summary.json",
        "metrics.json",
    ):
        assert not (run / name).exists()
    assert (run / "baselines_forecasts.csv").exists()
    assert (run / "origin_manifest.csv").exists()


# R5 -- forward/BYOD diagnostics and literal identity ---------------------------------------


def test_outlook_keeps_raw_quantiles_and_components(scored):
    outlook = pd.read_csv(scored / "outlook.csv")
    for column in (
        "raw_hotspot_c",
        "clipped_to_zero",
        "raw_q10",
        "q90",
        "known_dhw",
        "predicted_dhw",
    ):
        assert column in outlook
    chronos = outlook[outlook.arm == "chronos"]
    assert chronos[["raw_q10", "raw_q50", "raw_q90"]].notna().all().all()
    assert np.allclose(chronos.raw_q50, chronos.raw_hotspot_c)
    assert outlook[outlook.arm != "chronos"].raw_q50.isna().all()


def test_byod_reads_literal_ids_and_writes_receipt_per_input(run, ns, tmp_path):
    example = pd.read_csv(run / "byod_example.csv", dtype=str)
    first = tmp_path / "ids.csv"
    pd.concat(
        [example.assign(region_id="001"), example.assign(region_id="NA")], ignore_index=True
    ).to_csv(first, index=False)
    frame = ns["read_byod_csv"](first)
    assert set(frame.region_id) == {"001", "NA"}
    out = ns["reef_byod"](run, first)
    forecasts = pd.read_csv(out / "forecasts.csv", dtype={"region_id": str}, keep_default_na=False)
    assert set(forecasts.region_id) == {"001", "NA"} and len(forecasts) == 56
    for column in (
        "raw_hotspot_c",
        "q10",
        "raw_q90",
        "known_dhw",
        "predicted_dhw",
        "clipped_to_zero",
    ):
        assert column in forecasts
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["input_sha256"] == ns["reef_sha"](first)
    assert receipt["units"] == "degC" and receipt["scored"] is False
    assert receipt["model"]["revision"] and receipt["runner_sha256"] == "test-double"
    assert [r["region_id"] for r in receipt["regions"]] == ["001", "NA"]
    second = ns["reef_byod"](run, run / "byod_example.csv")
    assert second != out and (out / "forecasts.csv").exists()


@pytest.mark.parametrize(
    "edit",
    [
        lambda f: f.assign(hotspot_c=""),
        lambda f: f.assign(hotspot_c="warm"),
        lambda f: f.assign(hotspot_c="-1"),
        lambda f: f.iloc[:300],
        lambda f: f.drop(index=200),
    ],
)
def test_byod_invalid_inputs_refused(run, ns, tmp_path, edit):
    path = tmp_path / "bad.csv"
    edit(pd.read_csv(run / "byod_example.csv", dtype=str)).to_csv(path, index=False)
    with pytest.raises(ValueError):
        ns["reef_byod"](run, path)


def test_byod_cell_uses_runner_stage_and_example_default(notebook):
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "".join(c["source"]).startswith("USE_BYOD")
    )
    assert 'run_stage("byod", "--csv"' in cell and "byod_example.csv" in cell
    assert "pd.read_csv" not in cell


# R6 -- figures -----------------------------------------------------------------------------


def test_report_figures_name_frozen_origin_and_exclusions(run, ns, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure

    titles = []
    original = matplotlib.figure.Figure.savefig

    def capture(self, *args, **kwargs):
        titles.append(
            [self._suptitle.get_text() if self._suptitle else ""]
            + [ax.get_title() for ax in self.axes]
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    ns["reef_reload"](run)
    ns["reef_report"](run)
    outlook, _failure, components = titles
    assert "frozen origin 2026-09-26" in outlook[0] and "western" in outlook[0]
    assert all("2026-09-26" in t for t in outlook[1:])
    assert "2026-09-26" in components[1]


# V1 -- JSON-native metrics -----------------------------------------------------------------


def test_json_native_types(ns):
    assert ns["json_native"](np.int64(14)) == 14 and type(ns["json_native"](np.int64(1))) is int
    assert type(ns["json_native"](np.float32(0.5))) is float
    assert ns["json_native"](pd.Timestamp("2024-01-15")) == "2024-01-15"
    with pytest.raises(TypeError):
        ns["json_native"](object())


def test_metrics_round_trip_without_string_fallback(scored, ns):
    reports = ns["reef_metrics"](scored)
    assert json.loads(json.dumps(reports, allow_nan=False)) == reports
    assert json.loads((scored / "metrics.json").read_text()) == reports
    for report in reports.values():
        for table in ("paired_differences", "per_origin_errors", "thresholds", "macro"):
            assert all(type(r["horizon"]) is int for r in report[table])


# Hosted run 2026-09-28: stages run with a clean environment, a log file and a heartbeat ------


def _runner_helpers(notebook, tmp_path):
    setup = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "def run_logged" in "".join(c["source"])
    )
    helpers = setup[
        setup.index("def child_environment") : setup.index("run_logged([sys.executable")
    ]
    assert "def resource_note" in helpers
    import os
    import shutil
    import subprocess
    import sys
    import time

    space = dict(os=os, subprocess=subprocess, sys=sys, time=time, shutil=shutil)
    space.update(LOG_DIR=tmp_path, RUN_ROOT=tmp_path)
    exec(helpers, space)
    return space


def test_child_stages_get_clean_environment_and_logs(notebook, tmp_path, monkeypatch, capsys):
    import sys

    monkeypatch.setenv("PYTHONPATH", "/env/python")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    monkeypatch.setenv("UV_SYSTEM_PYTHON", "1")
    helpers = _runner_helpers(notebook, tmp_path)
    probe = (
        "import os; print(os.environ.get('PYTHONPATH'), os.environ['MPLBACKEND'],"
        " os.environ.get('UV_SYSTEM_PYTHON'))"
    )
    helpers["run_logged"]([sys.executable, "-c", probe], "probe")
    assert "None Agg None" in capsys.readouterr().out
    assert (tmp_path / "probe.log").read_text().strip() == "None Agg None"
    with pytest.raises(RuntimeError, match="exit code 3") as failure:
        helpers["run_logged"](
            [sys.executable, "-c", "print('last words'); raise SystemExit(3)"], "fails"
        )
    assert "last words" in str(failure.value)


def test_setup_and_stages_use_logged_runner(notebook):
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "subprocess.run([sys.executable" not in code
    assert 'run_logged(command, f"stage_{name}")' in code
    assert "still working" in code
    assert '_ = (RUN_ROOT / "setup_summary.json").write_text' in code


# Hosted run 2026-09-28 (2de49e1): Chronos-2 emitted crossing quantiles in the activity stage ----


def test_rearrangement_sorts_and_measures_crossing(ns):
    q = np.array([[0.2, 0.1, 0.3], [0.0, 0.5, 0.4], [0.1, 0.2, 0.3], [0.9, 0.5, 0.1]])
    ordered, crossing = ns["rearrange_quantiles"](q)
    assert (np.diff(ordered, axis=1) >= 0).all()
    assert np.allclose(crossing, [0.1, 0.1, 0.0, 0.8])
    with pytest.raises(ValueError):
        ns["rearrange_quantiles"](np.array([[0.1, np.nan, 0.3]]))


def test_crossing_model_output_is_rearranged_not_refused(scored, ns):
    """The adapter keeps model order; forecast_frame sorts, records and scores it."""

    class Tensor:
        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.tile([0.3, 0.2, 0.4], (1, 28, 1))

    class Model:
        def predict_quantiles(self, inputs, **kwargs):
            return [Tensor()], None

    carried = dict(ns)
    exec(compile(carried["MODEL_SOURCE"], "models.py", "exec"), carried)
    q = carried["reef_predict"](Model(), np.ones(365))
    assert np.allclose(q[0], [0.3, 0.2, 0.4])
    panel, plan = ns["reef_load_data"](scored), ns["reef_plan"](scored)
    row = plan[plan.split == "test"].iloc[0]
    frame = ns["forecast_frame"](
        panel, row.region_id, row.origin, "chronos", q[:, 1], "test", quantiles=q
    )
    assert np.allclose(frame.model_q10, 0.3) and np.allclose(frame.raw_q10, 0.2)
    assert np.allclose(frame.raw_hotspot_c, 0.3) and np.allclose(frame.hotspot_c, 0.3)
    assert np.allclose(frame.quantile_crossing_c, 0.1)


def test_crossings_survive_scoring_reload_and_are_reported(run, ns):
    test = pd.read_csv(run / "test_forecasts.csv")
    assert (test.quantile_crossing_c > 0).any() and (test.quantile_crossing_c == 0).any()
    assert (test[["raw_q10", "raw_q50", "raw_q90"]].diff(axis=1).iloc[:, 1:] >= 0).all().all()
    summary = json.loads((run / "test_summary.json").read_text())
    assert summary["quantile_crossing_days"] == int((test.quantile_crossing_c > 0).sum())
    ns["reef_reload"](run)
    outlook = pd.read_csv(run / "outlook.csv")
    chronos = outlook[outlook.arm == "chronos"]
    assert chronos.quantile_crossing_c.notna().all()
    _edit(run, "test", lambda f: f.assign(model_q10=np.nan))
    with pytest.raises(ValueError, match="quantiles"):
        ns["reef_metrics"](run)


def test_resource_note_and_flush_are_reported(notebook, tmp_path):
    note = _runner_helpers(notebook, tmp_path)["resource_note"]()
    assert "disk free" in note and ("RAM available" in note or "unavailable" in note)
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "os.sync()" in code and code.index("os.sync()") < code.index('run_stage("prepare")')
    assert "started ({resource_note()})" in code and '"resources_after_setup"' in code
