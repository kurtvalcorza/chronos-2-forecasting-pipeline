# ruff: noqa: F811 -- the imported pytest fixtures (ns, scored) are requested by parameter name.
"""Regression checks for the 2026-10-12 reef capstone review (REEF-m1 to REEF-m8).

The refusal checks run the carried stage runner in a separate CPU process and the notebook's own
run_logged()/run_stage() helpers; no model is loaded (refusals happen before the model stage).
These are software contracts, not hosted or model-performance evidence.
"""

import ast
import datetime
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "reef_builder_review_2026_10_12", ROOT / "tools/build_reef_capstone_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notebook():
    return _builder().build()


def _cells(notebook, kind):
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == kind]


def _code(notebook, needle):
    return next(s for s in _cells(notebook, "code") if needle in s)


def _markdown(notebook, needle):
    return next(s for s in _cells(notebook, "markdown") if needle in s)


@pytest.fixture(scope="module")
def carried(notebook):
    space = {}
    for source in _cells(notebook, "code"):
        if source.startswith("# Infrastructure: reef_"):
            exec(source, space)
    return space


@pytest.fixture
def runner(tmp_path, carried):
    """The runner file exactly as the notebook assembles it."""
    path = tmp_path / "reef_runner.py"
    text = "from __future__ import annotations\n" + "".join(
        carried[n] for n in ("CORE_SOURCE", "MODEL_SOURCE", "RUNTIME_SOURCE")
    )
    path.write_text(text, encoding="utf-8")
    return path


def _helpers(notebook, tmp_path):
    """The setup cell's run_logged() and its helpers, as functions only."""
    tree = ast.parse(_code(notebook, "def run_logged"))
    funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"child_environment", "resource_note", "run_logged"}
    ]
    import os
    import shutil
    import time

    space = dict(os=os, subprocess=subprocess, sys=sys, time=time, shutil=shutil)
    space.update(LOG_DIR=tmp_path, RUN_ROOT=tmp_path)
    exec(compile(ast.Module(funcs, []), "setup_cell", "exec"), space)
    return space


def _daily_csv(path, rows=400, change=None, header="region_id,date,hotspot_c"):
    start = datetime.date(2024, 1, 1)
    lines = [header]
    for k in range(rows):
        value = 0.5 if change is None else change(k)
        lines.append(f"r1,{(start + datetime.timedelta(days=k)).isoformat()},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


BAD_FILES = {
    "negative.csv": (
        lambda p: _daily_csv(p, change=lambda k: -0.5 if k == 0 else 0.5),
        "finite nonnegative",
    ),
    "sst_column.csv": (lambda p: _daily_csv(p, header="region_id,date,sst_c"), "Expected"),
    "short.csv": (lambda p: _daily_csv(p, rows=100), "365 continuous"),
    "missing.csv": (lambda p: None, "cannot read"),
}


# REEF-m1: one refusal message, no traceback, no RuntimeError ---------------------------------


@pytest.mark.parametrize("name", sorted(BAD_FILES))
def test_runner_refuses_bad_byod_with_one_message(tmp_path, runner, name):
    make, rule = BAD_FILES[name]
    path = tmp_path / name
    make(path)
    result = subprocess.run(
        [sys.executable, "-u", str(runner), "byod", "--root", str(tmp_path), "--csv", str(path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    shown = result.stdout + result.stderr
    assert result.returncode == 3, shown
    assert "Traceback (most recent call last)" not in shown
    refusals = [line for line in shown.splitlines() if line.startswith("Refused: ")]
    assert len(refusals) == 1 and refusals[0].startswith(f"Refused: {name}: ")
    assert rule in refusals[0]
    assert "re-run the BYOD cell" in shown
    assert not (tmp_path / "byod").exists()  # nothing saved for a refused file


def test_notebook_wrapper_shows_a_refusal_not_a_crash(notebook, tmp_path, runner, capsys):
    helpers = _helpers(notebook, tmp_path)
    bad = BAD_FILES["negative.csv"][0](tmp_path / "negative.csv")
    command = [sys.executable, "-u", runner, "byod", "--root", tmp_path, "--csv", bad]
    assert helpers["run_logged"](command, "stage_byod") is None
    out = capsys.readouterr().out
    assert "Traceback" not in out and out.count("Refused: negative.csv:") == 1
    # A genuine failure of a stage, or a refused-code exit of a non-stage child, still raises.
    with pytest.raises(RuntimeError, match="exit code 1"):
        helpers["run_logged"]([sys.executable, "-c", "raise SystemExit(1)"], "stage_byod")
    with pytest.raises(RuntimeError, match="exit code 3"):
        helpers["run_logged"]([sys.executable, "-c", "raise SystemExit(3)"], "setup_sync")


def test_run_stage_reports_refusal_and_byod_cell_has_no_raise(notebook):
    runner_cell = _code(notebook, "def run_stage")
    assert "if seconds is None:" in runner_cell and "refused the input" in runner_cell
    byod = _code(notebook, "USE_BYOD = False")
    assert "raise" not in byod
    assert "Refused: tick HOTSPOT_SEMANTICS_CONFIRMED" in byod


def test_unconfirmed_semantics_prints_refusal(notebook, capsys):
    byod = _code(notebook, "USE_BYOD = False").replace("USE_BYOD = False", "USE_BYOD = True")
    space = {"run_stage": lambda *a: pytest.fail("must not run"), "pathlib": __import__("pathlib")}
    exec(byod, space)
    assert capsys.readouterr().out.startswith("Refused: tick HOTSPOT_SEMANTICS_CONFIRMED")


# REEF-m2: where BYOD data goes ---------------------------------------------------------------


def test_byod_section_states_locality_inventory_and_warning(notebook):
    text = " ".join(_markdown(notebook, "## 11. Optional: bring your own").split())
    assert "reads your CSV only inside this Colab runtime and sends it to no other service" in text
    for item in ("forecasts.csv", "receipt.json", "stage_byod.log", "Copy path", "/content/"):
        assert item in text
    assert "never added to `reef_evidence.zip`" in text
    assert "Do not use confidential, restricted, personal or regulated data" in text


def test_byod_inventory_matches_the_runner(carried):
    runtime = carried["RUNTIME_SOURCE"]
    byod = runtime[runtime.index("def reef_byod") : runtime.index("def reef_main")]
    assert 'out = root / "byod" / digest[:16]' in byod
    assert set(re.findall(r'out / "([a-z_.]+)"', byod)) == {"forecasts.csv", "receipt.json"}
    report = runtime[runtime.index("def reef_report") : runtime.index("def reef_byod")]
    # The evidence bundle holds root-level files only, so the byod/ folder never enters it.
    assert "for p in root.iterdir()" in report and "p.is_file()" in report


# REEF-m3: opening status and measured figures ------------------------------------------------


def test_opening_status_is_not_stale_and_figures_are_labelled(notebook):
    opening = " ".join(_cells(notebook, "markdown")[0].split())
    assert "pending fresh Colab T4 qualification" not in opening
    assert "unverified until hosted qualification" not in opening
    assert "previous revision (`08b2343`, Colab CLI on a fresh Colab T4, 2026-10-03;" in opening
    assert "not re-run at this revision" in opening
    assert notebook["metadata"]["dimer"]["release_status"] == "candidate"


# REEF-m4, owner decision A: a bounded, validation-only context-length setting -----------------


def test_activity_setting_is_explained_before_use(notebook):
    objectives = _markdown(notebook, "## Roadmap and learning objectives")
    assert "Change one setting (the validation context length)" in objectives
    section = " ".join(_markdown(notebook, "## 6.").split())
    assert section.startswith("## 6. Change one setting: how much history?")
    for phrase in ("**Your control.**", "90, 180 (the default) or 270",
                   "The test in Section 7 always uses 365 days", "**What to notice:**"):
        assert phrase in section
    cells = _cells(notebook, "code")
    field = next(i for i, c in enumerate(cells) if c.startswith("ACTIVITY_CONTEXT_DAYS = 180"))
    assert '# @param [90, 180, 270] {type:"raw"}' in cells[field]
    later = "\n".join(cells[field + 1 :])
    assert "ACTIVITY_CONTEXT_DAYS" not in later and "--context-days" not in later


def test_activity_cell_refuses_other_values_and_later_stages_skip(notebook, capsys):
    cell = _code(notebook, "ACTIVITY_CONTEXT_DAYS = 180").replace(
        "ACTIVITY_CONTEXT_DAYS = 180", "ACTIVITY_CONTEXT_DAYS = 120", 1
    )
    calls = []
    space = {"run_stage": lambda *a: calls.append(a)}
    exec(cell, space)
    out = capsys.readouterr().out
    assert out.startswith("Refused: ACTIVITY_CONTEXT_DAYS=120: choose 90, 180 or 270 days")
    assert space["ACTIVITY_REFUSED"] is True
    assert ("activity", "--context-days", "120") not in calls
    runner_cell = _code(notebook, "def run_stage")
    assert "if ACTIVITY_REFUSED and name in AFTER_ACTIVITY:" in runner_cell
    after = '{"compare", "lock", "test", "score", "future", "reload", "report"}'
    assert f"AFTER_ACTIVITY = {after}" in runner_cell


@pytest.mark.parametrize("args", [["activity", "--context-days", "120"],
                                  ["test", "--context-days", "180"]])
def test_runner_refuses_bad_or_misplaced_context_days(tmp_path, runner, args):
    result = subprocess.run(
        [sys.executable, "-u", str(runner), args[0], "--root", str(tmp_path), *args[1:]],
        capture_output=True,
        text=True,
        timeout=300,
    )
    shown = result.stdout + result.stderr
    assert result.returncode == 3, shown
    assert "Traceback" not in shown
    assert len([line for line in shown.splitlines() if line.startswith("Refused: ")]) == 1


@pytest.fixture(scope="module")
def activity_runs(tmp_path_factory, notebook, monkeypatch_module):
    """Prepare, baselines, validation, activity (90 or default), compare, lock, test, score.

    Model calls are a deterministic persistence double that records its context length.
    """
    space = {"__name__": "reef_activity_double"}
    for source in _cells(notebook, "code"):
        if source.startswith("# Infrastructure: reef_"):
            exec(source, space)
    exec(
        compile(
            "from __future__ import annotations\n"
            + "".join(space[n] for n in ("CORE_SOURCE", "MODEL_SOURCE", "RUNTIME_SOURCE")),
            "carried.py",
            "exec",
        ),
        space,
    )
    lengths = []

    def fake_predict(model, history, horizon=28):
        values = np.asarray(history, dtype=float)
        lengths.append(len(values))
        return np.tile([values[-1] - 0.1, values[-1], values[-1] + 0.1], (horizon, 1))

    space["reef_load_model"] = lambda *a: object()
    space["reef_predict"] = fake_predict
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(max_memory_allocated=lambda: 0)
    monkeypatch_module.setitem(sys.modules, "torch", torch)
    data = ROOT / "tutorials/data/reef"
    manifest = json.loads((data / "manifest.json").read_text())
    runs = {}
    for label, days in (("default", None), ("ninety", 90)):
        root = tmp_path_factory.mktemp(f"activity_{label}")
        shutil.copy(data / "manifest.json", root / "dataset_manifest.json")
        shutil.copy(data / manifest["archive"]["filename"], root)
        shutil.copy(ROOT / "tools/reef_model_manifest.json", root / "model_manifest.json")
        (root / "source_identity.json").write_text('{"runner_sha256": "test-double"}')
        (root / "setup_summary.json").write_text('{"seconds": 1}')
        space["reef_prepare"](root)
        space["reef_experiment"](root, "baselines")
        space["reef_experiment"](root, "validation")
        lengths.clear()
        space["reef_experiment"](root, "activity", days)
        activity_lengths = set(lengths)
        space["reef_compare"](root)
        space["reef_lock"](root)
        lengths.clear()
        space["reef_experiment"](root, "test")
        test_lengths = set(lengths)
        space["reef_score"](root)
        runs[label] = dict(root=root, activity=activity_lengths, test=test_lengths)
    return runs


@pytest.fixture(scope="module")
def monkeypatch_module():
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


def test_default_activity_reproduces_the_180_day_comparison(activity_runs):
    run = activity_runs["default"]
    assert run["activity"] == {180}
    frame = pd.read_csv(run["root"] / "activity_forecasts.csv")
    assert set(frame.arm) == {"chronos_180"}
    text = (run["root"] / "validation_tables.txt").read_text(encoding="utf-8")
    assert "180 minus 365" in text and "vs 180-day activity" in text
    assert "(its final 180 days)" in text
    lock = json.loads((run["root"] / "experiment_lock.json").read_text())
    assert lock["activity_context_days"] == 180


def test_activity_setting_never_reaches_the_test(activity_runs):
    default, ninety = activity_runs["default"], activity_runs["ninety"]
    assert ninety["activity"] == {90} and default["test"] == ninety["test"] == {365}
    lock = json.loads((ninety["root"] / "experiment_lock.json").read_text())
    assert lock["activity_context_days"] == 90
    assert set(pd.read_csv(ninety["root"] / "activity_forecasts.csv").arm) == {"chronos_90"}
    for name in ("test_forecasts.csv", "baselines_forecasts.csv", "validation_forecasts.csv"):
        a = (default["root"] / name).read_bytes()
        b = (ninety["root"] / name).read_bytes()
        assert a == b, name
    ma = json.loads((default["root"] / "metrics.json").read_text())
    mb = json.loads((ninety["root"] / "metrics.json").read_text())
    test_rows = lambda m: [r for r in m["all"]["macro"] if r["split"] == "test"]  # noqa: E731
    assert test_rows(ma) == test_rows(mb)


# REEF-m5 as corrected by the peer review (CX1, CX5, CX7): signed differences, no equivalence ---


def test_interpretation_reports_signed_differences_without_equivalence(notebook, carried):
    section = " ".join(_markdown(notebook, "## 10. Interpret the evidence").split())
    assert "Which arm has the lowest" not in section
    assert "lower or higher MAE **on these evaluated cases**" in section
    assert "does not assess whether a difference is statistically distinguishable or" in section
    assert "`equal_region_difference`" in section and "table E2" in section
    assert "not to seasonal forecasting in general" in section
    for wrong in ("indistinguishable", "no meaningful difference", "0.01 °C"):
        assert wrong not in section
    runtime = carried["RUNTIME_SOURCE"]
    conclusion = runtime[runtime.index('"# Interpretation') : runtime.index("Personal completion")]
    assert "winning arm" not in conclusion.lower()
    assert "describe it only as lower " in conclusion
    assert "or higher MAE on these evaluated cases" in conclusion
    assert "Statistical distinguishability and equivalence " in conclusion
    assert "have not been assessed" in conclusion
    for wrong in ("indistinguishable", "meaningful difference", "0.01"):
        assert wrong not in conclusion


def test_scores_are_labelled_by_estimand(notebook):
    section = " ".join(_markdown(notebook, "## 7. Locked test").split())
    assert '"HotSpot MAE, leads 1–14" averages the errors of every forecast day' in section
    assert '"DHW MAE, lead-14 endpoint" scores DHW on the final day only' in section
    assert "Pinball loss scores each quantile" in section and "more than 80%" in section
    summary = _code(notebook, "def show_summary")
    assert "HotSpot MAE, leads 1–14 | DHW MAE, lead-14 endpoint" in summary


def test_hotspot_is_recorded_in_hundredths(notebook):
    """Recording precision of the snapshot (every HotSpot value is a multiple of 0.01 degC).

    A data fact only: it is not used as an equivalence rule for MAE differences.
    """
    data = ROOT / "tutorials/data/reef"
    manifest = json.loads((data / "manifest.json").read_text())
    archive = zipfile.ZipFile(io.BytesIO((data / manifest["archive"]["filename"]).read_bytes()))
    values = [
        float(parts[7])
        for name in archive.namelist()
        for parts in (line.split() for line in archive.read(name).decode().splitlines())
        if len(parts) == 10 and parts[0].isdigit()
    ]
    assert len(values) > 70000
    assert all(abs(v * 100 - round(v * 100)) < 1e-6 for v in values)


# REEF-m6: re-run rule and a guarded summary --------------------------------------------------


def test_rerun_rule_and_guarded_summary(notebook, tmp_path, capsys, monkeypatch):
    section = " ".join(_markdown(notebook, "## 3. Infrastructure").split())
    assert "Re-running a stage cell clears that stage's results and every later stage's" in section
    display = types.ModuleType("IPython.display")
    for name in ("display", "Markdown", "Image", "FileLink"):
        setattr(display, name, lambda *a, **k: pytest.fail("nothing to display"))
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    summary = _code(notebook, "def show_summary")
    exec(summary, {"RUN_ROOT": tmp_path, "json": json})
    out = capsys.readouterr().out
    assert out.startswith("Results not available yet: metrics.json, outlook.png")
    assert "Runtime → Run all" in out


# REEF-m7: verified uv bootstrap, binary-only install -----------------------------------------


def test_uv_bootstrap_is_pinned_and_verified(notebook):
    setup = _code(notebook, "def run_logged")
    assert '"pip", "install"' not in setup
    assert "uv-0.10.12-py3-none-manylinux_2_17_x86_64" in setup
    assert "101481a1f48db6becf219914a591a588c0b3bfd05bef90768a5d04972bd6455e" in setup
    assert '"--require-hashes",\n                      "--only-binary", ":all:"' in setup
    assert "nothing is installed into the kernel" in setup


def test_uv_bootstrap_refuses_a_tampered_wheel(notebook, tmp_path):
    setup = _code(notebook, "def run_logged")
    block = setup[setup.index("# Pinned uv 0.10.12") : setup.index("def environment_python")]

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("uv-0.10.12.data/scripts/uv", b"not uv")
    fake_request = types.SimpleNamespace(urlopen=lambda *a, **k: Response(wheel.getvalue()))
    space = {
        "sys": types.SimpleNamespace(platform="linux"),
        "platform": types.SimpleNamespace(machine=lambda: "x86_64"),
        "urllib": types.SimpleNamespace(request=fake_request),
        "hashlib": hashlib,
        "io": io,
        "zipfile": zipfile,
        "RUN_ROOT": tmp_path,
    }
    with pytest.raises(RuntimeError, match="size/SHA-256 mismatch"):
        exec(block, space)
    assert not (tmp_path / "uv-0.10.12").exists()


# REEF-m8: the 2025 exclusion reason matches the data -----------------------------------------


def test_2025_exclusion_note_matches_origin_reasons(notebook, carried, tmp_path):
    note = " ".join(_markdown(notebook, "no eligible 2025 origins").split())
    assert "365-day contexts or 28-day outcome windows" in note
    data = ROOT / "tutorials/data/reef"
    manifest = json.loads((data / "manifest.json").read_text())
    runtime_space = {}
    exec(
        compile(
            "from __future__ import annotations\n"
            + carried["CORE_SOURCE"]
            + carried["MODEL_SOURCE"]
            + carried["RUNTIME_SOURCE"],
            "carried.py",
            "exec",
        ),
        runtime_space,
    )
    import shutil

    shutil.copy(data / "manifest.json", tmp_path / "dataset_manifest.json")
    shutil.copy(data / manifest["archive"]["filename"], tmp_path)
    panel = runtime_space["reef_load_data"](tmp_path)
    origins = runtime_space["origin_manifest"](panel)
    late = origins[
        (origins.split == "test")
        & (origins.origin.astype(str).str.startswith("2025"))
        & origins.region_id.str.contains("Western|Southern", case=False)
    ]
    assert len(late) > 0 and not late.eligible.any()
    assert late.reason.str.contains("incomplete_28_day_outcome").any()


# Peer review follow-up (CX2, CX3, CX4, CX6) on a scored run with a model double ---------------


from test_reef_review_fixes import ns, scored  # noqa: E402,F401 -- shared fixtures


def test_table_f_equal_region_difference_matches_table_a(scored, ns):
    reports = json.loads((scored / "metrics.json").read_text())
    summary = ns["reef_paired_summary"](reports["all"])
    macro = {
        (r["period"], r["arm"]): r["hotspot_mae"]
        for r in reports["all"]["macro"]
        if r["split"] == "test" and r["horizon"] == 14 and r["arm"] != "dhw_persistence"
    }
    assert len(summary) == 6  # full, 2024, 2025 x persistence, seasonal
    for row in summary.itertuples():
        expected = macro[(row.period, "chronos")] - macro[(row.period, row.baseline)]
        assert row.equal_region_difference == pytest.approx(expected, abs=1e-8)  # JSON: 10 digits
    assert "pooled_mean_difference" in summary and "mean_difference" not in summary


def test_results_tables_add_labels_counts_and_paired_events(scored):
    text = (scored / "results_tables.txt").read_text(encoding="utf-8")
    for phrase in (
        "HotSpot MAE, leads 1–14",
        "DHW MAE, lead-14 endpoint",
        "target records, leads 1–14",
        "equal_region_difference",
        "pooled_mean_difference",
        "with crossing quantiles (rearranged)",
        "E2. Paired endpoint event predictions",
        "chronos_only",
        "persistence_only",
    ):
        assert phrase in text


def test_paired_event_counts_partition_the_test_origins(scored, ns):
    joined, _plan = ns["reef_joined"](scored)
    table = ns["reef_paired_events"](joined)
    assert len(table) == 4  # horizons 14, 28 x thresholds 4, 8
    for row in table.itertuples():
        assert row.pairs == 184
        assert row.both + row.chronos_only + row.persistence_only + row.neither == row.pairs


def test_compare_does_not_claim_an_unopened_test_set(carried):
    runtime = carried["RUNTIME_SOURCE"]
    assert "still unopened" not in runtime
    assert "Test baseline records exist, but test scores have not been" in runtime
