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
import subprocess
import sys
import types
import zipfile
from pathlib import Path

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


# REEF-m4: the activity is described as pre-set ----------------------------------------------


def test_activity_is_described_as_preset(notebook):
    objectives = _markdown(notebook, "## Roadmap and learning objectives")
    assert "Change one setting" not in objectives
    section = " ".join(_markdown(notebook, "## 6.").split())
    assert section.startswith("## 6. A pre-set comparison")
    assert "it is not a setting you change here" in section


# REEF-m5: claim levels and the 0.01 degC rule ------------------------------------------------


def test_interpretation_uses_claim_levels(notebook, carried):
    section = " ".join(_markdown(notebook, "## 10. Interpret the evidence").split())
    assert "Which arm has the lowest" not in section
    assert "no meaningful difference at this precision" in section
    assert "not independent" in section
    conclusion = carried["RUNTIME_SOURCE"]
    assert "winning arm" not in conclusion.lower()
    assert "no meaningful difference at this" in conclusion


def test_hotspot_is_recorded_in_hundredths(notebook):
    """The 0.01 degC rule rests on the data: every HotSpot value is a multiple of 0.01."""
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
