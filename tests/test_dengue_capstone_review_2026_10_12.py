# ruff: noqa: F811 -- the imported pytest fixtures (root, rows) are requested by parameter name.
"""Regression checks for the 2026-10-12 capstone review (DNG-M1, DNG-m1 to DNG-m9).

The refusal checks run the embedded runtime in a separate CPU process and the notebook's own
run()/record() helpers; model calls are synthetic doubles. These are software contracts, not
hosted or model-performance evidence.
"""

import ast
import json
import os
import shutil
import subprocess
import sys

import pytest

from test_dengue_review_fixes import (  # noqa: F401 -- shared fixtures and helpers
    MODULES,
    TOOLS,
    byod,
    cell,
    complete_run,
    csv_text,
    notebook,
    root,
    rows,
    runtime,
    series,
)

BAD_TEMP_LINE = (21, "2010,21,20,50.0,500")  # data line 21 is file line 22


def carrier_functions(*names):
    """The notebook's own helper definitions and constants, taken from the carrier cell."""
    tree = ast.parse(cell("def run(stage)"))
    wanted = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) in ("REFUSED_EXIT", "REFUSED") for t in n.targets)
        )
    ]
    return ast.Module(wanted, [])


def refusal_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    for name in MODULES:
        shutil.copy(TOOLS / name, root / name)
    path = root / "bad.csv"
    path.write_text(csv_text(series(8), BAD_TEMP_LINE), encoding="utf-8")
    runtime.write(
        root / "run_config.json",
        {
            "byod_csv": str(path),
            "source_blocks_confirmed": True,
            "area": "Synthetic",
            "source_citation": "fixture",
        },
    )
    return root


def child_env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)
    return env


# DNG-m3 -- a refused BYOD file is one actionable message, never a traceback.


def test_runtime_refusal_is_one_line_with_distinct_exit_code(tmp_path):
    root = refusal_root(tmp_path)
    result = subprocess.run(
        [sys.executable, "-u", str(root / "dengue_runtime.py"), "--root", str(root),
         "--stage", "prepare"],
        capture_output=True, text=True, env=child_env(), cwd=root, timeout=300,
    )
    output = (result.stdout + result.stderr).strip().splitlines()
    assert result.returncode == runtime.REFUSED_EXIT == 3
    assert output == [
        "Refused: bad.csv line 22, column temp: outside the declared Celsius range -30 to 60"
    ]


def test_year_count_refusal_names_the_file(root):
    byod(root, csv_text(series(5)))
    with pytest.raises(runtime.InputRefused, match=r"^byod\.csv: "):
        runtime.prepare(root)


def test_notebook_run_prints_refusal_once_and_skips_later_stages(tmp_path, capsys):
    root = refusal_root(tmp_path)
    namespace = {
        "ROOT": root,
        "PYTHON": sys.executable,
        "ENV": child_env(),
        "subprocess": subprocess,
    }
    exec(compile(carrier_functions("run"), "notebook-run", "exec"), namespace)
    assert namespace["run"]("prepare") is False
    assert namespace["run"]("baselines") is False
    shown = capsys.readouterr().out
    assert "Traceback" not in shown and "RuntimeError" not in shown
    assert shown.count("Refused: bad.csv line 22, column temp") == 1
    assert "Runtime > Run all" in shown
    assert "Skipped baselines: the input was refused in prepare" in shown
    assert not (root / "baselines.log").exists()


def test_every_stage_cell_continues_only_after_a_passing_stage():
    for c in notebook()["cells"]:
        if c["cell_type"] != "code" or "def run(stage)" in c["source"]:
            continue
        for line in c["source"].splitlines():
            if "run('" in line:
                assert line.startswith("if run('"), line


# DNG-m7 -- record() never truncates silently; the summary lists each limitation once.


def test_record_labels_an_excerpt_and_keeps_short_files_whole(tmp_path, capsys):
    (tmp_path / "results").mkdir()
    runtime.write(tmp_path / "results" / "small.json", {"a": 1})
    runtime.write(tmp_path / "results" / "big.json", {"x": "y" * 20000})
    namespace = {"json": json, "ROOT": tmp_path, "FileLink": str, "display": lambda value: None}
    exec(compile(carrier_functions("record"), "notebook-record", "exec"), namespace)
    namespace["record"]("small.json")
    assert capsys.readouterr().out == json.dumps({"a": 1}, indent=2) + "\n"
    namespace["record"]("big.json")
    shown = capsys.readouterr().out
    assert "[truncated excerpt: first 14,000 of 20," in shown
    assert "the full file is linked below" in shown and len(shown) < 14200


def test_run_summary_lists_each_limitation_topic_once(root, monkeypatch):
    out = complete_run(root, monkeypatch)
    audit = runtime.read(root / "dataset_audit.json")
    audit["limitations"] = runtime.read(TOOLS / "dengue_dataset_audit.json")["limitations"]
    runtime.write(root / "dataset_audit.json", audit)  # the pinned source's own audit list
    runtime.report(root)
    summary = runtime.read(out / "run_summary.json")
    items = [x.lower() for x in summary["limitations"]]
    assert len(items) == len(set(items))
    assert sum("pretraining overlap" in x for x in items) == 1
    assert sum("calendar" in x for x in items) == 1
    assert sum("zero" in x and "delay" in x for x in items) == 1
    assert any("centroid weather cells" in x for x in items)  # audit-only items are kept


# DNG-m6 -- progress lines say which window and delay they belong to.


def test_progress_lines_name_window_and_delay(root, capsys):
    series_rows = runtime.read(root / "data.json")
    runtime.model_forecasts(root, series_rows, [10 * 52 - 1], 52, "mitra_cases")
    # Chronos always reads 104 blocks, whatever window the stage passes.
    runtime.model_forecasts(root, series_rows, [10 * 52 - 1], 52, "chronos", delay=2)
    shown = capsys.readouterr().out
    assert "mitra_cases (window 52, delay 0): origin 1/1" in shown
    assert "chronos (window 104, delay 2): origin 1/1" in shown


# DNG-M1, m1, m2, m4, m5, m8, m9 -- learner-facing text and declaration.


def markdown():
    return "\n".join(
        c["source"] for c in notebook()["cells"] if c["cell_type"] == "markdown"
    )


def section(heading):
    cells = [c["source"] for c in notebook()["cells"] if c["cell_type"] == "markdown"]
    return next(text for text in cells if heading in text)


def test_conclusion_reads_weather_against_the_naive_floor_and_claim_levels():
    conclusion = section("## 8. Conclude with evidence")
    for phrase in (
        "**Signal: does Mitra + weather beat the naive floor?**",
        "**Comparator: does weather beat case-only?**",
        "**Operational: would it help a real warning system?** Out of scope",
        "persistence (the naive floor) had MAE ___",
        "with a window tuned for case-only Mitra",
        "An interval that includes zero means no weather gain is established",
    ):
        assert phrase in conclusion
    assert "it does not establish a weather gain either" in markdown()
    assert "Skill here is relative to seasonal naïve" in markdown()


def test_byod_text_states_residency_authorisation_and_rerun_procedure():
    text = section("## Infrastructure")
    assert "nothing is sent to any other service" in text
    assert "unless you are authorised to process it in Google Colab" in text
    assert "**Files** pane" in text and "**Runtime > Run all**" in text
    assert "about 780 MB" in text


def test_point_semantics_and_tuning_parity_are_stated():
    text = section("## 3. Compare Chronos")
    assert "0.5 quantile, the median" in text and "log1p(count)" in text
    assert "selected on case-only Mitra and reused for weather Mitra" in text
    assert "Chronos (fixed 104-block context) and Ridge use fixed settings" in text


def test_declaration_keys_and_write_up_layer():
    nb = notebook()
    dimer = nb["metadata"]["dimer"]
    assert {k: dimer.get(k) for k in ("notebook_spec", "notebook_profile", "notebook_mode",
                                      "notebook_kind")} == {
        "notebook_spec": "2.2", "notebook_profile": "E2E", "notebook_mode": "GUIDED",
        "notebook_kind": "CAPSTONE",
    }
    assert not {"spec_version", "profile", "mode"} & set(dimer)
    opening = nb["cells"][0]["source"]
    assert nb["cells"][1]["cell_type"] == "code"
    assert "## Problem statement" in opening and "**Use notice:**" in opening
    assert "not validated for production" in opening and '"as is"' in opening
    final = nb["cells"][-1]["source"]
    assert "## Troubleshooting" in final and "## References" in final
    assert len(nb["cells"]) == 19 and nb["cells"][-1]["id"] == "dengue-18"


def test_opening_states_measured_runtime_and_download_size():
    opening = notebook()["cells"][0]["source"]
    assert "measurements are pending" not in opening
    assert "about 780 MB of model weights" in opening
    assert "took 445 seconds end to end, with 0.48 GiB peak GPU allocation" in opening
