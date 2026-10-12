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


def test_conclusion_asks_independent_questions_without_a_gate():
    conclusion = section("## 8. Conclude with evidence")
    for phrase in (
        "none is a gate for the others",
        "**Does adding these weather features change Mitra's observed error?**",
        "**How do both Mitra variants compare with each predefined baseline?**",
        "**What uncertainty and source limitations constrain these observations?**",
        "Operational use, such as an outbreak warning system, is out of scope",
        "Persistence had MAE ___ and seasonal naïve ___",
        "at the window selected for case-only Mitra",
    ):
        assert phrase in conclusion
    for removed in ("cumulative", "stop at the first", "excludes / includes", "naive floor"):
        assert removed not in conclusion
    text = markdown()
    assert "neither is a pass/fail test" in text
    assert "sizes 4, 4, 4, 4, 4, 4 and 2" in text and "add no independent evidence" in text
    assert "report each model's MAE beside both baselines" in text
    assert "the weaker of the two simple baselines" not in text
    assert "common two-block observation-availability delay" in text
    assert "if anything favours case-only" not in text


def test_byod_text_states_residency_authorisation_and_rerun_procedure():
    text = section("## Infrastructure")
    assert "sends it to no other service" in text
    assert "save or share the executed notebook" in text
    assert "observed references for every validation, test and delay-activity forecast" in text
    assert "the only copy that leaves" not in text
    assert "unless you are authorised to process it in Google Colab" in text
    assert "**Files** pane" in text and "**Runtime > Run all**" in text
    assert "about 780 MB" in text


def test_point_semantics_and_tuning_parity_are_stated():
    text = section("## 3. Compare Chronos")
    assert "0.5 quantile (q50), the median of the model's own forecast distribution" in text
    assert "neither the conditional mean nor the conditional median" in text
    assert "does not turn their predictions into medians" in text
    assert "behave like a median" not in text
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


# Codex peer-review follow-up (CX4, CX8, CX9, CX10).


def test_artifact_manifest_describes_the_whole_export(root, monkeypatch):
    manifest = runtime.read(complete_run(root, monkeypatch) / "artifact_manifest.json")
    text = manifest["retained_data"]
    assert "contexts only" not in text and "No individual records" in text
    assert "observed aggregate counts (references) and predictions" in text


def test_future_input_audit_is_derived_from_source_indices(root, monkeypatch):
    runtime.prepare(root)
    audit = (root / "results" / "availability_audit.csv").read_text(encoding="utf-8")
    assert ",True" not in audit and audit.count(",False") == 208
    real = runtime.core.feature_sources

    def leaky(rows, origin, horizon, delay=0, weather=False):
        sources = real(rows, origin, horizon, delay, weather)
        return sources + [("leak", [origin + 1])] if weather else sources

    monkeypatch.setattr(runtime.core, "feature_sources", leaky)
    plan = runtime.read(root / "results" / "plan.json")
    rows = runtime.read(root / "data.json")
    assert runtime.future_input_used(rows, plan, "test", plan["test"][0], 1) is True


def test_hard_targets_show_every_system_with_reference_and_high_case_flag(root, monkeypatch):
    out = complete_run(root, monkeypatch)
    hard = (out / "hard_targets.csv").read_text(encoding="utf-8").splitlines()
    assert hard[0] == (
        "target_key,horizon,origin_key,reference,high_case,system,prediction,error,abs_error"
    )
    body = [line.split(",") for line in hard[1:]]
    assert len(body) == 3 * len(runtime.SYSTEMS)
    for start in range(0, len(body), len(runtime.SYSTEMS)):
        block = body[start : start + len(runtime.SYSTEMS)]
        assert [r[5] for r in block] == list(runtime.SYSTEMS)
        assert len({(r[0], r[1], r[3], r[4]) for r in block}) == 1
    source = cell("if run('test')")
    assert "table('hard_targets.csv'" in source


def test_worked_example_numbers_are_labelled_as_quoted():
    text = section("## 2. Build a fair chronological comparison")
    assert "recorded 2026-10-03 Colab CLI T4 run of revision `08b2343`" in text
    assert "quoted, not re-run here" in text
    assert "|15 − 11| = 4" in text and "20.50 with a 52-block window and 21.01" in text
