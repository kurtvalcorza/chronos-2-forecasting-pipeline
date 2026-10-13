"""Acceptance checks for the 2026-10-12 workshop review findings (TSW2-m1 … TSW2-m8).

Static checks read the generated notebook. Behaviour checks execute the notebook's own cells with
the fake runner from ``test_forecasting_workshop_review_fixes``; they are orchestration checks, not
model runs or Colab evidence.
"""

from __future__ import annotations

import json
import re
import zipfile

import pandas as pd
import pytest

from test_forecasting_workshop_review_fixes import (  # noqa: F401  (plots is a fixture, used via usefixtures)
    CELLS,
    NOTEBOOK,
    install_fake_models,
    panel_bytes,
    plots,
    run_cell,
    start,
    through_freeze,
    to_the_end,
)

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
MARKDOWN = [
    "".join(cell["source"]) for cell in NB["cells"] if cell["cell_type"] == "markdown"
]
ALL_MARKDOWN = "\n".join(MARKDOWN)


def _markdown_with(needle):
    return [text for text in MARKDOWN if needle in text]


# --- TSW2-m1: quoted numbers name the run that produced them


def test_quoted_open_meteo_answers_and_timings_name_their_source_revision():
    blocks = [
        text.split("**Open-Meteo sample", 1)[1]
        for text in _markdown_with("**Open-Meteo sample (")
    ]
    assert len(blocks) == 2
    for block in blocks:
        assert "`9e10f24`" in block and "tirex_cpu_rerun_0.2.0.json" in block
        assert "not re-run with this revision" in block
    timings = _markdown_with("# 10. Compare accuracy with cost")[0]
    assert "`8cda8d5`" in timings and "`4199536`" in timings
    assert "In the reference Colab T4 run" not in timings
    assert "Sample answers quote the reference runs on a Colab T4" not in ALL_MARKDOWN


# --- TSW2-m2: the model environments' versions, device and precision are displayed


@pytest.mark.usefixtures("plots")
def test_validation_prints_each_model_environment(tmp_path, monkeypatch, capsys):
    ns = install_fake_models(start(tmp_path, monkeypatch))
    capsys.readouterr()
    run_cell("7.1 Forecast the validation period with each model", ns)
    out = capsys.readouterr().out
    assert "Model environments as observed in the validation run" in out
    table = out.split("Model environments as observed in the validation run", 1)[1]
    for column in ("python", "device", "precision", "torch", "model_package"):
        assert column in table
    for name in ns["SELECTED_MODELS"]:
        assert ns["MODEL_SPECS"][name]["display_name"] in table
    assert "float32" in table and "fake 0" in table
    assert "Section 7.1" in _markdown_with("Participant submission checklist")[0]


# --- TSW2-m3 (option B): the freeze records a validation-based choice; Section 9 checks it


def test_prose_describes_a_recorded_choice():
    claim = "Choose the models, the horizons and the metrics using the validation period"
    assert claim not in ALL_MARKDOWN
    assert "The choice the freeze records" in ALL_MARKDOWN
    assert "CHOSEN_MODEL" in _markdown_with("Participant submission checklist")[0]


@pytest.mark.usefixtures("plots")
def test_default_choice_is_frozen_and_checked_on_the_test(tmp_path, monkeypatch, capsys):
    ns = through_freeze(start(tmp_path, monkeypatch))
    frozen = json.loads(
        (ns["OUTPUT_ROOT"] / "frozen" / "frozen_experiment.json").read_text(encoding="utf-8")
    )
    choice = frozen["validation_choice"]
    aggregate = ns["validation_aggregate"].set_index("model")["macro_mae"]
    assert choice["chosen_model"] == choice["validation_ranking"][0]
    assert aggregate[choice["validation_ranking"][0]] <= aggregate[choice["validation_ranking"][1]]
    capsys.readouterr()
    run_cell("9.1 Check the freeze, then forecast the test period", ns)
    out = capsys.readouterr().out
    assert "Frozen choice: " in out and "Test ranking:" in out
    check = json.loads(
        (ns["OUTPUT_ROOT"] / "test" / "validation_choice_check.json").read_text(encoding="utf-8")
    )
    assert check["chosen_model"] == choice["chosen_model"]
    assert set(check) >= {"choice_confirmed", "ranking_held", "test_ranking"}


@pytest.mark.usefixtures("plots")
def test_a_changed_choice_after_the_freeze_stops_the_test(tmp_path, monkeypatch):
    ns = through_freeze(start(tmp_path, monkeypatch))
    other = ns["foundation_ranking"](ns["validation_aggregate"])[1]
    ns["CHOSEN_MODEL"] = other
    with pytest.raises(RuntimeError, match="validation_choice"):
        run_cell("9.1 Check the freeze, then forecast the test period", ns)


@pytest.mark.usefixtures("plots")
def test_an_unknown_choice_is_refused_with_guidance(tmp_path, monkeypatch):
    ns = install_fake_models(start(tmp_path, monkeypatch))
    run_cell("7.1 Forecast the validation period with each model", ns)
    with pytest.raises(ValueError, match="is not one of this run's models"):
        run_cell(
            "8.1 Freeze the experiment",
            ns,
            {'CHOSEN_MODEL = "lowest validation macro MAE"': 'CHOSEN_MODEL = "Toto 2.0 2.5B"'},
        )


# --- TSW2-m4: the input copy is written after validation and can be left out of the report


@pytest.mark.usefixtures("plots")
def test_input_copy_is_written_only_after_validation_passes(tmp_path, monkeypatch):
    assert "validated_panel_source.csv" not in CELLS["3.1 Generate or load the sample"]
    monthly = pd.DataFrame({
        "series_id": ["A"] * 90,
        "timestamp": pd.date_range("2018-01-01", periods=90, freq="MS").strftime("%Y-%m-%d"),
        "target": range(90),
    }).to_csv(index=False).encode()
    with pytest.raises(ValueError, match="fixed-width step"):
        start(tmp_path, monkeypatch, monthly)
    assert not list(tmp_path.glob("outputs/*/data/validated_panel_source.csv"))


@pytest.mark.parametrize("include", [True, False])
@pytest.mark.usefixtures("plots")
def test_report_input_copy_follows_the_opt_out(tmp_path, monkeypatch, include):
    ns = start(tmp_path, monkeypatch, panel_bytes())
    ns["REPORT_INCLUDES_INPUT_DATA"] = include
    to_the_end(through_freeze(ns))
    bundle = next(ns["EXPERIMENTS_ROOT"].glob("*_Report.zip"))
    names = zipfile.ZipFile(bundle).namelist()
    inventory = json.loads(
        (ns["OUTPUT_ROOT"] / "provenance" / "inventory.json").read_text(encoding="utf-8")
    )
    listed = {item["path"] for item in inventory["files"]}
    assert ("data/validated_panel_source.csv" in names) is include
    assert ("data/validated_panel_source.csv" in listed) is include
    assert "data/dataset_manifest.json" in names


def test_report_contents_are_disclosed():
    assert re.search(r"report ZIP[^.]{0,120}copy of your input", ALL_MARKDOWN)
    control = 'REPORT_INCLUDES_INPUT_DATA = True  # @param {type:"boolean"}'
    assert control in CELLS["1.1 Notebook controls"]


# --- TSW2-m5: fixed-width steps only; horizons and season length in the data's own steps


def test_byod_text_and_message_name_fixed_width_steps():
    byod = _markdown_with("## Bring your own data (optional)")[0]
    assert "fixed-width" in byod and "Calendar months" in byod
    assert "counted in your data's own steps" in byod
    assert "SEASON_LENGTH = 7" in byod
    assert "hourly (or other regularly spaced)" not in ALL_MARKDOWN


# --- TSW2-m6: recovery messages name the full rerun range


def test_recovery_messages_say_rerun_from_section_1_1_onward():
    messages = []
    titles = ("8.1 Freeze the experiment", "9.1 Check the freeze, then forecast the test period")
    for title in titles:
        messages += re.findall(r"[Rr]erun[^\"']*Section 1\.1[^\"']*", CELLS[title])
    assert len(messages) == 4
    assert all("onward" in message for message in messages)


# --- TSW2-m7: the comparison is scoped as untuned whole configurations


def test_comparison_scope_is_stated():
    assert "use it **the same way**" not in ALL_MARKDOWN
    assert "zero-shot and untuned, at its upstream default settings" in ALL_MARKDOWN
    assert "whole configurations" in ALL_MARKDOWN
    assert "Coverage moved nearer to 0.80" not in ALL_MARKDOWN
    assert "too few, and too correlated" in ALL_MARKDOWN


# --- TSW2-m8 (option A): NOTEBOOK_SPEC 2.5 declaration and write-up layer


def test_spec_2_5_declaration_and_write_up_layer():
    meta = NB["metadata"]["dimer"]
    assert meta["notebook_kind"] == "WORKSHOP"
    assert meta["notebook_spec"] == "2.5"
    assert "Notebook Specification 2.5" in MARKDOWN[0]
    for title in ("8.1 Freeze the experiment", "13.2 Run-all completion summary"):
        assert '"notebook_spec": "2.5"' in CELLS[title]
    assert '"spec": "2.5"' in CELLS["13.1 Export the experiment record"]
    first_code = next(i for i, cell in enumerate(NB["cells"]) if cell["cell_type"] == "code")
    opening = "\n".join("".join(cell["source"]) for cell in NB["cells"][:first_code])
    assert "## Problem statement" in opening
    assert "**Out of scope.**" in opening and "**What a useful outcome looks like.**" in opening
    assert "## Data context" in ALL_MARKDOWN and "# What did not work" in ALL_MARKDOWN
    assert "> **⚠️ Notice.** For research, training and evaluation only." in opening
    assert re.search(r"^## References$", ALL_MARKDOWN, re.M)
    for arxiv in ("2510.15821", "2607.01204", "2605.20119"):
        assert f"https://arxiv.org/abs/{arxiv}" in ALL_MARKDOWN
