"""Acceptance checks for the Notebook Review Framework v1 forecasting findings (TS-R01 … TS-R10).

The tests execute the workshop notebook's own cells. Pretrained models are replaced by a tiny fake
runner program with the same command-line and file contract, run through the notebook's real
``run_model``. These are orchestration and contract checks, not model runs or Colab evidence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb"
CELLS = {
    "".join(cell["source"]).splitlines()[0].removeprefix("# @title "): "".join(cell["source"])
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    if cell["cell_type"] == "code"
}
NOTEBOOK_TEXT = "\n".join(
    "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
)

# The last lines of cell 6.1, which install real environments; tests replace them.
ENVIRONMENT_BUILD = (
    "ensure_uv()\nMODEL_PYTHONS = {name: ensure_environment(name) for name in SELECTED_MODELS}"
)

FAKE_RUNNER = r"""
import argparse, json
import pandas as pd
ap = argparse.ArgumentParser()
for flag in ("--input", "--output", "--metadata", "--cache", "--partition", "--freq"):
    ap.add_argument(flag)
ap.add_argument("--horizon", type=int)
args = ap.parse_args()
print("[stage] fake model forecasting", flush=True)
frame = pd.read_csv(
    args.input, parse_dates=["timestamp"], dtype={"series_id": str}, keep_default_na=False
)
ids = list(frame["series_id"].drop_duplicates())
delta = frame["timestamp"].drop_duplicates().sort_values().diff().dropna().iloc[0]
rows = []
for sid in ids:
    block = frame[frame["series_id"] == sid].sort_values("timestamp")
    origin, last = block["timestamp"].max(), float(block["target"].iloc[-1])
    for step in range(1, args.horizon + 1):
        rows.append({"model": MODEL, "partition": args.partition, "forecast_origin": origin,
                     "series_id": sid, "timestamp": origin + delta * step, "step": step,
                     "q0.1": last - 1, "q0.5": last, "q0.9": last + 1, "prediction": last})
pd.DataFrame(rows).to_csv(args.output, index=False)
json.dump({"model": MODEL, "device": "cpu", "dtype": "float32", "packages": {"fake": "0"},
           "python": "test",
           "conditioning": {"mode": "independent-per-series", "target_groups": GROUPS},
           "load_seconds": 0.0, "inference_seconds": 0.0}, open(args.metadata, "w"))
"""


class Recorder:
    """A stand-in for matplotlib.pyplot that records every plotted x/y array."""

    def __init__(self):
        self.lines = []

    def subplots(self, *args, **kwargs):
        return self, self

    def plot(self, x, y, *args, **kwargs):
        self.lines.append((pd.to_datetime(pd.Series(list(x))), np.asarray(list(y), dtype=float)))

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def run_cell(title, ns, replacements=None):
    source = CELLS[title]
    for old, new in (replacements or {}).items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, title, "exec"), ns)
    return ns


def panel_bytes(ids=("A", "B"), hours=96):
    steps = np.arange(hours)
    return (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "series_id": sid,
                        "timestamp": pd.date_range("2026-01-01", periods=hours, freq="h").strftime(
                            "%Y-%m-%dT%H:%M:%S"
                        ),
                        "target": 50 + 10 * k + steps * 0.1 + np.sin(steps * 2 * np.pi / 24),
                    }
                )
                for k, sid in enumerate(ids)
            ]
        )
        .to_csv(index=False)
        .encode()
    )


@pytest.fixture
def plots(monkeypatch):
    recorder = Recorder()
    # The CI image has no matplotlib; the stub stands in for it, so report a tested version for it.
    import importlib.metadata

    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "3.10.0" if name == "matplotlib" else real_version(name),
    )
    package = types.ModuleType("matplotlib")
    package.pyplot = recorder
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", recorder)
    return recorder


def start(tmp_path, monkeypatch, payload=None, tier="STANDARD", sample="SYNTHETIC"):
    """Sections 1.1 to 5.1: controls, runtime, data, EDA and baselines."""
    monkeypatch.chdir(tmp_path)
    ns = {"display": lambda *a, **k: None}
    replacements = {
        'WORKSHOP_TIER = "STANDARD"  # @param': f'WORKSHOP_TIER = "{tier}"  # @param',
        'SAMPLE_DATASET = "SYNTHETIC"  # @param': f'SAMPLE_DATASET = "{sample}"  # @param',
    }
    run_cell("1.1 Notebook controls", ns, replacements)
    if tier == "FULL":
        with monkeypatch.context() as m:
            m.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
            m.setattr("shutil.disk_usage", lambda path: types.SimpleNamespace(free=200e9))
            m.setattr(
                "subprocess.run",
                lambda *a, **k: subprocess.CompletedProcess(a, 0, "Tesla T4, 15360\n", ""),
            )
            run_cell("1.2 Check the notebook runtime", ns)
    else:
        run_cell("1.2 Check the notebook runtime", ns)
    run_cell("2.1 Pinned model identities and footprint", ns)
    if payload is not None:
        path = tmp_path / f"byod-{ns['EXPERIMENT_ID']}.csv"
        path.write_bytes(payload)
        ns["BYOD_CSV_PATH"] = str(path)
    run_cell("3.1 Generate or load the sample", ns)
    run_cell("3.2 Check the data before any model download", ns)
    run_cell("4.1 Plot the development period and mark the hidden test period", ns)
    run_cell("5.1 Split the data, build the baselines and define the metrics", ns)
    return ns


def install_fake_models(ns, groups="[[sid] for sid in ids]"):
    """Section 6: real pins and runner cell, with each runner replaced by the fake program."""
    run_cell(
        "6.1 Build one environment per model",
        ns,
        {
            ENVIRONMENT_BUILD: "MODEL_PYTHONS = {n: Path(sys.executable) for n in SELECTED_MODELS}",
        },
    )
    run_cell("6.2 Write the model runner programs", ns)
    for name in ns["SELECTED_MODELS"]:
        source = f"MODEL = {ns['MODEL_SPECS'][name]['display_name']!r}\n" + FAKE_RUNNER.replace(
            "GROUPS}", groups + "}"
        )
        (ns["RUN_WORK_ROOT"] / "runners" / f"{name}_runner.py").write_text(source, encoding="utf-8")
        ns["RUNNER_SHA256"][name] = ns["file_sha"](
            ns["RUN_WORK_ROOT"] / "runners" / f"{name}_runner.py"
        )
        marker = ns["WORK_ROOT"] / "envs" / name / ".dimer_workshop_ready"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("signature-" + name, encoding="utf-8")
    return ns


def through_freeze(ns):
    install_fake_models(ns)
    run_cell("7.1 Forecast the validation period with each model", ns)
    run_cell("7.2 Plot the validation forecasts", ns)
    run_cell("8.1 Freeze the experiment", ns)
    return ns


def to_the_end(ns):
    run_cell("9.1 Check the freeze, then forecast the test period", ns)
    run_cell("9.2 Plot the test forecasts", ns)
    run_cell("10.1 Compare model size and measured runtime", ns)
    run_cell("11.1 Forecast 24 hours beyond the supplied data", ns)
    run_cell("13.1 Export the experiment record", ns)
    run_cell("13.2 Run-all completion summary", ns)
    return ns


# --- TS-R01: nothing before the freeze shows the test targets


@pytest.mark.parametrize("sample", ["SYNTHETIC", "OPEN_METEO_PH", "BYOD"])
def test_pre_freeze_eda_plots_and_summaries_exclude_the_test_period(
    tmp_path, monkeypatch, plots, sample
):
    payload = panel_bytes() if sample == "BYOD" else None
    ns = start(tmp_path, monkeypatch, payload, sample="SYNTHETIC" if sample == "BYOD" else sample)
    test_start = ns["test_truth"]["timestamp"].min()
    assert plots.lines
    for timestamps, _ in plots.lines:
        assert timestamps.max() < test_start
    development_rows = int((ns["frame"]["timestamp"] < test_start).sum())
    assert ns["EDA_ROWS_SHOWN"] == development_rows
    assert int(ns["development_summary"]["count"].sum()) == development_rows
    # The chronological split itself still keeps each history before its truth.
    for sid in ns["panel"]["series_ids"]:
        history = ns["test_history"][ns["test_history"]["series_id"] == sid]
        assert history["timestamp"].max() < test_start


def test_first_baseline_table_scores_validation_only(tmp_path, monkeypatch, plots):
    ns = start(tmp_path, monkeypatch)
    assert set(ns["baseline_validation_aggregate"]["model"]) == {"last_value", "seasonal_naive"}
    macro = ns["baseline_validation_aggregate"].set_index("model")["macro_mae"]
    assert macro["seasonal_naive"] == pytest.approx(3.60)
    assert macro["last_value"] == pytest.approx(5.5348, abs=1e-4)
    assert not list(ns["OUTPUT_ROOT"].glob("test/*"))


# --- TS-R02: one conditioning policy, reported by every model and checked


def _runner_source(tmp_path, monkeypatch, plots, name):
    ns = start(tmp_path, monkeypatch)
    run_cell(
        "6.1 Build one environment per model",
        ns,
        {
            ENVIRONMENT_BUILD: "MODEL_PYTHONS = {}",
        },
    )
    run_cell("6.2 Write the model runner programs", ns)
    return ns, ns["RUNNERS"][name]


def test_tirex_forecasts_each_series_as_its_own_univariate_item(tmp_path, monkeypatch, plots):
    ns, source = _runner_source(tmp_path, monkeypatch, plots, "tirex")
    received = []

    class FakeTimeseries:
        def __init__(self, target, past_covariates, future_covariates):
            self.target = target

    class FakeModel:
        def forecast(self, items, prediction_length, output_type):
            received.extend(items)
            # Each item's forecast depends only on its own history, as a univariate model would.
            return [
                np.full((len(item.target), 9, prediction_length), float(item.target[0, -1]))
                for item in items
            ]

    monkeypatch.setitem(
        sys.modules,
        "tirex2",
        types.SimpleNamespace(
            load_model=lambda *a, **k: FakeModel(), TimeseriesType=FakeTimeseries
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=lambda array: array))
    runner = {"__name__": "tirex_runner"}
    exec(source, runner)
    runner["stage"] = lambda root: root
    out, meta = tmp_path / "tirex.csv", tmp_path / "tirex.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            str(ns["REQUEST_PATHS"]["validation"]),
            "--output",
            str(out),
            "--metadata",
            str(meta),
            "--cache",
            str(tmp_path),
            "--horizon",
            "24",
            "--partition",
            "validation",
        ],
    )
    runner["main"]()
    assert len(received) == ns["panel"]["n_series"]
    assert all(item.target.shape == (1, 48) for item in received)
    reported = json.loads(meta.read_text())["conditioning"]
    assert reported["mode"] == "independent-per-series"
    assert reported["target_groups"] == [["A"], ["B"]]


def test_chronos_is_called_without_cross_learning(tmp_path, monkeypatch, plots):
    ns, source = _runner_source(tmp_path, monkeypatch, plots, "chronos")
    calls = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def predict_df(self, frame, **kwargs):
            calls.update(kwargs)
            rows = []
            for sid, block in frame.groupby("series_id", sort=False):
                origin = block["timestamp"].max()
                for step in range(1, kwargs["prediction_length"] + 1):
                    rows.append(
                        {
                            "series_id": sid,
                            "timestamp": origin + pd.Timedelta(hours=step),
                            "target_name": "target",
                            "predictions": 1.0,
                            "0.1": 0.0,
                            "0.5": 1.0,
                            "0.9": 2.0,
                        }
                    )
            return pd.DataFrame(rows)

    monkeypatch.setitem(
        sys.modules, "chronos", types.SimpleNamespace(BaseChronosPipeline=FakePipeline)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    runner = {"__name__": "chronos_runner"}
    exec(source, runner)
    runner["stage"] = lambda root: root
    out, meta = tmp_path / "chronos.csv", tmp_path / "chronos.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            str(ns["REQUEST_PATHS"]["validation"]),
            "--output",
            str(out),
            "--metadata",
            str(meta),
            "--cache",
            str(tmp_path),
            "--horizon",
            "24",
            "--partition",
            "validation",
            "--freq",
            "h",
        ],
    )
    runner["main"]()
    assert calls["cross_learning"] is False
    assert json.loads(meta.read_text())["conditioning"]["target_groups"] == [["A"], ["B"]]


def test_toto_gives_every_series_its_own_id_and_checks_cuda_before_download(
    tmp_path, monkeypatch, plots
):
    _, source = _runner_source(tmp_path, monkeypatch, plots, "toto")
    assert "series_ids=torch.arange(values.shape[0]" in source
    assert source.index("torch.cuda.is_available()") < source.index("root=stage(Path(args.cache))")
    assert '"mode":"independent-per-series"' in source


def test_a_model_reporting_joint_conditioning_is_refused(tmp_path, monkeypatch, plots):
    ns = start(tmp_path, monkeypatch)
    install_fake_models(ns, groups="[ids]")
    with pytest.raises(RuntimeError, match="did not report the common conditioning"):
        run_cell("7.1 Forecast the validation period with each model", ns)


# --- TS-R03: identifiers stay text through every file


def test_numeric_and_leading_zero_ids_survive_the_whole_workflow(tmp_path, monkeypatch, plots):
    ns = start(tmp_path, monkeypatch, panel_bytes(ids=("001", "1", "101", "A")))
    assert ns["panel"]["series_ids"] == ["001", "1", "101", "A"]
    to_the_end(through_freeze(ns))
    for table in (ns["validation_scored"], ns["test_scored"], ns["future_forecasts"]):
        assert set(table["series_id"]) == {"001", "1", "101", "A"}
    exported = pd.read_csv(
        ns["OUTPUT_ROOT"] / "test/forecasts_with_truth.csv", dtype={"series_id": str}
    )
    assert set(exported["series_id"]) == {"001", "1", "101", "A"}


def test_series_named_like_missing_values_are_kept(tmp_path, monkeypatch, plots):
    ns = start(tmp_path, monkeypatch, panel_bytes(ids=("NA", "null")))
    assert ns["panel"]["series_ids"] == ["NA", "null"]


def test_a_forecast_with_integer_ids_is_rejected_not_silently_repaired(
    tmp_path, monkeypatch, plots
):
    ns = start(tmp_path, monkeypatch, panel_bytes(ids=("101", "202")))
    table = ns["validation_seasonal"].copy()
    table["series_id"] = table["series_id"].astype(int)
    with pytest.raises(ValueError, match="series_id must be text"):
        ns["evaluate_forecasts"](
            table,
            ns["validation_truth"],
            ns["validation_seasonal"],
            ["seasonal_naive"],
            "validation",
        )


# --- TS-R04: the evaluator enforces the complete, coherent forecast contract


def _model_table(ns):
    table = ns["validation_seasonal"].copy()
    table["model"] = "M"
    table["q0.1"] = table["prediction"] - 1.0
    table["q0.9"] = table["prediction"] + 1.0
    return table


def _corrupt(table, kind):
    table = table.copy()
    if kind == "missing_row":
        return table.drop(index=5)
    if kind == "missing_series":
        return table[table["series_id"] != "B"]
    if kind == "unexpected_series":
        extra = table[table["series_id"] == "B"].assign(series_id="C")
        return pd.concat([table, extra], ignore_index=True)
    if kind == "duplicate_row":
        return pd.concat([table, table.iloc[[3]]], ignore_index=True)
    if kind == "shifted_timestamp":
        table.loc[7, "timestamp"] = table.loc[7, "timestamp"] + pd.Timedelta(minutes=30)
        return table
    if kind == "renumbered_step":
        table.loc[2, "step"] = 99
        return table
    if kind == "non_finite_quantile":
        table.loc[4, "q0.9"] = np.inf
        return table
    if kind == "missing_quantile":
        table.loc[4, "q0.1"] = np.nan
        return table
    if kind == "crossed_bounds":
        table.loc[6, ["q0.1", "q0.9"]] = table.loc[6, ["q0.9", "q0.1"]].to_numpy()
        return table
    if kind == "median_not_point":
        table.loc[8, "q0.5"] = table.loc[8, "prediction"] + 0.5
        return table
    raise ValueError(kind)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing_row", "1 expected row\\(s\\) missing"),
        ("missing_series", "24 expected row\\(s\\) missing"),
        ("unexpected_series", "24 unexpected row"),
        ("duplicate_row", "duplicate forecast rows"),
        ("shifted_timestamp", "forecast grid is incomplete"),
        ("renumbered_step", "step numbers"),
        ("non_finite_quantile", "non-finite"),
        ("missing_quantile", "non-finite"),
        ("crossed_bounds", "crossed quantiles"),
        ("median_not_point", "point forecast must equal q0.5"),
    ],
)
def test_invalid_forecast_tables_are_rejected_before_scoring(
    tmp_path, monkeypatch, plots, kind, message
):
    ns = start(tmp_path, monkeypatch)
    table = pd.concat(
        [_corrupt(_model_table(ns), kind), ns["validation_seasonal"]], ignore_index=True
    )
    with pytest.raises(ValueError, match=message):
        ns["evaluate_forecasts"](
            table,
            ns["validation_truth"],
            ns["validation_seasonal"],
            ["M", "seasonal_naive"],
            "validation",
        )


def test_a_missing_model_is_rejected(tmp_path, monkeypatch, plots):
    ns = start(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="expected forecasts from"):
        ns["evaluate_forecasts"](
            ns["validation_seasonal"],
            ns["validation_truth"],
            ns["validation_seasonal"],
            ["M", "seasonal_naive"],
            "validation",
        )


def test_a_complete_valid_table_keeps_its_metrics(tmp_path, monkeypatch, plots):
    ns = start(tmp_path, monkeypatch)
    table = pd.concat([_model_table(ns), ns["validation_seasonal"]], ignore_index=True)
    _, per_series, aggregate = ns["evaluate_forecasts"](
        table,
        ns["validation_truth"],
        ns["validation_seasonal"],
        ["M", "seasonal_naive"],
        "validation",
    )
    row = aggregate.set_index("model").loc["M"]
    assert row["macro_mae"] == pytest.approx(3.60)
    assert row["mean_interval_width"] == pytest.approx(2.0)
    assert row["skill_series"] == 2
    # Equal-length series: pooled and macro MAE are the same number (TS-R08).
    assert row["pooled_mae"] == pytest.approx(row["macro_mae"])


# --- TS-R05: the freeze binds the staged inputs and everything else the test depends on


@pytest.mark.parametrize(
    "change",
    [
        "request_value",
        "request_timestamp",
        "runner",
        "season_length",
        "environment",
        "context_limit",
    ],
)
def test_any_change_after_the_freeze_stops_the_test_before_a_model_call(
    tmp_path, monkeypatch, plots, change
):
    ns = through_freeze(start(tmp_path, monkeypatch))
    request = ns["REQUEST_PATHS"]["test"]
    if change == "request_value":
        staged = pd.read_csv(request)
        staged.loc[5, "target"] += 1.0
        staged.to_csv(request, index=False)
    elif change == "request_timestamp":
        request.write_text(
            request.read_text().replace("2026-01-03 23:00:00", "2026-01-03 23:30:00", 1)
        )
    elif change == "runner":
        runner = ns["RUN_WORK_ROOT"] / "runners" / "tirex_runner.py"
        runner.write_text(runner.read_text() + "\n# edited\n")
    elif change == "season_length":
        ns["SEASON_LENGTH"] = 12
    elif change == "environment":
        (ns["WORK_ROOT"] / "envs" / "chronos" / ".dimer_workshop_ready").write_text("rebuilt")
    else:
        ns["COMMON_MAX_CONTEXT"] = 64
    calls = []
    ns["run_model"] = lambda *a, **k: calls.append(a)
    with pytest.raises(RuntimeError, match="no longer matches the freeze"):
        run_cell("9.1 Check the freeze, then forecast the test period", ns)
    assert calls == []


def test_the_freeze_is_kept_and_the_test_runs_once(tmp_path, monkeypatch, plots):
    ns = through_freeze(start(tmp_path, monkeypatch))
    freeze = ns["freeze_path"].read_bytes()
    run_cell("8.1 Freeze the experiment", ns)
    assert ns["freeze_path"].read_bytes() == freeze
    ns["SEASON_LENGTH"] = 12
    with pytest.raises(RuntimeError, match="already frozen with different settings"):
        run_cell("8.1 Freeze the experiment", ns)
    ns["SEASON_LENGTH"] = 24
    run_cell("9.1 Check the freeze, then forecast the test period", ns)
    with pytest.raises(RuntimeError, match="already been scored"):
        run_cell("9.1 Check the freeze, then forecast the test period", ns)


def test_a_request_changed_just_before_the_call_is_refused(tmp_path, monkeypatch, plots):
    ns = through_freeze(start(tmp_path, monkeypatch))
    with pytest.raises(RuntimeError, match="request file .* changed"):
        ns["run_model"]("tirex", "test", ns["REQUEST_PATHS"]["test"], 24, "0" * 64)


# --- TS-R06: every report holds only its own experiment


def test_each_report_contains_only_its_own_experiment(tmp_path, monkeypatch, plots):
    full = to_the_end(
        through_freeze(start(tmp_path, monkeypatch, tier="FULL", sample="OPEN_METEO_PH"))
    )
    run_cell(
        "Optional validation-only seasonal-period comparison",
        full,
        {"RUN_SEASONAL_ACTIVITY = False": "RUN_SEASONAL_ACTIVITY = True"},
    )
    full_report = (
        full["EXPERIMENTS_ROOT"]
        / f"{full['EXPERIMENT_ID']}_DIMER_TimeSeries_FM_Workshop_Report.zip"
    )
    full_bytes = full_report.read_bytes()

    standard = to_the_end(through_freeze(start(tmp_path, monkeypatch)))
    assert standard["OUTPUT_ROOT"] != full["OUTPUT_ROOT"]
    report = (
        standard["EXPERIMENTS_ROOT"]
        / f"{standard['EXPERIMENT_ID']}_DIMER_TimeSeries_FM_Workshop_Report.zip"
    )
    with zipfile.ZipFile(report) as archive:
        names = set(archive.namelist())
        inventory = json.loads(archive.read("provenance/inventory.json"))
        summary = json.loads(archive.read("workshop_summary.json"))
    assert not any("toto" in name for name in names)
    assert not any(name.startswith("activities/") for name in names)
    assert names == {item["path"] for item in inventory["files"]} | {"provenance/inventory.json"}
    assert summary["models"] == ["TiRex-2", "Chronos-2"]
    assert summary["experiment_id"] == standard["EXPERIMENT_ID"]
    # The earlier experiment and its report are preserved unchanged.
    assert full_report.read_bytes() == full_bytes
    assert (full["OUTPUT_ROOT"] / "test/toto.csv").exists()
    assert (full["OUTPUT_ROOT"] / "activities/seasonal_period_comparison.csv").exists()


def test_a_rerun_export_includes_an_activity_marked_optional(tmp_path, monkeypatch, plots):
    ns = to_the_end(through_freeze(start(tmp_path, monkeypatch)))
    run_cell(
        "Optional validation-only seasonal-period comparison",
        ns,
        {"RUN_SEASONAL_ACTIVITY = False": "RUN_SEASONAL_ACTIVITY = True"},
    )
    run_cell("13.1 Export the experiment record", ns)
    inventory = json.loads((ns["OUTPUT_ROOT"] / "provenance/inventory.json").read_text())
    flagged = {item["path"]: item["optional_activity"] for item in inventory["files"]}
    assert flagged["activities/seasonal_period_comparison.csv"] is True
    assert flagged["test/aggregate_metrics.csv"] is False


# --- TS-R07 / TS-R10: honest status and a complete reproducibility record


def test_forecast_beyond_the_data_is_labelled_not_evaluated(tmp_path, monkeypatch, plots):
    ns = to_the_end(through_freeze(start(tmp_path, monkeypatch)))
    assert (
        ns["FUTURE_FORECAST_STATUS"] == "not evaluated: matching future targets were not provided"
    )
    assert ns["future_forecasts"]["timestamp"].min() == pd.Timestamp("2026-01-05 00:00")
    assert "Forecast the real future" not in NOTEBOOK_TEXT
    assert "has not happened yet" not in NOTEBOOK_TEXT


def test_the_manifest_records_notebook_identity_pins_and_observed_versions(
    tmp_path, monkeypatch, plots
):
    ns = to_the_end(through_freeze(start(tmp_path, monkeypatch)))
    manifest = json.loads((ns["OUTPUT_ROOT"] / "provenance/experiment_manifest.json").read_text())
    assert manifest["notebook"]["revision"] == ns["NOTEBOOK_REVISION"] != "0.1.0-candidate"
    assert manifest["notebook"]["file"] == NOTEBOOK.name
    assert manifest["parent_runtime"]["host_versions"]["pandas"] == pd.__version__
    tirex = manifest["models"]["tirex"]
    assert "tirex-2==0.2.1" in tirex["environment"]["intended_pins"]
    assert tirex["observed"]["test"]["packages"] == {"fake": "0"}
    assert tirex["observed"]["test"]["input_sha256"] == ns["REQUEST_SHA256"]["test"]
    freeze = json.loads(ns["freeze_path"].read_text())
    assert freeze["models"]["chronos"]["environment"]["pins"] == ns["ENV_PINS"]["chronos"]
    assert freeze["request_sha256"] == ns["REQUEST_SHA256"]


# --- TS-R09: early hardware check and visible progress


def test_full_tier_without_a_gpu_stops_before_any_download(tmp_path, monkeypatch, plots):
    monkeypatch.chdir(tmp_path)
    ns = {"display": lambda *a, **k: None}
    run_cell(
        "1.1 Notebook controls",
        ns,
        {'WORKSHOP_TIER = "STANDARD"  # @param': 'WORKSHOP_TIER = "FULL"  # @param'},
    )
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="needs a CUDA GPU"):
        run_cell("1.2 Check the notebook runtime", ns)


def test_model_runs_print_stage_progress(tmp_path, monkeypatch, plots, capsys):
    ns = start(tmp_path, monkeypatch)
    install_fake_models(ns)
    capsys.readouterr()
    run_cell("7.1 Forecast the validation period with each model", ns)
    printed = capsys.readouterr().out
    assert "[tirex] fake model forecasting" in printed
    assert "[chronos] finished in" in printed


# --- TS-R08: wording no longer claims scale normalisation


def test_metric_and_overlay_wording():
    assert (
        "It is dominated by series with large values, so it depends on scale." not in NOTEBOOK_TEXT
    )
    assert "scaled to the same range" not in NOTEBOOK_TEXT
    assert "neither removes differences in scale" in NOTEBOOK_TEXT
    assert "z-scores" in NOTEBOOK_TEXT


def test_committed_notebook_reads_ids_as_text_in_every_csv_reader():
    for title in (
        "3.1 Generate or load the sample",
        "7.1 Forecast the validation period with each model",
    ):
        assert 'dtype={"series_id": str' in CELLS[title]
    runners = CELLS["6.2 Write the model runner programs"]
    assert runners.count('dtype={"series_id":str}') == 3
    assert 'pd.read_csv(args.input,parse_dates=["timestamp"])' not in runners


def test_notebook_metadata_revision_matches_the_code():
    metadata = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]
    expected = f'NOTEBOOK_REVISION = "{metadata["workshop_revision"]}"'
    assert expected in CELLS["1.1 Notebook controls"]
