"""Regression tests for the row-11 review of tutorials/chronos_2_forecasting_colab.ipynb (CHR-M1..M3, CHR-m1..m4).

The learner cells are executed verbatim from the generated notebook JSON, in one namespace holding the carried
package's names, with a stand-in model (no weights, no torch): `predict_df` repeats each series' last value. These are
interface/journey tests, not pretrained-inference evidence.
"""
# ruff: noqa: E501  -- notebook source lines and messages are kept whole
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# Windows DLL load-order trap (FIX_PACKET addendum): import torch, if present, before any NumPy linear algebra.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

from chronos2_pipeline import config as _config
from chronos2_pipeline import errors as _errors
from chronos2_pipeline import evaluation as _evaluation
from chronos2_pipeline import inference as _inference
from chronos2_pipeline import model as _model
from chronos2_pipeline import provenance as _provenance
from chronos2_pipeline import validation as _validation
from conftest import FakeLoadedModel, make_identity

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "chronos_2_forecasting_colab.ipynb"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_tool("build_notebook")
TEMPLATE = _load_tool("notebook_template").TEMPLATE


def _src(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell(nb: dict, marker: str) -> str:
    hits = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


def _with_field(source: str, prefix: str, line: str) -> str:
    lines = source.splitlines()
    hits = [i for i, text in enumerate(lines) if text.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    lines[hits[0]] = line
    return "\n".join(lines) + "\n"


# --- CHR-M1: isolated runtime, no in-kernel install -------------------------------------------------------------


def test_isolated_runtime_uses_the_fleet_uv_mechanism(nb: dict) -> None:
    code = _code_cells(nb)
    kernel = [i for i, c in enumerate(code) if "# dimer: kernel cell" in _src(c)]
    assert kernel == [0, 1], "only the install and router cells may run in the notebook kernel"
    install, router = _src(code[0]), _src(code[1])
    assert '"venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON' in install
    assert f"MANAGED_PYTHON = {TEMPLATE['managed_python']!r}" in install
    assert '"--require-hashes", "--only-binary", ":all:"' in install
    assert f"UV_SHA256 = {TEMPLATE['uv']['sha256']!r}" in install and f"UV_BYTES = {TEMPLATE['uv']['bytes']}" in install
    assert 'platform.machine() != "x86_64"' in install
    assert "sys.executable" not in install.split("SKIP_INSTALL = ", 1)[1], "nothing may be pip-installed into the kernel"
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    assert 'DIMER_NOTEBOOK_CI_PREINSTALLED="1"' in router
    # The record cell keeps the guarded pip path for pre-installed executors only; the worker sets the flag.
    record = _src(code[2])
    assert record.startswith("# @title Infrastructure: record the runtime")
    assert "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'" in record


def test_carried_lock_equals_the_repository_lock_and_the_pins(nb: dict) -> None:
    install = _src(_code_cells(nb)[0])
    lock_text = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    assert f"LOCK_TEXT = r'''{lock_text}'''" in install
    assert f"LOCK_SHA256 = {hashlib.sha256(lock_text.encode('utf-8')).hexdigest()!r}" in install
    BUILD.check_lock(BUILD._pins(ROOT, TEMPLATE), lock_text)  # raises SystemExit on drift
    assert "--generate-hashes" in lock_text.splitlines()[1] and "x86_64-manylinux_2_28" in lock_text.splitlines()[1]


def test_router_needs_no_ipython_when_an_executor_runs_every_cell(nb: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setitem(__import__("sys").modules, "IPython", None)  # `import IPython` now raises ImportError
    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    assert "Routing disabled" in capsys.readouterr().out


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """Colab T4 run of f9e605e: accelerate's find_spec("google.colab") raised on the worker's spec-less stub."""
    import importlib.util
    import sys

    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start:worker.index('_main = types.ModuleType("__main__")', start)]
    names = ("google", "google.colab", "google.colab.files")
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    try:
        for name in names:
            sys.modules.pop(name, None)
        # Exercise both branches: no importable `google` (stub created) and an existing namespace package.
        sys.modules["google"] = fake_google if real_google else None
        monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)  # raised ValueError before the fix
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_release_records_no_longer_call_a_restart_dependent_run_a_clean_pass() -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "restarted_after_install_cell: true" in text
    assert "not a one-pass `Run all`" in text
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "verified — clean-runtime `Run all` execution recorded" not in registry


# --- CHR-M3 / CHR-m1: guided layer ---------------------------------------------------------------------------------


def test_infrastructure_cells_are_titled_and_collapsed(nb: dict) -> None:
    infra = [c for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c) or c.get("metadata", {}).get("dimer", {}).get("embedded_module") or _src(c).startswith("# @title Infrastructure: record")]
    assert len(infra) == 3 + len(TEMPLATE["modules"])
    for cell in infra:
        assert _src(cell).startswith("# @title Infrastructure:"), _src(cell)[:80]
        assert cell["metadata"].get("cellView") == "form"


def test_guided_layer_is_present_and_stale_text_is_gone(nb: dict) -> None:
    md = _markdown(nb)
    for marker in (
        "### Who this is for", "### How to use this notebook", "### Roadmap", "### Input → Model → Output",
        "## Before Section 4: glossary", "## Troubleshooting", "## Conclusion template",
        "#### What to notice (Section 6)", "#### What to notice (Section 7)", "#### What to notice (Section 9)",
    ):
        assert marker in md, marker
    for section in ("## 6. Forecast", "## 7. Evaluate", "## 9. Primary Mode C"):
        block = md.split(section, 1)[1].split("#### What to notice", 1)[0]
        assert "**Predict first.**" in block, section
    assert md.count("<details><summary>Sample answer") >= 4
    assert md.count("**proves**") == 1 and "Successful execution proves that" not in md
    assert "toward the native 1,024-step horizon" not in md
    assert "the cell stops with a restart instruction" not in md


def test_section7_explains_the_degenerate_baseline_and_coverage(nb: dict) -> None:
    notice = _markdown(nb).split("#### What to notice (Section 7)", 1)[1].split("## 8.", 1)[0]
    assert "24 × 0.25 = **6.0 per day**" in notice
    assert "**Coverage of 1.0 here says nothing about calibration.**" in notice


# --- learner cells with a stand-in model ---------------------------------------------------------------------------


class LastValuePipeline:
    """`predict_df` stand-in in the pinned upstream row layout: each step repeats the series' last value."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict_df(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        id_col, ts_col, horizon = kwargs["id_column"], kwargs["timestamp_column"], kwargs["prediction_length"]
        self.calls.append({"context_length": kwargs.get("context_length"), "rows": len(df), "future": kwargs.get("future_df") is not None})
        rows = []
        for sid in pd.unique(df[id_col]):
            block = df[df[id_col] == sid].sort_values(ts_col)
            stamps = pd.to_datetime(block[ts_col])
            step = stamps.iloc[-1] - stamps.iloc[-2]
            for target in kwargs["target"]:
                last = float(block[target].iloc[-1])
                for k in range(horizon):
                    rows.append({id_col: sid, ts_col: stamps.iloc[-1] + (k + 1) * step, "target_name": target, "predictions": last, "0.1": last - 1.0, "0.5": last, "0.9": last + 1.0})
        return pd.DataFrame(rows)


def _learner_cells(nb: dict) -> list[str]:
    out = []
    for cell in _code_cells(nb):
        text = _src(cell)
        if text.startswith("# @title Infrastructure:") or "MANIFEST = {" in text:
            continue
        out.append(text)
    return out


def _namespace(pipeline: LastValuePipeline) -> dict[str, Any]:
    ns: dict[str, Any] = {"__name__": "__main__"}
    for module in (_errors, _provenance, _config, _model, _validation, _inference, _evaluation):
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update(
        pipe=FakeLoadedModel(pipeline=pipeline, identity=make_identity()),
        NOTEBOOK_SOURCE={"repository": "chronos-2-forecasting-pipeline", "repository_revision": "test", "notebook_spec": "2.2"},
        platform=__import__("platform"),
        torch=types.SimpleNamespace(__version__="stand-in"),
        transformers=types.SimpleNamespace(__version__="stand-in"),
        pandas=pd,
        numpy=__import__("numpy"),
    )
    return ns


def _run(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, tuple[str, str]] | None = None, env: dict[str, str] | None = None) -> tuple[dict[str, Any], LastValuePipeline]:
    for name in ("DIMER_BYOD_PATH", "DIMER_RUN_COVARIATE_DEMO", "DIMER_FORECAST_FUTURE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    pipeline = LastValuePipeline()
    ns = _namespace(pipeline)
    for source in _learner_cells(nb):
        for prefix, line in (overrides or {}).values():
            if any(text.startswith(prefix) for text in source.splitlines()):
                source = _with_field(source, prefix, line)
        exec(compile(source, "learner-cell", "exec"), ns)
    return ns, pipeline


def _result(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "outputs" / "chronos_2_forecasting_result.json").read_text(encoding="utf-8"))


def test_default_path_runs_and_records_the_horizon_activity(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ns, pipeline = _run(nb, tmp_path, monkeypatch)
    assert ns["result"].inference["effective_context_length"] == 84
    table = ns["activity_table"]
    assert table["horizon"].tolist() == [12, 24, 48]
    assert set(table["effective_context_length"]) == {48}, "CHR-M2: the activity must hold the context fixed"
    activity_calls = [c for c in pipeline.calls if c["context_length"] == 48]
    assert len(activity_calls) == 3 and {c["rows"] for c in activity_calls} == {48}, "one cut-off for every horizon"
    runs = _result(tmp_path)["learner_runs"]
    assert runs["horizon_activity"]["run"] is True and runs["mode_d"] == {"run": False} and runs["future_forecast"] == {"run": False}
    files = set(p.name for p in (tmp_path / "outputs").iterdir())
    assert "chronos_horizon_activity.csv" in files and not any("covariate" in f or "future" in f for f in files)


def test_activity_change_one_thing_and_infeasible_settings(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ns, _ = _run(nb, tmp_path, monkeypatch)
    activity = _cell(nb, 'HORIZONS = "12, 24, 48"')
    exec(compile(_with_field(activity, "HORIZONS = ", 'HORIZONS = "6, 12, 24, 36, 48"'), "s11", "exec"), ns)
    assert ns["activity_table"]["horizon"].tolist() == [6, 12, 24, 36, 48]
    assert set(ns["activity_table"]["effective_context_length"]) == {48}
    capsys.readouterr()
    exec(compile(_with_field(activity, "HORIZONS = ", 'HORIZONS = "12, 60"'), "s11", "exec"), ns)
    out = capsys.readouterr().out
    assert "Activity skipped" in out and "Use horizons up to 48" in out
    assert _result(tmp_path)["learner_runs"]["horizon_activity"]["run"] is False
    assert not (tmp_path / "outputs" / "chronos_horizon_activity.csv").exists()


def test_prediction_length_prose_states_the_coupling_and_feasible_range(nb: dict) -> None:
    md = _markdown(nb)
    assert "**`PREDICTION_LENGTH` sets two things at once.**" in md
    assert "Values from 1 to 93 run on the sample" in md and "skips it above 72" in md


def test_mode_d_compares_with_and_without_covariates_then_cleans_up(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ns, pipeline = _run(nb, tmp_path, monkeypatch, env={"DIMER_RUN_COVARIATE_DEMO": "1"})
    out = tmp_path / "outputs"
    provenance = json.loads((out / "chronos_covariate_provenance.json").read_text(encoding="utf-8"))
    assert provenance["known_future_covariate_names"] == ["temperature", "holiday"]
    assert [row["run"] for row in provenance["comparison"]] == ["with known-future covariates", "without covariates"]
    assert all(row["n"] == 48 and "mae" in row for row in provenance["comparison"])
    assert [c["future"] for c in pipeline.calls[-2:]] == [True, False]
    record = _result(tmp_path)["learner_runs"]["mode_d"]
    assert record["run"] is True and record["provenance"] == "outputs/chronos_covariate_provenance.json"
    # On, then off: no Mode D file survives and the result JSON says it did not run (CHR-m4).
    monkeypatch.delenv("DIMER_RUN_COVARIATE_DEMO")
    exec(compile(_cell(nb, "RUN_COVARIATE_DEMO = False"), "s12", "exec"), ns)
    assert not [p.name for p in out.iterdir() if "covariate" in p.name]
    assert _result(tmp_path)["learner_runs"]["mode_d"] == {"run": False}


def test_future_forecast_starts_after_the_last_input_and_is_not_measurable(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ns, _ = _run(nb, tmp_path, monkeypatch, env={"DIMER_FORECAST_FUTURE": "1"})
    out = tmp_path / "outputs"
    forecast_rows = pd.read_csv(out / "chronos_future_forecast.csv")
    assert pd.Timestamp(forecast_rows["timestamp"].min()) == pd.Timestamp("2026-01-04T23:00:00") + pd.Timedelta(hours=1)
    assert json.loads((out / "chronos_future_evaluation_report.json").read_text(encoding="utf-8"))["verdict"] == "not-measurable"
    monkeypatch.delenv("DIMER_FORECAST_FUTURE")
    exec(compile(_cell(nb, "FORECAST_FUTURE = False"), "s13", "exec"), ns)
    assert not [p.name for p in out.iterdir() if "future" in p.name]


# --- CHR-m2: BYOD fields and messages ------------------------------------------------------------------------------


def _byod_frame() -> pd.DataFrame:
    stamps = pd.date_range("2026-03-01", periods=72, freq="h")
    return pd.concat([pd.DataFrame({"site": sid, "time": stamps, "value": base + (stamps.hour % 24) * 0.5}) for sid, base in (("north", 50.0), ("south", 80.0))], ignore_index=True)


def test_byod_with_own_column_names_runs_once_the_fields_are_set(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "mine.csv"
    _byod_frame().to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Set ID_COLUMN, TIMESTAMP_COLUMN and TARGET_COLUMN"):
        _run(nb, tmp_path, monkeypatch, env={"DIMER_BYOD_PATH": str(csv_path)})
    fields = {
        "id": ("ID_COLUMN = ", 'ID_COLUMN = "site"  # @param {type:"string"}'),
        "ts": ("TIMESTAMP_COLUMN = ", 'TIMESTAMP_COLUMN = "time"  # @param {type:"string"}'),
        "target": ("TARGET_COLUMN = ", 'TARGET_COLUMN = "value"  # @param {type:"string"}'),
        "path": ('BYOD_PATH = ""', f'BYOD_PATH = {str(csv_path)!r}  # @param {{type:"string"}}'),
    }
    ns, _ = _run(nb, tmp_path, monkeypatch, overrides=fields)
    assert ns["sample_kind"] == "BYOD" and set(ns["result"].forecast["target_name"]) == {"value"}
    assert set(ns["mode_c_result"].forecast["target_name"]) == {"value", "value_aux"}
    # 72 rows cannot hold 48 context + 48 horizon: the activity is skipped, the run is not stopped.
    assert _result(tmp_path)["learner_runs"]["horizon_activity"]["run"] is False


def test_byod_non_utf8_is_refused_with_a_utf8_message(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "utf16.csv"
    csv_path.write_bytes(_byod_frame().rename(columns={"site": "series_id", "time": "timestamp", "value": "target"}).to_csv(index=False).encode("utf-16"))
    with pytest.raises(ValueError, match="must be UTF-8"):
        _run(nb, tmp_path, monkeypatch, env={"DIMER_BYOD_PATH": str(csv_path)})


def test_byod_missing_path_names_the_field(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FileNotFoundError, match="BYOD_PATH"):
        _run(nb, tmp_path, monkeypatch, env={"DIMER_BYOD_PATH": str(tmp_path / "absent.csv")})


def test_mode_c_never_overwrites_a_user_target_aux_column(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "aux.csv"
    frame = _byod_frame().rename(columns={"site": "series_id", "time": "timestamp", "value": "target"}).assign(target_aux=1.0)
    frame.to_csv(csv_path, index=False)
    ns, _ = _run(nb, tmp_path, monkeypatch, env={"DIMER_BYOD_PATH": str(csv_path)})
    assert ns["aux_target"] == "target_aux_derived"
    assert (ns["frame"]["target_aux"] == 1.0).all()
    assert set(ns["mode_c_result"].forecast["target_name"]) == {"target", "target_aux_derived"}
