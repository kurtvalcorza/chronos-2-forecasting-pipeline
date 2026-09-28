"""Regression checks for the 2026-09-28 capstone review (DENGUE-01 to DENGUE-06).

Foundation-model calls are explicit synthetic doubles and the GPU gate is not exercised: these
are software contracts, not hosted or model-performance evidence.
"""

import ast
import copy
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
NOTEBOOK = TOOLS.parent / "tutorials" / "DIMER_Philippine_Dengue_Forecasting_Capstone.ipynb"
sys.path.insert(0, str(TOOLS))
import dengue_core as core  # noqa: E402
import dengue_runtime as runtime  # noqa: E402

MODULES = ("dengue_runtime.py", "dengue_core.py", "dengue_data.py", "dengue_models.py")
BYOD_META = {"area": "Synthetic test area", "source_citation": "Synthetic fixture"}


def series(years=16, cases=lambda i: 10 + i % 52 + (i // 52) % 3):
    return [
        {
            "year": 2010 + i // 52,
            "block": i % 52 + 1,
            "cases": int(cases(i)),
            "rain": float(i % 11),
            "temp": 27.0 + (i % 5) / 10,
        }
        for i in range(years * 52)
    ]


@pytest.fixture
def rows():
    return series()


@pytest.fixture
def root(tmp_path, rows, monkeypatch):
    root = tmp_path / "run"
    (root / "results").mkdir(parents=True)
    runtime.write(root / "data.json", rows)
    runtime.write(root / "dataset_audit.json", {"scope": runtime.SCOPE})
    runtime.write(root / "model_manifest.json", {"synthetic_test": True})
    for name in MODULES:
        shutil.copy(TOOLS / name, root / name)
    runtime.write(root / "source.json", {"files": {n: runtime.sha(root / n) for n in MODULES}})
    (root / "DATA_LICENSE.md").write_text("Synthetic fixture; no source data.")
    monkeypatch.setattr(runtime.data, "acquire", lambda root: runtime.read(root / "data.json"))
    monkeypatch.setattr(runtime.models, "load_chronos", lambda *args: object())
    monkeypatch.setattr(runtime.models, "load_mitra", lambda *args: object())
    monkeypatch.setattr(
        runtime.models,
        "chronos_predict",
        lambda model, history, h: np.tile(
            [history[-1] - 1, history[-1], history[-1] + 1], (h, 1)
        ).astype(float),
    )
    monkeypatch.setattr(
        runtime.models,
        "mitra_predict",
        lambda model, X, y, query: np.repeat(np.mean(y) + 0.01 * query[0][0], len(query)),
    )
    monkeypatch.setattr(runtime, "free_gpu", lambda: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "synthetic-test")
    return root


def complete_run(root, monkeypatch):
    for name in runtime.STAGES[:7]:
        getattr(runtime, name)(root)
    manifest = runtime.read(root / "results" / "artifact_manifest.json")
    monkeypatch.setattr(runtime.os, "getpid", lambda: manifest["creator_pid"] + 1)
    runtime.reload(root)
    for stage in runtime.STAGES[:-1]:
        runtime.write(root / "results" / f"receipt_{stage}.json", {"seconds": 1.0})
    runtime.report(root)
    return root / "results"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cell(fragment):
    matches = [
        c["source"]
        for c in notebook()["cells"]
        if c["cell_type"] == "code" and fragment in c["source"]
    ]
    assert len(matches) == 1, fragment
    return matches[0]


def helpers(root):
    """The notebook's own table helper, extracted from the carrier cell with display stubs."""
    tree = ast.parse(cell("def table(name"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "table"]
    shown = []
    namespace = {
        "csv": __import__("csv"),
        "ROOT": root,
        "Markdown": lambda text: ("markdown", text),
        "FileLink": lambda path: ("link", path),
        "display": shown.append,
        "print": lambda *args, **kwargs: shown.append(("print", " ".join(map(str, args)))),
    }
    exec(compile(ast.Module(wanted, []), "notebook-table", "exec"), namespace)
    return namespace, shown


def run_cell(root, fragment):
    namespace, shown = helpers(root)
    namespace.update(
        run=lambda stage: None,
        record=lambda name: None,
        figure=lambda name: None,
        json=json,
        consume=lambda: None,
    )
    exec(cell(fragment), namespace)
    return [text for kind, text in (s for s in shown if isinstance(s, tuple)) if kind == "markdown"]


def markdown_rows(text):
    lines = text.splitlines()
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    return [
        dict(zip(header, (v.strip() for v in line.strip("|").split("|")), strict=True))
        for line in lines[2:]
    ]


# DENGUE-01 — principal results are visible, with columns chosen by meaning.


def test_future_section_shows_all_24_forecasts_from_six_systems(root, monkeypatch):
    complete_run(root, monkeypatch)
    pivot, detail = map(markdown_rows, run_cell(root, "run('future')"))
    assert len(pivot) == 4 and [r["horizon"] for r in pivot] == ["1", "2", "3", "4"]
    for row in pivot:
        assert all(float(row[s]) >= 0 for s in runtime.SYSTEMS)
    assert len(detail) == 24
    assert {r["system"] for r in detail} == set(runtime.SYSTEMS)
    assert all(float(r["prediction"]) >= 0 for r in detail)
    assert all(r["q50"] for r in detail if r["system"] == "chronos")
    assert "reference" not in detail[0]


def test_largest_misses_and_delay_views_show_the_compared_quantities(root, monkeypatch):
    out = complete_run(root, monkeypatch)
    [misses] = map(markdown_rows, run_cell(root, "run('test')"))
    assert len(misses) == 10
    for row in misses:
        assert float(row["error"]) == pytest.approx(
            float(row["prediction"]) - float(row["reference"])
        )
        assert float(row["abs_error"]) == pytest.approx(abs(float(row["error"])))
    summary, paired = map(markdown_rows, run_cell(root, "run('activity')"))
    zero, delayed = (
        runtime.read(out / "validation_metrics.json"),
        runtime.read(out / "delay_metrics.json"),
    )
    assert [r["system"] for r in summary] == list(runtime.SYSTEMS)
    for row in summary:
        assert float(row["mae_delay_0"]) == pytest.approx(zero[row["system"]]["mae"], abs=1e-3)
        assert float(row["mae_delay_2"]) == pytest.approx(delayed[row["system"]]["mae"], abs=1e-3)
        assert row["pairs"] == "104"
    assert len(paired) == 12 and all(
        r["prediction_delay_0"] and r["prediction_delay_2"] for r in paired
    )


def test_every_notebook_table_names_existing_columns(root, monkeypatch):
    complete_run(root, monkeypatch)
    sections = [
        c["source"]
        for c in notebook()["cells"]
        if c["cell_type"] == "code" and "table('" in c["source"] and "FILES = " not in c["source"]
    ]
    assert len(sections) == 5
    for source in sections:
        fragment = source.split("\n")[0]
        assert run_cell(root, fragment), fragment


def test_table_helper_refuses_missing_column_instead_of_hiding_it(root, monkeypatch):
    complete_run(root, monkeypatch)
    namespace, _ = helpers(root)
    with pytest.raises(KeyError, match="lacks columns"):
        namespace["table"]("future_predictions.csv", ["system", "forecast_value"])


# DENGUE-02 — BYOD semantics match the default source and are checked before any model.


def byod(root, text, **meta):
    path = root / "byod.csv"
    path.write_text(text, encoding="utf-8")
    runtime.write(
        root / "run_config.json",
        {"byod_csv": str(path), "source_blocks_confirmed": True, **(meta or BYOD_META)},
    )
    return path


def csv_text(rows, extra_line=None):
    lines = ["year,block,cases,rain,temp"]
    lines += [f"{r['year']},{r['block']},{r['cases']},{r['rain']},{r['temp']}" for r in rows]
    if extra_line:
        index, line = extra_line
        lines[index] = line
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "line,message",
    [
        ("2010,3,0.5,1.0,27.0", "line 4, column cases: must be a nonnegative integer count"),
        ("2010,3,12,1.0,500", "line 4, column temp: outside the declared Celsius range"),
        ("2010,3,,1.0,27.0", "line 4, column cases: missing value; missing is never replaced"),
        ("2010,3,12,1.0,27.0,sentinel", "line 4: expected 5 values, found 6"),
        ("2010,3,12,1.0", "line 4: expected 5 values, found 4"),
        ("2010,3,twelve,1.0,27.0", "line 4, column cases: not a number"),
    ],
)
def test_byod_values_refused_with_file_line_and_column(root, monkeypatch, line, message):
    monkeypatch.setattr(runtime.models, "load_mitra", pytest.fail)
    byod(root, csv_text(series(6), (3, line)))
    with pytest.raises(ValueError, match=message):
        runtime.prepare(root)


def test_default_and_byod_share_count_and_unit_semantics(rows):
    for field, value in [("cases", 0.5), ("temp", 500.0), ("cases", None)]:
        broken = copy.deepcopy(rows)
        broken[10][field] = value
        with pytest.raises(ValueError, match=f"Invalid {field} at row 10"):
            core.validate_rows(broken)


def test_zero_counts_in_a_varying_series_remain_valid(root):
    byod(root, csv_text(series(6, cases=lambda i: 0 if i % 7 == 0 else i % 52)))
    runtime.prepare(root)
    preflight = runtime.read(root / "results" / "context_preflight.json")
    assert preflight["contexts_refused"] == 0 and preflight["contexts_checked"] == 1264


def test_constant_support_refused_before_any_model_is_staged(root, monkeypatch):
    monkeypatch.setattr(runtime.models, "load_mitra", pytest.fail)
    monkeypatch.setattr(runtime.models, "load_chronos", pytest.fail)
    byod(root, csv_text(series(6, cases=lambda i: 0)))
    with pytest.raises(ValueError, match="planned Mitra contexts cannot be fitted"):
        runtime.prepare(root)
    preflight = runtime.read(root / "results" / "context_preflight.json")
    assert preflight["contexts_refused"] == preflight["contexts_checked"] == 1264
    assert {r["system"] for r in preflight["refused"]} == {"mitra_cases", "mitra_weather"}


def test_locally_constant_stretch_is_named_not_dropped(root):
    # Varying overall, but the 53 blocks before the first validation origin never change.
    byod(root, csv_text(series(6, cases=lambda i: 5 if 50 <= i <= 104 else 3 + i % 17)))
    with pytest.raises(ValueError, match=r"first: validation origin 2011-B52"):
        runtime.prepare(root)


def test_mitra_adapter_and_preflight_share_one_support_rule():
    X = np.arange(12, dtype=float).reshape(6, 2)
    assert runtime.models.support_problem(X, np.zeros(6)) == (
        "all mature support targets are identical (constant target)"
    )
    assert runtime.models.support_problem(np.ones((6, 2)), np.arange(6.0)) == (
        "no support feature varies"
    )
    assert runtime.models.support_problem(X, np.arange(6.0)) is None
    with pytest.raises(ValueError, match="nonconstant support targets"):
        runtime.models.mitra_predict(None, X, np.zeros(6), X[:1])


# DENGUE-03 — the exported bundle reconstructs without the original workspace.


def consumer(tmp_path, root):
    code = tmp_path / "embedded"
    code.mkdir()
    for name in (*MODULES, "source.json", "model_manifest.json"):
        shutil.copy(root / name, code / name)
    bundle = tmp_path / "results.zip"
    shutil.copy(root / "results" / "results.zip", bundle)
    return code, bundle


def rewrite(bundle, change):
    with zipfile.ZipFile(bundle) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    change(files)
    checksums = json.loads(files["checksums.json"])
    for name in list(checksums):
        if name in files:
            checksums[name] = hashlib.sha256(files[name]).hexdigest()
    files["checksums.json"] = json.dumps(checksums).encode()
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)


def test_bundle_reconstructs_after_original_workspace_is_deleted(tmp_path, root, monkeypatch):
    complete_run(root, monkeypatch)
    expected = runtime.read(root / "results" / "verification.json")
    code, bundle = consumer(tmp_path, root)
    shutil.rmtree(root)
    result = runtime.consume(bundle, tmp_path / "check", tmp_path / "models", code)
    assert (
        result["passed"] and result["predictions_checked"] == expected["predictions_checked"] == 32
    )
    assert result["max_absolute_difference"] <= 1e-9
    assert runtime.read(tmp_path / "check" / "consumer_verification.json") == result


def json_edit(name, change):
    def edit(files):
        value = json.loads(files[name])
        change(value)
        files[name] = json.dumps(value).encode()

    return edit


@pytest.mark.parametrize(
    "edit,message",
    [
        (
            json_edit(
                "experiment_lock.json", lambda v: v.update(window=104 if v["window"] == 52 else 52)
            ),
            "feature schema or experiment lock changed",
        ),
        (
            json_edit("dataset_audit.json", lambda v: v.update(area="Relabelled area")),
            "provenance, configuration or code identity changed",
        ),
        (
            json_edit("plan.json", lambda v: v.update(windows=[26])),
            "provenance, configuration or code identity changed",
        ),
        (
            json_edit("data_manifest.json", lambda v: v.update(data_sha256="0" * 64)),
            "provenance, configuration or code identity changed",
        ),
        (lambda files: files.update({"extra.json": b"{}"}), "differ from checksums"),
    ],
)
def test_bundle_consumer_rejects_mutated_configuration(tmp_path, root, monkeypatch, edit, message):
    complete_run(root, monkeypatch)
    code, bundle = consumer(tmp_path, root)
    rewrite(bundle, edit)
    with pytest.raises(ValueError, match=message):
        runtime.consume(bundle, tmp_path / "check", tmp_path / "models", code)


def test_bundle_consumer_rejects_mutated_context_even_with_updated_hashes(
    tmp_path, root, monkeypatch
):
    complete_run(root, monkeypatch)
    code, bundle = consumer(tmp_path, root)

    def edit(files):
        name = "artifact_future_mitra_weather.json"
        state = json.loads(files[name])
        origin = next(iter(state["origins"]))
        state["origins"][origin]["1"]["y"][0] += 3.0
        files[name] = json.dumps(state).encode()
        manifest = json.loads(files["artifact_manifest.json"])
        manifest["files"][name] = hashlib.sha256(files[name]).hexdigest()
        files["artifact_manifest.json"] = json.dumps(manifest).encode()

    rewrite(bundle, edit)
    with pytest.raises(ValueError, match="future mitra_weather context does not reproduce"):
        runtime.consume(bundle, tmp_path / "check", tmp_path / "models", code)


def test_bundle_consumer_requires_the_trusted_model_manifest(tmp_path, root, monkeypatch):
    complete_run(root, monkeypatch)
    code, bundle = consumer(tmp_path, root)
    runtime.write(code / "model_manifest.json", {"other": True})
    with pytest.raises(ValueError, match="model_manifest.json differs"):
        runtime.consume(bundle, tmp_path / "check", tmp_path / "models", code)


def test_artifact_manifest_discloses_retained_aggregate_data(root, monkeypatch):
    out = complete_run(root, monkeypatch)
    manifest = runtime.read(out / "artifact_manifest.json")
    assert "No individual records" in manifest["retained_data"]
    assert manifest["bundle_identity"]


# DENGUE-04 — BYOD area identity labels outputs and provenance, never predictors.


def test_byod_area_and_source_recorded_and_required(root):
    byod(root, csv_text(series(6)))
    runtime.prepare(root)
    plan, audit = (
        runtime.read(root / "results" / "plan.json"),
        runtime.read(root / "dataset_audit.json"),
    )
    assert plan["area"] == audit["area"] == "Synthetic test area"
    assert audit["source_citation"] == "Synthetic fixture" and audit["units"] == core.UNITS
    assert "area" not in " ".join(core.feature_names(weather=True))
    for meta in ({"area": "", "source_citation": "x"}, {"area": "x\ny", "source_citation": "x"}):
        byod(root, csv_text(series(6)), **meta)
        with pytest.raises(ValueError, match="BYOD_AREA"):
            runtime.prepare(root)


def test_default_source_names_quezon_city(root, monkeypatch):
    out = complete_run(root, monkeypatch)
    summary = runtime.read(out / "run_summary.json")
    assert summary["area"] == runtime.DEFAULT_AREA == "Quezon City, Philippines"


def test_notebook_requires_byod_area_and_citation_before_running():
    source = cell("USE_BYOD = False")
    assert 'BYOD_AREA = "" # @param' in source and 'BYOD_SOURCE_CITATION = "" # @param' in source
    assert "never relabelled Quezon City" in source
    assert "'area': BYOD_AREA.strip() if USE_BYOD else None" in cell("FILES = ")


# DENGUE-05 — less raw infrastructure; one traced feature row.


def test_carrier_cell_is_collapsed_with_a_title():
    [carrier] = [c for c in notebook()["cells"] if "FILES = " in c.get("source", "")]
    assert carrier["metadata"]["cellView"] == "form"
    assert carrier["metadata"]["jupyter"]["source_hidden"] is True
    assert carrier["source"].startswith("# @title Embedded source")


def test_feature_example_traces_values_to_available_blocks(root, rows):
    runtime.prepare(root)
    plan = runtime.read(root / "results" / "plan.json")
    example = runtime.feature_example(rows, plan)
    origin = plan["test"][0]
    features = example[2:-1]
    values = core.feature_row(rows, origin, 1, weather=True)
    assert [r["item"] for r in features] == core.feature_names(weather=True)
    assert [r["value"] for r in features] == pytest.approx(values.round(4).tolist())
    latest = runtime.key(rows, origin)
    for row in features:
        for block in row["source_blocks"].split(" .. "):
            if block.startswith("20"):
                assert block <= latest
    assert example[-1]["source_blocks"] == latest and example[-1]["value"] == rows[origin]["cases"]
    shown = markdown_rows(run_cell(root, "run('baselines')")[0])
    assert len(shown) == len(example)


# DENGUE-06 — limitations and runtime identity carried into the summary.


def test_run_summary_carries_limitations_environment_and_targets(root, monkeypatch):
    out = complete_run(root, monkeypatch)
    summary = runtime.read(out / "run_summary.json")
    text = " ".join(summary["limitations"])
    for phrase in (
        "Final data vintages",
        "pretraining overlap",
        "calendar dates",
        "Two evaluation",
    ):
        assert phrase in text
    env = summary["environment"]
    assert env["python"] and env["device"] and "float32" in env["precision"]
    assert {"autogluon.tabular", "pandas", "chronos-forecasting", "torch"} <= set(env["packages"])
    assert summary["resource_targets"]["kind"] == "targets, not measurements"
    assert "peak_allocated_gpu_bytes" in summary["resource_measurements"]
    assert runtime.read(out / "environment.json") == env
