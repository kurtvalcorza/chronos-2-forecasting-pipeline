"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/chronos_2_forecasting_colab.ipynb"
BYOD_CELL = "chronos_2_forecasting-26"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "chronos_2_forecasting"
SPLIT_CELL = "chronos_2_forecasting-28"
EARLIER = {"frame": "old", "split": "old", "result": "old", "evaluation": "old", "report": "old", "input_manifest": "old", "payload_out": "old"}
GOOD_CSV = "series_id,timestamp,target\n" + "".join(f"A,2026-01-01T{h:02d}:00:00,{h}.0\n" for h in range(20))


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    for name in (f"{STEM}_input_manifest.json", f"{STEM}_result.json", "chronos_forecast.csv", "chronos_provenance.json"):
        (out / name).write_text("{}", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str, *, use_byod: bool = True, source: str | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DIMER_BYOD_PATH", raising=False)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    if use_byod:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace('BYOD_PATH = ""  #', f"BYOD_PATH = {path!r}  #", 1)
    ns: dict = {"pipe": "kept", **EARLIER}
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    (tmp_path / "utf16.csv").write_bytes("series_id,timestamp,target\n".encode("utf-16"))
    (tmp_path / "dupes.csv").write_text("series_id,timestamp,target,target\nA,2026-01-01,1,2\n", encoding="utf-8")
    (tmp_path / "empty.csv").write_bytes(b"")
    return {
        "text file (the scripted /etc/hostname run)": ("hostname", "has no column named ['series_id', 'timestamp', 'target']"),
        "folder": ("folder", "is not a file in this runtime"),
        "missing path": ("nope.csv", "is not a file in this runtime"),
        "not UTF-8": ("utf16.csv", "must be UTF-8 encoded"),
        "duplicate header": ("dupes.csv", "Duplicate CSV header(s) are ambiguous"),
        "empty file": ("empty.csv", "CSV is empty"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing path", "not UTF-8", "duplicate header", "empty file"])
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, rule = _cases(tmp_path)[case]
    holder: dict = {}
    with pytest.raises(ValueError) as caught:
        holder = _section4(monkeypatch, tmp_path, str(tmp_path / file_name))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert not holder
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


def test_refusal_clears_the_session_but_keeps_the_model(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("text\n", encoding="utf-8")
    source = _cell(BYOD_CELL).replace('BYOD_PATH = ""  #', "BYOD_PATH = 'notes.txt'  #", 1)
    ns: dict = {"pipe": "kept", **EARLIER}
    with pytest.raises(ValueError, match="^notes.txt: The CSV has no column named"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): USE_BYOD opens the Colab upload dialog, and this runtime is not Colab")


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "received 0 files"), ({"a.csv": b"1", "b.csv": b"2"}, "a.csv, b.csv", "received 2 files")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: Upload exactly one CSV") and rule in line


def test_uploaded_file_is_named_in_the_refusal(monkeypatch, tmp_path):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"mine.csv": b"a,b\n1,2\n"})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: mine.csv: The CSV has no column named")


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    (tmp_path / "ok.csv").write_text(GOOD_CSV, encoding="utf-8")
    source = _cell(BYOD_CELL).replace("    rows = csv.reader(io.StringIO(text))\n", "    raise RuntimeError('internal bug')\n", 1)
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / "ok.csv"), source=source)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / "chronos_forecast.csv").exists()  # nothing is removed for an internal error


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    source = _cell(BYOD_CELL).replace('SAMPLE_SHA256 = "eff96b1a', 'SAMPLE_SHA256 = "0ff96b1a', 1)
    with pytest.raises(ValueError, match="Synthetic sample digest mismatch") as caught:
        _section4(monkeypatch, tmp_path, "", use_byod=False, source=source)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_sample_path_is_unchanged(monkeypatch, tmp_path):
    ns = _section4(monkeypatch, tmp_path, "", use_byod=False)
    assert ns["sample_kind"] == "synthetic" and len(ns["frame"]) == 96


def test_good_byod_csv_still_loads(monkeypatch, tmp_path):
    (tmp_path / "ok.csv").write_text(GOOD_CSV, encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "ok.csv"))
    assert ns["sample_kind"] == "BYOD" and len(ns["frame"]) == 20 and ns["byod_name"] == "ok.csv"


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/etc/hostname': a file present in every fresh runtime."""
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: The CSV has no column named")


# --- Section 5: holdout split and validate_inputs ------------------------------------------------------------------


def _section5(monkeypatch, tmp_path, sample_kind: str, split_error=None, validate_error=None) -> dict:
    """Section 4 on a good BYOD file (it defines refuse_byod), then Section 5 with stub pipeline functions."""
    (tmp_path / "ok.csv").write_text(GOOD_CSV, encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, str(tmp_path / "ok.csv"))
    _seed_outputs(tmp_path)
    frame = ns["frame"]
    ns.update(EARLIER, sample_kind=sample_kind, frame=frame)

    def holdout(frame, config):
        if split_error:
            raise split_error
        return types.SimpleNamespace(history=frame.iloc[:15], truth=frame.iloc[15:])

    def validate(history, config, pipe, names=None):
        if validate_error:
            raise validate_error
        return {"findings": []}

    limits = types.SimpleNamespace(max_ids=1, max_targets=1, max_covariates=1, max_rows=1, max_context_length=1, max_prediction_length=1)
    ns.update(ForecastConfig=lambda **kw: types.SimpleNamespace(id_column=kw["id_column"], timestamp_column=kw["timestamp_column"], target=kw["target"]),
              chronological_holdout=holdout, validate_inputs=validate, runtime_versions=dict, ResourceLimits=dict, MIN_OBSERVATIONS=3,
              DEFAULT_LIMITS=limits, ValidationError=ValueError, pipe=types.SimpleNamespace(model_context_length=1, model_prediction_length=1))
    exec(compile(_cell(SPLIT_CELL), "<section 5>", "exec"), ns)
    return ns


@pytest.mark.parametrize("where", ["split", "validate"])
def test_section5_contract_violation_on_byod_is_one_refused_line(monkeypatch, tmp_path, where):
    error = ValueError("series 'A' has 2 observations; at least 3 are required")
    kwargs = {"split_error": error} if where == "split" else {"validate_error": error}
    with pytest.raises(ValueError) as caught:
        _section5(monkeypatch, tmp_path, "BYOD", **kwargs)
    line = _refusal_line(caught.value)
    assert line.startswith("Refused: ok.csv: ValueError: series 'A' has 2 observations") and "run Section 4 and every cell after it again" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []


@pytest.mark.parametrize("where", ["split", "validate"])
def test_section5_errors_on_the_sample_keep_their_traceback(monkeypatch, tmp_path, where):
    error = ValueError("sample broke")
    kwargs = {"split_error": error} if where == "split" else {"validate_error": error}
    with pytest.raises(ValueError, match="sample broke") as caught:
        _section5(monkeypatch, tmp_path, "synthetic", **kwargs)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_section5_normal_path_writes_the_manifest(monkeypatch, tmp_path):
    _section5(monkeypatch, tmp_path, "BYOD")
    assert (tmp_path / "outputs" / f"{STEM}_input_manifest.json").exists()
