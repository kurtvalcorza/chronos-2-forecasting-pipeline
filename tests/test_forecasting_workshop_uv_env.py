"""The forecasting workshop builds its model environments with uv, never in the notebook kernel.

Static and orchestration checks of the 2026-10-03 uv isolated environment change: nothing is
installed into the kernel and no restart is needed; a pinned, hash-checked uv builds one
managed CPython 3.12.12 environment per model from a hash-locked, wheels-only lock carried in
the notebook; model stages run through those environments' Python. Not clean-runtime evidence.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_multimodel_forecasting_workshop as builder  # noqa: E402

NOTEBOOK = ROOT / "tutorials" / builder.NOTEBOOK_NAME
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CODE = {
    "".join(cell["source"]).splitlines()[0].removeprefix("# @title "): "".join(cell["source"])
    for cell in NB["cells"]
    if cell["cell_type"] == "code"
}
ENV_CELL = CODE["6.1 Build one environment per model"]
ALL_CODE = "\n".join(CODE.values())


def assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise AssertionError(f"{name} is not assigned")


def test_no_kernel_install_and_no_restart_guard():
    for forbidden in (
        "pip install",
        '"-m", "pip"',
        "%pip",
        "!pip",
        "uv_pin",
        "os.kill",
        "restart session",
        "restart the",
    ):
        assert forbidden not in ALL_CODE.lower(), forbidden
    assert "Restart session" not in "\n".join("".join(c["source"]) for c in NB["cells"])


def test_uv_is_a_pinned_hash_checked_wheel_and_python_is_managed_3_12_12():
    namespace = {}
    for name in (
        "MODEL_ENV_PYTHON",
        "UV_VERSION",
        "UV_WHEEL_URL",
        "UV_WHEEL_BYTES",
        "UV_WHEEL_SHA256",
    ):
        namespace[name] = ast.literal_eval(assignment(ENV_CELL, name))
    assert namespace["MODEL_ENV_PYTHON"] == "3.12.12"
    assert namespace["UV_VERSION"] == "0.12.15"
    assert namespace["UV_WHEEL_URL"].endswith(
        "uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", namespace["UV_WHEEL_SHA256"])
    # The same uv wheel the repository's dengue capstone already runs on hosted T4.
    dengue = (ROOT / "tools" / "build_dengue_capstone.py").read_text(encoding="utf-8")
    assert namespace["UV_WHEEL_URL"] in dengue and namespace["UV_WHEEL_SHA256"] in dengue
    assert '"--managed-python", "--python", MODEL_ENV_PYTHON' in ENV_CELL


def test_installs_are_hash_locked_and_wheels_only():
    assert '"--require-hashes", "--only-binary", ":all:", "-r", str(lock_path)' in ENV_CELL
    assert '[str(UV), "pip", "install", "--python", str(python)' in ENV_CELL


def test_carried_locks_equal_the_repository_lock_files():
    carried = ast.literal_eval(assignment(ENV_CELL, "ENV_LOCKS"))
    assert set(carried) == {"tirex", "chronos", "toto"}
    for name, text in carried.items():
        assert text == builder.carried_lock(name)
        requirements = [line for line in text.splitlines() if not line.startswith(" ")]
        assert requirements and all(
            re.fullmatch(r"[A-Za-z0-9_.\-]+==\S+ \\", r) for r in requirements
        )
        # Every requirement carries at least one hash on the line after it.
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if not line.startswith(" "):
                assert lines[index + 1].startswith("    --hash=sha256:"), line


def test_direct_pins_are_unchanged_and_resolved_exactly_in_each_lock():
    pins = ast.literal_eval(assignment(ENV_CELL, "ENV_PINS"))
    carried = ast.literal_eval(assignment(ENV_CELL, "ENV_LOCKS"))
    assert pins["tirex"][0] == "tirex-2==0.2.1"
    assert "torch==2.14.0" in pins["chronos"] and "toto-2==2.0.0" in pins["toto"]
    for name, model_pins in pins.items():
        source_in = (ROOT / "tools" / f"forecasting-workshop-{name}-requirements.in").read_text(
            encoding="utf-8"
        )
        assert source_in.split() == model_pins
        locked = {
            line.split(" ")[0].lower()
            for line in carried[name].splitlines()
            if not line.startswith(" ")
        }
        for pin in model_pins:
            assert pin.lower() in locked, (name, pin)


def test_long_literals_are_split_into_pieces_and_no_line_exceeds_2000_characters():
    assert max(len(line) for cell in NB["cells"] for line in cell["source"]) <= 2000
    locks = assignment(ENV_CELL, "ENV_LOCKS")
    for value in locks.values:
        pieces = ENV_CELL.splitlines()[value.lineno - 1 : value.end_lineno]
        assert len(pieces) > 100
        assert all(len(piece) <= builder.MAX_PIECE + 8 for piece in pieces)
    assert ast.literal_eval(locks) == {n: builder.carried_lock(n) for n in builder.LOCK_MODELS}


def test_model_stages_run_through_the_environment_python_with_a_clean_environment():
    run_cell = CODE["7.1 Forecast the validation period with each model"]
    assert "python = MODEL_PYTHONS[name]" in run_cell
    assert "str(python), str(runner)" in run_cell
    assert "env = model_environment_variables()" in run_cell
    assert '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"' in ENV_CELL
    assert 'env["MPLBACKEND"] = "Agg"' in ENV_CELL
    assert 'return WORK_ROOT / "envs" / name / "bin" / "python"' in ENV_CELL


def test_non_linux_runtime_stops_before_any_download(tmp_path, monkeypatch):
    import platform
    import urllib.request

    def refuse(*args, **kwargs):
        raise AssertionError("nothing may be downloaded")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    namespace = {"WORK_ROOT": tmp_path / "work", "SELECTED_MODELS": ["tirex", "chronos"]}
    with pytest.raises(RuntimeError, match="Linux x86-64 wheels"):
        exec(compile(ENV_CELL, "6.1", "exec"), namespace)
    assert not (tmp_path / "work").exists()


def test_generator_is_in_sync_and_logs_the_revision():
    assert NOTEBOOK.read_text(encoding="utf-8") == builder.serialized()
    log = NB["metadata"]["dimer"]["revision_log"]
    assert log[-1]["date"] == "2026-10-03" and "uv isolated environment" in log[-1]["change"]
