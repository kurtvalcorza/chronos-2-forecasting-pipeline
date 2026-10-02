"""Capstone notebooks carry files as short string pieces, never one huge source line."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPSTONES = [
    "tutorials/DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb",
    "tutorials/DIMER_Philippine_Dengue_Forecasting_Capstone.ipynb",
]
MAX_LINE = 2000


def load_carrier():
    spec = importlib.util.spec_from_file_location(
        "notebook_carrier_test", ROOT / "tools/notebook_carrier.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", CAPSTONES)
def test_no_capstone_cell_line_exceeds_limit(path):
    notebook = json.loads((ROOT / path).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = cell["source"]
        text = "".join(source) if isinstance(source, list) else source
        longest = max(len(line) for line in text.split("\n"))
        assert longest <= MAX_LINE, (cell.get("id"), longest)


SAMPLES = ["", "short", "x" * 2500 + "\n", "one\ntwo\r\nthree\n" + "y" * 2500 + "\nlast"]


@pytest.mark.parametrize("text", SAMPLES)
def test_carried_literal_round_trips(text):
    literal = load_carrier().carried_literal(text)
    assert ast.literal_eval(literal) == text
    assert max(len(line) for line in literal.split("\n")) <= MAX_LINE


def test_carried_dict_round_trips():
    files = {f"file{i}.txt": text for i, text in enumerate(SAMPLES)}
    literal = load_carrier().carried_dict(files)
    assert ast.literal_eval(literal) == files
    assert max(len(line) for line in literal.split("\n")) <= MAX_LINE
