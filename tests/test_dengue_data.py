"""Offline refusal tests for pinned acquisition and source-block schema."""

import importlib.util
import io
import math
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "dengue_data", Path(__file__).parents[1] / "tools/dengue_data.py"
)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def cohort():
    return [
        dict(year=y, block=b, cases=5, rain=1.5, temp=27.0, covid=int(y in (2020, 2021)))
        for y in range(2010, 2026)
        for b in range(1, 53)
    ]


def test_complete_order_and_year_boundary():
    rows = cohort()
    data.validate_qc(rows)
    assert (rows[51]["year"], rows[51]["block"]) == (2010, 52)
    assert (rows[52]["year"], rows[52]["block"]) == (2011, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cases", None),
        ("cases", -1),
        ("cases", 1.5),
        ("cases", math.inf),
        ("rain", None),
        ("rain", -1),
        ("temp", float("nan")),
        ("temp", -999),
        ("covid", 1),
    ],
)
def test_missing_invalid_never_becomes_zero(field, value):
    rows = cohort()
    rows[0][field] = value
    with pytest.raises(ValueError):
        data.validate_qc(rows)


@pytest.mark.parametrize("mutation", ["duplicate", "reorder", "missing", "week53"])
def test_invalid_cohort(mutation):
    rows = cohort()
    if mutation == "duplicate":
        rows[1] = rows[0].copy()
    elif mutation == "reorder":
        rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "missing":
        rows.pop()
    else:
        rows[51]["block"] = 53
    with pytest.raises(ValueError):
        data.validate_qc(rows)


def test_pin_rejects_equal_size_tamper(tmp_path, monkeypatch):
    path = tmp_path / "test.zip"
    path.write_bytes(b"bad")
    monkeypatch.setattr(data, "ARCHIVE_BYTES", 3)
    with pytest.raises(ValueError, match="digest"):
        data.verified_archive(path)


def test_pin_rejects_size_before_read(tmp_path):
    path = tmp_path / "test.zip"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="size"):
        data.verified_archive(path)


def test_zip_duplicate_member_and_bounded_read():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("expected", b"abc")
    with zipfile.ZipFile(stream) as z:
        assert data.member(z, "expected", 3) == b"abc"
        with pytest.raises(ValueError, match="oversized"):
            data.member(z, "expected", 2)
        with pytest.raises(ValueError, match="Missing"):
            data.member(z, "../escape")


def test_dictionary_schema_fails_closed(monkeypatch):
    monkeypatch.setattr(data, "workbook_tables", lambda _: {"wrong sheet": []})
    with pytest.raises(ValueError, match="sheet schema"):
        data.parse_workbook(b"")
