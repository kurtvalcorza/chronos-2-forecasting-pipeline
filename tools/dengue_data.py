"""Pinned QC source acquisition for an exploratory published-block benchmark.

Only standard-library modules are required. No dates or verified alignment are
inferred from the publisher's conflicting calendar descriptions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import posixpath
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ARCHIVE_URL = (
    "https://zenodo.org/api/records/21978184/files/dengue-environmental-dataset-main.zip/content"
)
ARCHIVE_BYTES = 37293635
ARCHIVE_SHA256 = "f9f6f85e2d035f3dd0c3c1fb56ea41fb46f6ce7fe2f1ebb20808d6692a39a412"
ARCHIVE_MD5 = "1400691b6d9e7f9116a5619b26c96d50"
PREFIX = "dengue-environmental-dataset-main/"
WORKBOOK_MEMBER = PREFIX + "Dengue_Environmental Vars_Dataset.xlsx"
WORKBOOK_SHA256 = "94c65ce24e13c132007f67da9b4d10f7023e9429fff79c358b1de938f982dd78"
SHEETS = ["Dataset Summary", "Data Dictionary", "QC Data", "Regional Data", "Multi-Setting Data"]
MAPPING = {
    "year": "YR",
    "block": "WN",
    "cases": "DC_QC",
    "rain": "RF_NASA",
    "temp": "Temp_ERA5-Land",
    "covid": "FLAG_COVID",
}
FLAGS = ["FLAG_COVID", "FLAG_SINGLE_CELL_RF", "FLAG_PLAUSIBILITY", "FLAG_PRESSURE_SH_GAP"]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_archive(path: Path) -> bytes:
    """Refuse altered cached data before parsing any ZIP or XML."""
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Pinned archive size mismatch")
    data = path.read_bytes()
    if digest(data) != ARCHIVE_SHA256 or hashlib.md5(data).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Pinned archive digest mismatch")
    return data


def download(destination: Path) -> None:
    """Bound the fixed public download; never retain an unverified final cache."""
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "DIMER-dengue-capstone/1"})
    temporary = destination.with_suffix(".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            if response.status != 200 or not response.url.startswith("https://zenodo.org/"):
                raise ValueError("Unexpected download response or redirect")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != ARCHIVE_BYTES:
                raise ValueError("Pinned download size mismatch")
            size = 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > ARCHIVE_BYTES:
                    raise ValueError("Download exceeded pinned byte limit")
                out.write(block)
        verified_archive(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def member(z: zipfile.ZipFile, name: str, limit: int = 8_000_000) -> bytes:
    """Read exactly one member without extracting archive paths."""
    matches = [x for x in z.infolist() if x.filename == name]
    if len(matches) != 1 or matches[0].file_size > limit:
        raise ValueError(f"Missing, duplicate or oversized member: {name}")
    return z.read(matches[0])  # zipfile checks CRC; archive has a stronger SHA pin.


def workbook_tables(data: bytes) -> dict[str, list[list]]:
    """Read sparse XLSX values, including rich strings and formatted empty rows."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        rels = ET.fromstring(member(z, "xl/_rels/workbook.xml.rels"))
        relationships = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        book = ET.fromstring(member(z, "xl/workbook.xml"))
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = [
                "".join(si.itertext()) for si in ET.fromstring(member(z, "xl/sharedStrings.xml"))
            ]
        result = {}
        for sheet in book.find("s:sheets", NS):
            rel = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = relationships[rel]
            name = (
                target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            )
            if not name.startswith("xl/worksheets/") or ".." in name.split("/"):
                raise ValueError("Unexpected worksheet relationship")
            rows = []
            for row in ET.fromstring(member(z, name)).findall("s:sheetData/s:row", NS):
                values = {}
                for cell in row.findall("s:c", NS):
                    letters = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                    column = 0
                    for letter in letters:
                        column = column * 26 + ord(letter) - 64
                    if cell.find("s:f", NS) is not None:
                        raise ValueError("Formula cells require an explicit evaluation policy")
                    raw = cell.find("s:v", NS)
                    kind = cell.attrib.get("t", "n")
                    if kind == "inlineStr":
                        value = "".join(cell.find("s:is", NS).itertext())
                    elif raw is None or raw.text is None:
                        value = None
                    elif kind == "s":
                        value = shared[int(raw.text)]
                    elif kind in ("str", "e"):
                        value = raw.text
                    else:
                        value = float(raw.text)
                    values[column - 1] = value
                if any(value is not None for value in values.values()):
                    rows.append([values.get(i) for i in range(max(values) + 1)])
            result[sheet.attrib["name"]] = rows
        return result


def validate_qc(records: list[dict]) -> None:
    """Require the exact published cohort and never coerce missing values to zero."""
    if len(records) != 832:
        raise ValueError("Expected 832 nonempty QC records")
    expected = [(year, block) for year in range(2010, 2026) for block in range(1, 53)]
    if [(r.get("year"), r.get("block")) for r in records] != expected:
        raise ValueError("QC keys must be unique, ordered and complete: 52 blocks/year")
    for row in records:
        for field in MAPPING:
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Missing or nonfinite {field}; never replace with zero")
        if row["cases"] < 0 or row["cases"] != int(row["cases"]):
            raise ValueError("Cases must be nonnegative integer counts")
        if row["rain"] < 0 or not -30 <= row["temp"] <= 60:
            raise ValueError("Weather outside declared physical bounds")
        if row["covid"] != int(row["year"] in (2020, 2021)):
            raise ValueError("COVID flag disagrees with published years")


def parse_workbook(data: bytes) -> tuple[list[dict], dict]:
    tables = workbook_tables(data)
    if list(tables) != SHEETS:
        raise ValueError("Workbook sheet schema changed")
    dictionary = {row[0]: row for row in tables["Data Dictionary"][2:]}
    required_units = {
        "DC_QC": "Count (integer)",
        "RF_NASA": "mm per week (float)",
        "Temp_ERA5-Land": "degrees Celsius (float)",
    }
    for key, unit in required_units.items():
        if key not in dictionary or dictionary[key][5] != unit:
            raise ValueError(f"Dictionary unit mismatch: {key}")
    if "Weekly confirmed and probable dengue cases" not in dictionary["DC_QC"][2]:
        raise ValueError("Target meaning changed")
    header, *raw_rows = tables["QC Data"]
    if (
        len(header) != 24
        or len(set(header)) != 24
        or not set(MAPPING.values()).union(FLAGS).issubset(header)
    ):
        raise ValueError("QC column schema changed")
    source_rows = [
        dict(zip(header, row + [None] * (len(header) - len(row)), strict=True)) for row in raw_rows
    ]
    rows = []
    for source in source_rows:
        row = {key: source[column] for key, column in MAPPING.items()}
        row["flags"] = {key: source[key] for key in FLAGS}
        if any(value not in (0, 1) for value in row["flags"].values()):
            raise ValueError("Invalid quality flag")
        rows.append(row)
    validate_qc(rows)
    for row in rows:
        for key in ("year", "block", "cases", "covid"):
            row[key] = int(row[key])
        row["flags"] = {key: int(value) for key, value in row["flags"].items()}
    summary_note = next(row[1] for row in tables["Dataset Summary"] if row[0] == "Weekly alignment")
    dictionary_note = dictionary["WN"][2]
    if (
        "continuous 7-day blocks" not in summary_note
        or "ISO epidemiological" not in dictionary_note
    ):
        raise ValueError("Calendar evidence changed; review approved exploratory scope")
    audit = {
        "format_version": 1,
        "scope": "exploratory_source_block_benchmark",
        "calendar_verified": False,
        "weather_alignment_verified": False,
        "prospective_availability_verified": False,
        "qualification": "Candidate: exploratory cohort only; calendar gate unresolved",
        "sheet_names": SHEETS,
        "row_count": len(rows),
        "column_mapping": MAPPING,
        "dictionary": {
            key: dict(zip(tables["Data Dictionary"][1], dictionary[key], strict=True))
            for key in MAPPING.values()
        },
        "calendar_evidence": {
            "summary": summary_note,
            "dictionary": dictionary_note,
            "resolution": "Retain original YR/WN order; no actual start/end dates inferred",
        },
        "missingness_by_year": {
            str(y): {h: sum(r[h] is None for r in source_rows if r["YR"] == y) for h in header}
            for y in range(2010, 2026)
        },
        "flags_by_year": {
            str(y): {h: int(sum(r[h] for r in source_rows if r["YR"] == y)) for h in FLAGS}
            for y in range(2010, 2026)
        },
        "annual_cases": {
            str(y): sum(r["cases"] for r in rows if r["year"] == y) for y in range(2010, 2026)
        },
        "ranges": {
            key: [min(r[key] for r in rows), max(r[key] for r in rows)]
            for key in ("cases", "rain", "temp")
        },
        "exclusions": [],
        "transformations": [
            "Select QC sheet only",
            "Rename six explicitly mapped fields; retain original quality flags",
            "Retain all 832 rows and original values; "
            "no imputation, clipping or date reconstruction",
        ],
        "limitations": [
            "Source labels describe block-specific reported counts, "
            "not rates or cumulative totals; "
            "underlying case line lists were not independently audited",
            "Conflicting calendar descriptions and case/weather alignment remain unresolved",
            "No case revision or publication-time vintage archive is provided in this release",
            "Zero assumed reporting delay is an experimental assumption",
            "IMERG Final Run and ERA5-Land reanalysis are retrospective products; "
            "event-date truncation does not reconstruct historical availability",
            "Single centroid weather cells do not represent all within-city variability",
            "COVID-period flag retained; no post-outcome exclusion",
            "Foundation-model pretraining overlap is unknown",
        ],
    }
    return rows, audit


def acquire(root: Path, archive_path: Path | None = None) -> list[dict]:
    """Write verified rows, an audit and full publisher notices under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(archive_path) if archive_path is not None else root / "cache" / "dataset.zip"
    if not cache.exists():
        if archive_path is not None:
            raise FileNotFoundError(cache)
        download(cache)
    archive = verified_archive(cache)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        workbook = member(z, WORKBOOK_MEMBER, 1_028_105)
        if len(workbook) != 1_028_105 or digest(workbook) != WORKBOOK_SHA256:
            raise ValueError("Workbook pin mismatch")
        rows, audit = parse_workbook(workbook)
        readme = member(z, PREFIX + "README.md").decode("utf-8")
        licence = member(z, PREFIX + "LICENSE").decode("utf-8")
        validation = member(z, PREFIX + "Dengue_Environmental Vars_Validation.R").decode("utf-8")
        audit["source"] = {
            "release": "https://zenodo.org/records/21978184",
            "archive_url": ARCHIVE_URL,
            "archive_bytes": len(archive),
            "archive_sha256": digest(archive),
            "archive_md5": ARCHIVE_MD5,
            "workbook_member": WORKBOOK_MEMBER,
            "workbook_sha256": digest(workbook),
            "members": z.namelist(),
            "readme_sha256": digest(readme.encode()),
            "license_sha256": digest(licence.encode()),
            "validation_script_sha256": digest(validation.encode()),
        }
        audit["calendar_evidence"]["validation_script_lines"] = [
            line
            for line in validation.splitlines()
            if "not ISO-week validation" in line or "Data Dictionary calls WN ISO" in line
        ]
        audit["licence"] = {
            "database": "ODC-ODbL-1.0",
            "url": "https://opendatacommons.org/licenses/odbl/1.0/",
            "publisher_citation_doi": "10.5281/zenodo.19448854",
            "pinned_release_doi": "10.5281/zenodo.21978184",
            "source_terms": "Retain source-specific surveillance and meteorological licensing "
            "and redistribution terms from publisher README; no new permissions inferred",
            "notice": "DATA_LICENSE.md contains original publisher LICENSE and complete README "
            "including creators and source citations",
        }
    encoded = json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    audit["derived_data_sha256"] = digest(encoded.encode("utf-8"))
    (root / "data.json").write_text(encoded, encoding="utf-8", newline="\n")
    (root / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    notice = (
        "# Data attribution and licence\n\nDerived QC table: selected from release "
        "https://doi.org/10.5281/zenodo.21978184. The publisher's citation DOI below is "
        "retained separately. Credit Project NOAH and the named contributors.\n\n"
        "Transformations: select QC sheet; retain original order and numerical values; "
        "rename selected fields; preserve quality flags. No calendar reconstruction or "
        "imputation. The derived database is ODC-ODbL 1.0, separate from the notebook/code "
        "licence. Source-specific terms remain applicable. The complete archive is not "
        "part of the results bundle.\n\n## Original publisher licence\n\n"
        + licence
        + "\n\n## Original publisher README, citations and source notices\n\n"
        + readme
    )
    (root / "DATA_LICENSE.md").write_text(notice, encoding="utf-8", newline="\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = acquire(args.root, args.archive)
    print(f"Verified {len(result)} source-block rows; calendar and alignment remain unverified.")
