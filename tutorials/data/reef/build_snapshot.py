"""Package the audited NOAA snapshot without downloading or modifying source data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(source_dir: Path, output_dir: Path) -> dict:
    """Verify audited source bytes and create a deterministic data-only archive."""
    audit_bytes = (source_dir / "audit.json").read_bytes()
    audit = json.loads(audit_bytes)
    members = []
    content = {}
    for station in sorted(audit["stations"], key=lambda item: item["region"]):
        filename = f"{station['region']}_philippines.txt"
        raw = (source_dir / filename).read_bytes()
        if len(raw) != station["bytes"] or digest(raw) != station["sha256"]:
            raise ValueError(f"Audited source checksum or size mismatch: {filename}")
        lines = raw.decode("utf-8").splitlines()
        heading = next(index for index, line in enumerate(lines) if line.startswith("YYYY MM DD "))
        rows = [line.split() for line in lines[heading + 1:] if line.strip()]
        if len(rows) != station["rows"]:
            raise ValueError(f"Audited row count mismatch: {filename}")
        content[filename] = raw
        members.append({
            "filename": filename,
            "region_id": station["region"],
            "source_url": station["url"],
            "bytes": len(raw),
            "sha256": digest(raw),
            "rows": len(rows),
            "start": station["start"],
            "end": station["end"],
            "missing_dates": station["missing_dates"],
        })
    if set(content) != {f"{region}_philippines.txt" for region in
                        ("northern", "central", "western", "eastern", "southern")}:
        raise ValueError("Snapshot must contain exactly five Philippine regions")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for filename, raw in sorted(content.items()):
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    archive_bytes = buffer.getvalue()
    archive_name = "noaa_crw_philippines_2026-09-28.zip"
    manifest = {
        "schema_version": 1,
        "dataset_id": "noaa-crw-rvs-v3.1-philippines-2026-09-28",
        "product": "NOAA Coral Reef Watch Regional Virtual Stations",
        "product_version": "3.1",
        "retrieval_utc": audit["retrieved_utc"],
        "coverage_start": "1985-01-01",
        "coverage_end": "2026-09-26",
        "archive": {"filename": archive_name, "bytes": len(archive_bytes),
                    "sha256": digest(archive_bytes), "format": "zip"},
        "members": members,
        "original_audit": {"filename": "audit.json", "bytes": len(audit_bytes),
                           "sha256": digest(audit_bytes)},
        "rights": {"status": "Public domain NOAA Coral Reef Watch website content",
                   "source_url": "https://coralreefwatch.noaa.gov/satellite/docs/recommendations_crw_citation.php",
                   "attribution": "NOAA Coral Reef Watch",
                   "separate_from_code_license": True},
        "citation": "NOAA Coral Reef Watch. 2019. NOAA Coral Reef Watch Version 3.1 Daily Global 5km Satellite Coral Bleaching Heat Stress Monitoring Products: Regional Virtual Stations, Northern, Central, Western, Eastern and Southern Philippines; 1 January 1985 to 26 September 2026. NOAA Coral Reef Watch, College Park, Maryland, USA. Accessed 28 September 2026. https://coralreefwatch.noaa.gov/product/vs/data.php",
        "scientific_contract": {
            "interpretation": "Daily moving-pixel regional 90th-percentile HotSpot summary; not a fixed sensor, regional-mean SST, or observed bleaching label",
            "dhw_first_valid_date": "1985-03-25",
            "baa_first_valid_date": "1985-03-31",
            "dhw_formula": "sum(HotSpot if HotSpot >= 1 C else 0) over 84 calendar days / 7",
            "missing_policy": "Reindex daily; preserve gaps; never zero-fill missing observations",
            "alert_scheme": "Legacy BAA; not interchangeable with current global alert categories",
        },
        "distribution": {
            "default": "Notebook embeds this exact ZIP as base64 data and verifies archive/member hashes before parsing",
            "spec_amendment": "An embedded immutable data-only archive replaces the proposed remote immutable asset download, avoiding an unpublished URL. No executable source is fetched; no live-data fallback.",
            "live_refresh": "Different dataset version requiring a new audit and manifest; not a default fallback",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / archive_name).write_bytes(archive_bytes)
    (output_dir / "audit.json").write_bytes(audit_bytes)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.source_dir, arguments.output_dir)["archive"], indent=2))
