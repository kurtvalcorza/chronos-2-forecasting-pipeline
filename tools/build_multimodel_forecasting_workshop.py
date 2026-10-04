# ruff: noqa: E501,I001
"""Generate the DIMER multi-model time-series forecasting workshop notebook.

NOTEBOOK_SPEC 2.1. Pure-stdlib generator so generation itself adds no project dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodel_forecasting_workshop_source import CELLS

NOTEBOOK_NAME = "DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb"
TOOLS = Path(__file__).resolve().parent
LOCK_MODELS = ("tirex", "chronos", "toto")
LOCK_PLACEHOLDER = "ENV_LOCKS = {}  # @carried-locks"
MAX_PIECE = 1000

def lock_file(name):
    return TOOLS / f"forecasting-workshop-{name}-requirements.lock"

def carried_lock(name):
    """The lock as the notebook carries it: requirement and hash lines only, no comments."""
    lines = [
        line.rstrip()
        for line in lock_file(name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return "\n".join(lines) + "\n"

def render_locks():
    """ENV_LOCKS as parenthesised runs of string pieces, one lock line per piece."""
    out = ["ENV_LOCKS = {"]
    for name in LOCK_MODELS:
        out.append(f"    {json.dumps(name)}: (")
        for line in carried_lock(name).splitlines(keepends=True):
            piece = json.dumps(line, ensure_ascii=True)
            if len(piece) > MAX_PIECE:
                raise ValueError(f"lock piece over {MAX_PIECE} characters in {name}")
            out.append(f"        {piece}")
        out.append("    ),")
    out.append("}")
    return "\n".join(out)

def cell_source(cell):
    source = cell["source"]
    if LOCK_PLACEHOLDER in source:
        if source.count(LOCK_PLACEHOLDER) != 1:
            raise ValueError("the lock placeholder must appear once")
        source = source.replace(LOCK_PLACEHOLDER, render_locks())
    return source

def build_notebook():
    rendered = []
    for index, cell in enumerate(CELLS):
        if cell["kind"] == "markdown":
            rendered.append({
                "cell_type": "markdown",
                "id": f"dimer-ts-workshop-{index:02d}",
                "metadata": cell.get("metadata", {}),
                "source": cell_source(cell).splitlines(keepends=True),
            })
        elif cell["kind"] == "code":
            rendered.append({
                "cell_type": "code",
                "execution_count": None,
                "id": f"dimer-ts-workshop-{index:02d}",
                "metadata": cell.get("metadata", {}),
                "outputs": [],
                "source": cell_source(cell).splitlines(keepends=True),
            })
        else:
            raise ValueError(f"Unknown cell kind: {cell['kind']}")
    return {
        "cells": rendered,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "dimer": {
                "notebook_spec": "2.1",
                "notebook_profile": "TASK-INFERENCE",
                "notebook_mode": "WORKSHOP",
                "standalone": True,
                "capability": "multi-model-time-series-forecasting",
                "carrier": "self-contained multi-model comparative forecasting workshop",
                "dataset": "chronos_multi_series synthetic fixture",
                "default_tier": "STANDARD",
                "canonical_runtime": "NVIDIA Tesla T4",
                "worker_required": False,
                "credentials_required": False,
                "clean_runtime_evidence": "pending",
                "release_status": "candidate",
                "revision_log": [
                    {
                        "date": "2026-10-03",
                        "change": "uv isolated environment: pinned uv 0.12.15 wheel (SHA-256 checked), "
                        "managed CPython 3.12.12 per model, hash-locked wheels-only installs "
                        "(--require-hashes --only-binary :all:), no kernel install and no restart; "
                        "Linux x86-64 only. Direct model pins unchanged.",
                        "locks": [f"tools/forecasting-workshop-{name}-requirements.lock" for name in LOCK_MODELS],
                    }
                ],
                "generated_from": {
                    "repository": "kurtvalcorza/chronos-2-forecasting-pipeline",
                    "source": "tools/multimodel_forecasting_workshop_source.py",
                    "generator": "tools/build_multimodel_forecasting_workshop.py",
                },
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "workshop_revision": "0.2.0-candidate",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def serialized():
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out or repo / "tutorials" / NOTEBOOK_NAME
    content = serialized()
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != content:
            raise SystemExit(f"STALE: {out}; run python tools/build_multimodel_forecasting_workshop.py")
        print(f"OK: {out}")
        return 0
    out.write_text(content, encoding="utf-8", newline="\n")
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
