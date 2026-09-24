"""Generate the DIMER multi-model time-series forecasting workshop notebook.

NOTEBOOK_SPEC 2.1. Pure-stdlib generator so generation itself adds no project dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodel_forecasting_workshop_source import CELLS

NOTEBOOK_NAME = "DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb"

def build_notebook():
    rendered = []
    for index, cell in enumerate(CELLS):
        if cell["kind"] == "markdown":
            rendered.append({
                "cell_type": "markdown",
                "id": f"dimer-ts-workshop-{index:02d}",
                "metadata": {},
                "source": cell["source"].splitlines(keepends=True),
            })
        elif cell["kind"] == "code":
            rendered.append({
                "cell_type": "code",
                "execution_count": None,
                "id": f"dimer-ts-workshop-{index:02d}",
                "metadata": {},
                "outputs": [],
                "source": cell["source"].splitlines(keepends=True),
            })
        else:
            raise ValueError(f"Unknown cell kind: {cell['kind']}")
    return {
        "cells": rendered,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "dimer": {
                "notebook_profile": "TASK-INFERENCE",
                "notebook_mode": "WORKSHOP",
                "notebook_spec": "2.1",
                "standalone": True,
                "carrier": "self-contained multi-model comparative workshop using pinned upstream packages in isolated environments",
                "clean_runtime_evidence": "pending",
                "generated_from": {
                    "repository": "kurtvalcorza/chronos-2-forecasting-pipeline",
                    "source": "tools/multimodel_forecasting_workshop_source.py",
                    "generator": "tools/build_multimodel_forecasting_workshop.py",
                },
            },
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
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
    out.write_text(content, encoding="utf-8")
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
