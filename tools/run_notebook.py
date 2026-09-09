"""Execute pure-Python notebook code cells sequentially in the current environment.

This is intentionally smaller than Jupyter/nbconvert. Tutorial notebooks in
this repository use ordinary Python cells only; CI executes those exact cell
sources in one shared namespace. Any shell/magic syntax fails loudly instead of
being silently skipped.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    raise TypeError(f"unsupported notebook cell source type: {type(source).__name__}")


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
        "NOTEBOOK_CI": True,
    }
    old_cwd = Path.cwd()
    try:
        os.chdir(path.resolve().parents[1])
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = _source(cell)
            if not source.strip():
                continue
            for line in source.splitlines():
                stripped = line.lstrip()
                if stripped.startswith(("!", "%")):
                    raise RuntimeError(
                        f"{path}: cell {index} uses Jupyter shell/magic syntax; "
                        "tutorial CI requires ordinary Python cells."
                    )
            code = compile(source, f"{path}:cell-{index}", "exec")
            exec(code, namespace, namespace)
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args()
    for notebook in args.notebooks:
        print(f"[notebook-smoke] {notebook}")
        execute_notebook(notebook)


if __name__ == "__main__":
    main()
