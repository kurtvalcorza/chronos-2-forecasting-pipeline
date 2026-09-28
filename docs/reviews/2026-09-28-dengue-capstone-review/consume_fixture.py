"""Separate-process bundle-consumer check. Review doubles only; no pretrained model or GPU.

Usage: python consume_fixture.py <fresh code dir> <results.zip> <workdir>
The code dir receives a copy of the notebook-embedded files only (no data.json/run_config.json).
"""
import pathlib
import shutil
import sys

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
code, bundle, workdir = map(pathlib.Path, sys.argv[1:4])
shutil.copytree(BASE / "embedded", code)
sys.path.insert(0, str(code))
import dengue_models as m  # noqa: E402
import dengue_runtime as r  # noqa: E402

assert pathlib.Path(r.__file__).resolve().parent == code.resolve()


class Dummy:
    pass


m.load_chronos = lambda *a: Dummy()
m.load_mitra = lambda *a: Dummy()
m.chronos_predict = lambda model, history, h: np.array(
    [[max(0, float(history[-1]) - 2), float(history[-1]), float(history[-1]) + 2] for _ in range(h)]
)
m.mitra_predict = lambda model, X, y, q: np.full(
    len(np.atleast_2d(q)), float(np.mean(y)) + (0.001 if np.asarray(X).shape[1] > 13 else 0)
)
r.free_gpu = lambda: None
sys.argv = ["dengue_runtime.py", "--consume", str(bundle), "--workdir", str(workdir),
            "--models", str(workdir / "models")]
r.main()
