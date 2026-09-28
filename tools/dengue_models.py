"""Pinned upstream inference adapters; no repository imports or learned-weight export.

Mitra uses AutoGluon 1.5.0's Tab2D and inference preprocessor directly. Every
call builds a fresh preprocessor from support rows; the frozen network is shared.
Persist raw support arrays to reconstruct this state, never pickle a trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.request
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(manifest: dict, key: str) -> dict:
    return manifest["models"][key] if "models" in manifest else manifest


def stage_snapshot(manifest: dict, cache: str | Path) -> Path:
    """Stage one snapshot; verify sizes/digests even on cache hits, refusing tampering."""
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Snapshot requires an immutable 40-character revision")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", manifest["modelId"]):
        raise ValueError("Invalid model repository ID")
    root = Path(cache).resolve() / manifest["revision"]
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("Snapshot directory must not be a symbolic link")
    seen = set()
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in item["path"]
            or ":" in item["path"]
            or item["path"] in seen
        ):
            raise ValueError("Unsafe or duplicate snapshot path")
        seen.add(item["path"])
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Snapshot path escapes cache")
        if not isinstance(item["bytes"], int) or not 0 < item["bytes"] < 2 * 1024**3:
            raise ValueError("Invalid snapshot size")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid snapshot digest")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                f"https://huggingface.co/{manifest['modelId']}/resolve/"
                f"{manifest['revision']}/{item['path']}"
            )
            temporary = path.with_name(path.name + ".partial")
            count = 0
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    temporary.open("wb") as out,
                ):
                    while chunk := response.read(1 << 20):
                        count += len(chunk)
                        if count > item["bytes"]:
                            raise ValueError("Snapshot download exceeds pinned size")
                        out.write(chunk)
                if count != item["bytes"] or _sha(temporary) != item["sha256"]:
                    raise ValueError("Downloaded snapshot integrity failure")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise ValueError(f"Cached snapshot integrity failure: {item['path']}")
    return root


def _require(package: str, expected: str) -> None:
    if version(package) != expected:
        raise RuntimeError(f"This adapter requires {package}=={expected}")


def _seed() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_chronos(cache: str | Path, manifest: dict) -> Any:
    """Load the byte-verified Chronos-2 snapshot locally on the hosted CUDA device."""
    _require("chronos-forecasting", "2.3.1")
    _seed()
    import torch
    from chronos import Chronos2Pipeline

    path = stage_snapshot(_snapshot(manifest, "chronos"), cache)
    if not torch.cuda.is_available():
        raise RuntimeError("Run model stages in the hosted Colab GPU runtime")
    return Chronos2Pipeline.from_pretrained(
        str(path), device_map="cuda", dtype=torch.float32, local_files_only=True
    )


def chronos_predict(model: Any, history: Any, horizon: int) -> np.ndarray:
    """Forecast ordered reporting blocks; return raw q0.1/q0.5/q0.9 as H by 3."""
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Chronos history must be a finite one-dimensional sequence")
    if not isinstance(horizon, int) or not 1 <= horizon <= 6:
        raise ValueError("This experiment supports effective horizons 1 through 6")
    quantiles, _ = model.predict_quantiles(
        [values.copy()],
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=1,
        context_length=len(values),
    )
    if len(quantiles) != 1:
        raise ValueError("Chronos returned an unexpected number of series")
    result = quantiles[0].detach().float().cpu().numpy()
    if result.shape != (1, horizon, 3) or not np.isfinite(result).all():
        raise ValueError("Chronos quantile shape or finiteness failure")
    if (np.diff(result, axis=-1) < 0).any():
        raise ValueError("Chronos produced crossing quantiles")
    return result[0].astype(float)


def load_mitra(cache: str | Path, manifest: dict) -> Any:
    """Load frozen Tab2D directly, avoiding upstream's mutable default HF revision."""
    _require("autogluon.tabular", "1.5.0")
    _seed()
    import torch
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
    from safetensors.torch import load_file

    path = stage_snapshot(_snapshot(manifest, "mitra"), cache)
    if not torch.cuda.is_available():
        raise RuntimeError("Run model stages in the hosted Colab GPU runtime")
    config = json.loads((path / "config.json").read_text())
    model = Tab2D(**config, use_pretrained_weights=False, path_to_weights="", device="cuda")
    model.load_state_dict(load_file(path / "model.safetensors", device="cpu"), strict=True)
    model.to("cuda").eval()
    return model


def mitra_predict(model: Any, X: Any, y_log: Any, Xquery: Any) -> np.ndarray:
    """Fresh context-only conditioning; return log-target estimates without clipping.

    No call to trainer.train: its default would perform validation and possibly
    gradient updates. Only the upstream preprocessor.fit and predict path run.
    """
    support = np.array(X, dtype=np.float64, copy=True)
    target = np.array(y_log, dtype=np.float64, copy=True)
    query = np.array(Xquery, dtype=np.float64, copy=True)
    if (
        support.ndim != 2
        or query.ndim != 2
        or target.shape != (len(support),)
        or support.shape[1] != query.shape[1]
        or not 2 <= len(support) <= 8192
        or not 1 <= support.shape[1] <= 100
        or not len(query)
    ):
        raise ValueError("Mitra requires compatible support/target/query arrays")
    if not all(np.isfinite(a).all() for a in (support, target, query)):
        raise ValueError("Mitra inputs must be finite; impute from support only beforehand")
    if np.ptp(target) == 0 or not np.any(np.ptp(support, axis=0) > 0):
        raise ValueError(
            "Mitra requires nonconstant support targets and at least one varying feature"
        )
    _seed()
    from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor, TrainerFinetune

    device = str(next(model.parameters()).device)
    adapter = MitraRegressor(
        device=device,
        fine_tune=False,
        fine_tune_steps=0,
        n_estimators=1,
        random_mirror_x=False,
        random_mirror_regression=False,
        shuffle_features=False,
        use_random_transforms=False,
        seed=42,
        verbose=False,
    )
    config, _ = adapter._create_config("regression", 1)
    # T4 lacks native bfloat16. Use stable float32, with autocast disabled by
    # torch for this unsupported autocast dtype; no numerical precision claim.
    config.hyperparams["precision"] = "float32"
    trainer = TrainerFinetune(
        config, model, n_classes=0, device=device, rng=np.random.RandomState(42), verbose=False
    )
    trainer.preprocessor.fit(support, target)
    trainer.post_fit_optimize()
    prediction = np.asarray(trainer.predict(support, target, query), dtype=float).reshape(-1)
    if prediction.shape != (len(query),) or not np.isfinite(prediction).all():
        raise ValueError("Mitra returned invalid predictions")
    return prediction
