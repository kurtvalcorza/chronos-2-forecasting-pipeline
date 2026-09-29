"""Pinned upstream Chronos adapter, carried visibly in the standalone notebook."""

from __future__ import annotations

import hashlib
import os
import random
import urllib.request
from pathlib import Path

import numpy as np


def reef_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reef_snapshot(manifest: dict, cache: Path) -> Path:
    """Download only declared immutable weight/config files and verify cache hits."""
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("An immutable model revision is required")
    if manifest["modelId"] != "amazon/chronos-2":
        raise ValueError("Unexpected model identity")
    root = cache / manifest["revision"]
    root.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        if item["path"] not in {"config.json", "model.safetensors"}:
            raise ValueError("Only configuration and safetensors are allowed")
        path = root / item["path"]
        if path.is_symlink() or root.is_symlink():
            raise ValueError("Model cache must not use symlinks")
        if not path.exists():
            url = (
                f"https://huggingface.co/{manifest['modelId']}/resolve/"
                f"{manifest['revision']}/{item['path']}"
            )
            temporary = path.with_suffix(".partial")
            total = 0
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(1 << 20):
                    total += len(chunk)
                    if total > item["bytes"]:
                        raise ValueError("Model download exceeds pinned size")
                    out.write(chunk)
            if total != item["bytes"] or reef_sha(temporary) != item["sha256"]:
                raise ValueError("Model download integrity failure")
            temporary.replace(path)
        if path.stat().st_size != item["bytes"] or reef_sha(path) != item["sha256"]:
            raise ValueError(f"Model integrity failure: {item['path']}")
    return root


def reef_load_model(manifest: dict, cache: Path):
    """Hosted GPU execution only; never silently substitute a different model."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from importlib.metadata import version

    import torch
    from chronos import Chronos2Pipeline

    if version("chronos-forecasting") != "2.3.1":
        raise RuntimeError("Expected chronos-forecasting 2.3.1")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab T4 GPU for model stages")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    root = reef_snapshot(manifest, cache)
    return Chronos2Pipeline.from_pretrained(
        str(root),
        device_map="cuda",
        dtype=torch.float32,
        local_files_only=True,
    )


def reef_predict(model, history, horizon: int = 28) -> np.ndarray:
    """Return H by 3 raw q10/q50/q90 as the model emits them; one series per call."""
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < 84:
        raise ValueError("Expected a finite one-dimensional daily history")
    if not 1 <= horizon <= 30:
        raise ValueError("Supported horizon is 1–30 days")
    quantiles, _ = model.predict_quantiles(
        [values.copy()],
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=1,
        context_length=len(values),
    )
    if len(quantiles) != 1:
        raise ValueError("Unexpected series count")
    result = quantiles[0].detach().float().cpu().numpy()
    if result.shape != (1, horizon, 3) or not np.isfinite(result).all():
        raise ValueError("Invalid quantile dimensions or nonfinite prediction")
    # Quantiles are returned in model order; crossings are rearranged and recorded downstream.
    return result[0].astype(float)
