"""Pinned, integrity-verified Chronos-2 loader.

One model, one revision, checked four ways before any weight is deserialised:

1. the requested source must be exactly ``amazon/chronos-2`` at the pinned commit;
2. the directory Hugging Face resolved must be that commit's snapshot;
3. ``model.safetensors`` must match the recorded SHA-256 and byte size;
4. ``config.json`` must match the recorded SHA-256.

Pickle-based weights (``.bin``/``.pt``/``.pth``/``.ckpt``) in the snapshot are a
hard refusal, not a fallback.

The revision SHA is the primary integrity anchor (RFC C-7): it covers repository
configuration as well as the weight file. The digests are secondary, file-level
assertions that additionally catch a corrupted or truncated download.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ModelIntegrityError, ModelSourceError

__all__ = [
    "PINNED_MODEL_ID",
    "PINNED_REVISION",
    "PINNED_WEIGHTS_SHA256",
    "PINNED_WEIGHTS_BYTES",
    "PINNED_CONFIG_SHA256",
    "PINNED_LICENSE",
    "PINNED_LICENSE_SOURCE",
    "PINNED_MODEL_URL",
    "EXPECTED_TRAINED_QUANTILES",
    "WEIGHTS_FILENAME",
    "CONFIG_FILENAME",
    "REFUSED_WEIGHT_SUFFIXES",
    "ModelIdentity",
    "LoadedModel",
    "sha256_file",
    "check_model_source",
    "verify_snapshot",
    "resolve_hub_revision",
    "load_pinned_model",
]

# --------------------------------------------------------------------------
# Supply-chain pins. Independently verified against the Hugging Face Hub on
# 2026-09-08; see MODEL_CARD.md for the provenance record.
# --------------------------------------------------------------------------

PINNED_MODEL_ID = "amazon/chronos-2"
PINNED_REVISION = "95a9710e2596287d08352589f42634fa5abdf0a7"
PINNED_WEIGHTS_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
PINNED_WEIGHTS_BYTES = 477_930_472
PINNED_CONFIG_SHA256 = "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"

PINNED_LICENSE = "apache-2.0"
#: There is no LICENSE file in the Hugging Face repository at the pinned
#: revision — the files present are .gitattributes, README.md, config.json and
#: model.safetensors. The licence is declared in the model card metadata only,
#: and this string is what gets exported rather than a claim about a file.
PINNED_LICENSE_SOURCE = (
    f"apache-2.0 declared in the Hugging Face model card metadata of "
    f"{PINNED_MODEL_ID} at revision {PINNED_REVISION}; the repository contains "
    f"no LICENSE file at that revision"
)
PINNED_MODEL_URL = f"https://huggingface.co/{PINNED_MODEL_ID}/tree/{PINNED_REVISION}"

#: The 21-level grid Chronos-2 was trained on, read from the pinned config.json.
#: Held here only so a test can assert the loaded pipeline still reports it; the
#: runtime value always comes from the model, never from this constant.
EXPECTED_TRAINED_QUANTILES: tuple[float, ...] = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)  # fmt: skip

WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"

#: Pickle-format checkpoints execute arbitrary code on load. If any appear in
#: the snapshot the load is refused rather than silently preferring safetensors.
REFUSED_WEIGHT_SUFFIXES: tuple[str, ...] = (".bin", ".pt", ".pth", ".ckpt", ".pkl")

_MUTABLE_REFS = frozenset({"main", "master", "latest", "head", "HEAD"})
_URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:[\\/]")
_SHA1_HEX = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelIdentity:
    """Everything a downstream export needs to say which model produced a result."""

    name: str
    revision: str
    config_sha256: str
    weights_sha256: str
    weights_bytes: int
    license: str
    license_source: str
    source_url: str
    trained_quantiles: tuple[float, ...]
    model_context_length: int
    model_prediction_length: int
    snapshot_path: str = field(default="", compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
            "weights_bytes": self.weights_bytes,
            "license": self.license,
            "license_source": self.license_source,
            "source_url": self.source_url,
            "trained_quantiles": list(self.trained_quantiles),
            "model_context_length": self.model_context_length,
            "model_prediction_length": self.model_prediction_length,
        }


@dataclass(frozen=True)
class LoadedModel:
    """A loaded pipeline plus the identity that was proven before loading it."""

    pipeline: Any
    identity: ModelIdentity
    device: str
    dtype: str

    @property
    def trained_quantiles(self) -> tuple[float, ...]:
        return self.identity.trained_quantiles

    @property
    def model_context_length(self) -> int:
        return self.identity.model_context_length

    @property
    def model_prediction_length(self) -> int:
        return self.identity.model_prediction_length


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed so a 478 MB weight file is cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_local_path(model_id: str) -> bool:
    if _WINDOWS_DRIVE.match(model_id):
        return True
    if "\\" in model_id:
        return True
    if model_id.startswith(("/", "~", "./", "../", ".\\", "..\\")):
        return True
    if model_id in (".", ".."):
        return True
    # A bare "a/b" repo id that also happens to exist on disk is still a path.
    try:
        return Path(model_id).exists()
    except OSError:  # pragma: no cover - pathological names only
        return False


def check_model_source(model_id: Any, revision: Any) -> None:
    """Refuse anything but the single pinned ``(model_id, revision)`` pair.

    Raises
    ------
    ModelSourceError
        For URI schemes (``s3://``, ``hf://``, ``http://``, ``file://`` ...),
        local filesystem paths, any repo id other than ``amazon/chronos-2``,
        mutable refs (``main``, ``latest``), and any other revision.
    """
    if not isinstance(model_id, str) or not model_id:
        raise ModelSourceError(
            f"model_id must be the string {PINNED_MODEL_ID!r}; got {model_id!r}."
        )

    scheme = _URI_SCHEME.match(model_id)
    if scheme:
        raise ModelSourceError(
            f"Refusing model source {model_id!r}: URI schemes such as "
            f"{scheme.group(0)!r} are not permitted in the standard path. Only "
            f"the pinned Hugging Face repo {PINNED_MODEL_ID!r} at revision "
            f"{PINNED_REVISION} is accepted."
        )

    if _looks_like_local_path(model_id):
        raise ModelSourceError(
            f"Refusing model source {model_id!r}: local filesystem paths are not "
            f"permitted in the standard path, because they carry no revision to "
            f"verify. Only {PINNED_MODEL_ID!r} at revision {PINNED_REVISION} is "
            f"accepted."
        )

    if model_id != PINNED_MODEL_ID:
        raise ModelSourceError(
            f"Refusing model source {model_id!r}: this pipeline is pinned to "
            f"{PINNED_MODEL_ID!r} and does not load other checkpoints."
        )

    if revision is None or not isinstance(revision, str) or not revision:
        raise ModelSourceError(
            f"A revision is required and must be the pinned commit "
            f"{PINNED_REVISION}; got {revision!r}."
        )

    if revision in _MUTABLE_REFS or not _SHA1_HEX.match(revision):
        raise ModelSourceError(
            f"Refusing mutable or non-commit revision {revision!r}: the pinned "
            f"commit SHA is the primary supply-chain invariant. Use "
            f"{PINNED_REVISION}."
        )

    if revision != PINNED_REVISION:
        raise ModelSourceError(
            f"Refusing revision {revision!r}: this pipeline is pinned to "
            f"{PINNED_REVISION}. A different revision must go through an RFC "
            f"update, not a keyword argument."
        )


def resolve_hub_revision(model_id: str, revision: str) -> str:
    """Ask the Hub which commit ``revision`` names, independently of the download.

    This is the oracle the revision check needs. ``snapshot_download`` lays a
    snapshot out under ``snapshots/<requested-sha>/``, so when the request *is* a
    SHA the directory name is that SHA by construction and comparing the two
    proves nothing (review R-5). ``model_info(...).sha`` is the commit the Hub
    itself says it served, so a hub answering the pinned name with a different
    commit is detectable.

    Raises
    ------
    ModelIntegrityError
        The Hub could not be asked. Failing closed is deliberate: RFC C-7 makes
        the revision SHA the *primary* integrity anchor, so an unverifiable
        revision is not a load that should quietly proceed on the digests alone.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id=model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001 - every hub failure has the same outcome
        raise ModelIntegrityError(
            f"Could not resolve {model_id!r}@{revision} against the Hub to confirm the "
            f"pinned commit: {type(exc).__name__}: {exc}. The revision SHA is this "
            f"pipeline's primary supply-chain anchor (RFC C-7), so the load is refused "
            f"rather than falling back to the file digests alone."
        ) from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha:
        raise ModelIntegrityError(
            f"The Hub returned no commit SHA for {model_id!r}@{revision}; the resolved "
            f"revision cannot be verified against the pin."
        )
    return sha


def verify_snapshot(
    snapshot_path: str | os.PathLike[str],
    *,
    expected_revision: str = PINNED_REVISION,
    expected_config_sha256: str = PINNED_CONFIG_SHA256,
    expected_weights_sha256: str = PINNED_WEIGHTS_SHA256,
    expected_weights_bytes: int = PINNED_WEIGHTS_BYTES,
    check_resolved_revision: bool = True,
    resolved_revision: str | None = None,
) -> dict[str, Any]:
    """Prove a downloaded snapshot directory is the pinned revision, byte for byte.

    Parameters
    ----------
    resolved_revision
        The commit an *independent* resolution says was served — normally
        :func:`resolve_hub_revision`'s answer. When given, it is compared against
        ``expected_revision`` and the snapshot directory name must agree with it
        too. When omitted, the directory name is used alone, which is the weaker
        check: ``huggingface_hub`` names the directory after the *requested*
        revision, so with a SHA request that comparison is a tautology and cannot
        fail (review R-5). Production goes through :func:`load_pinned_model`,
        which always supplies it.

    Returns
    -------
    dict
        ``{"revision", "config_sha256", "weights_sha256", "weights_bytes"}``.
    """
    root = Path(snapshot_path)
    if not root.is_dir():
        raise ModelIntegrityError(f"Model snapshot path is not a directory: {root}")

    if check_resolved_revision:
        if resolved_revision is not None and resolved_revision != expected_revision:
            raise ModelIntegrityError(
                f"The Hub resolved this model to revision {resolved_revision!r}, which "
                f"does not match the pinned revision {expected_revision!r}. The pinned "
                f"commit is the primary supply-chain invariant, so a different commit is "
                f"refused whatever its contents."
            )
        if root.name != expected_revision:
            raise ModelIntegrityError(
                f"Resolved model revision {root.name!r} does not match the "
                f"pinned revision {expected_revision!r} (snapshot at {root})."
            )
    resolved_revision = resolved_revision or root.name

    refused = sorted(
        p.name
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in REFUSED_WEIGHT_SUFFIXES
    )
    if refused:
        raise ModelIntegrityError(
            f"Refusing to load: pickle-format weight files are present in the "
            f"snapshot and would permit arbitrary code execution: {refused}. "
            f"Only {WEIGHTS_FILENAME} is acceptable."
        )

    weights = root / WEIGHTS_FILENAME
    if not weights.is_file():
        raise ModelIntegrityError(f"Missing {WEIGHTS_FILENAME} in snapshot {root}.")

    actual_bytes = weights.stat().st_size
    if actual_bytes != expected_weights_bytes:
        raise ModelIntegrityError(
            f"{WEIGHTS_FILENAME} is {actual_bytes} bytes, expected "
            f"{expected_weights_bytes} for revision {expected_revision}."
        )

    weights_sha256 = sha256_file(weights)
    if weights_sha256 != expected_weights_sha256:
        raise ModelIntegrityError(
            f"{WEIGHTS_FILENAME} SHA-256 {weights_sha256} does not match the "
            f"pinned digest {expected_weights_sha256}."
        )

    config = root / CONFIG_FILENAME
    if not config.is_file():
        raise ModelIntegrityError(f"Missing {CONFIG_FILENAME} in snapshot {root}.")

    config_sha256 = sha256_file(config)
    if config_sha256 != expected_config_sha256:
        raise ModelIntegrityError(
            f"{CONFIG_FILENAME} SHA-256 {config_sha256} does not match the "
            f"pinned digest {expected_config_sha256}."
        )

    return {
        "revision": resolved_revision,
        "config_sha256": config_sha256,
        "weights_sha256": weights_sha256,
        "weights_bytes": actual_bytes,
    }


def resolve_device(device: str) -> str:
    """Turn ``"auto"`` into a concrete device name without importing torch eagerly."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_trained_quantiles(pipeline: Any) -> tuple[float, ...]:
    quantiles = getattr(pipeline, "quantiles", None)
    if quantiles is None:  # pragma: no cover - upstream contract change
        raise ModelIntegrityError(
            "Loaded pipeline does not expose a `quantiles` attribute; the "
            "trained quantile grid cannot be read from the pinned model."
        )
    return tuple(float(q) for q in quantiles)


def _describe_dtype(pipeline: Any) -> tuple[str, str]:
    """Return ``(device, dtype)`` actually in effect, read off the loaded weights."""
    model = getattr(pipeline, "model", None) or getattr(pipeline, "inner_model", None)
    if model is None:  # pragma: no cover - upstream contract change
        return ("unknown", "unknown")
    try:
        param = next(model.parameters())
    except (StopIteration, AttributeError):  # pragma: no cover
        return ("unknown", "unknown")
    return (str(param.device), str(param.dtype))


def load_pinned_model(
    model_id: str = PINNED_MODEL_ID,
    revision: str = PINNED_REVISION,
    *,
    device: str = "auto",
    dtype: str = "auto",
    cache_dir: str | os.PathLike[str] | None = None,
    revision_resolver: Callable[[str, str], str] = resolve_hub_revision,
) -> LoadedModel:
    """Download, verify and load the pinned Chronos-2 checkpoint.

    Parameters
    ----------
    model_id, revision
        Present so that a caller *can* pass something else — and be refused with
        a clear message. Both default to the pin.
    device
        ``"auto"``, ``"cpu"`` or ``"cuda"``. Recorded as resolved, not as asked.
    dtype
        Passed to ``BaseChronosPipeline.from_pretrained``; ``"auto"`` follows the
        checkpoint. The dtype actually in effect is read back off the weights.
    revision_resolver
        How the resolved commit is established, independently of the download.
        Defaults to :func:`resolve_hub_revision`, which asks the Hub. Injectable
        so the mismatch path is testable without a hostile hub; production has no
        reason to pass anything else.

    Raises
    ------
    ModelSourceError
        The source is not the pinned pair.
    ModelIntegrityError
        The resolved revision or a file digest does not match the pin.
    """
    check_model_source(model_id, revision)

    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download

    hub_revision = revision_resolver(model_id, revision)
    if hub_revision != PINNED_REVISION:
        raise ModelIntegrityError(
            f"The Hub resolved {model_id!r}@{revision} to commit {hub_revision!r}, not "
            f"the pinned {PINNED_REVISION!r}. Nothing is downloaded and nothing is loaded."
        )

    snapshot_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
    )
    verified = verify_snapshot(snapshot_path, resolved_revision=hub_revision)

    resolved_device = resolve_device(device)
    pipeline = BaseChronosPipeline.from_pretrained(
        str(snapshot_path),
        device_map=resolved_device,
        dtype=dtype,
    )

    actual_device, actual_dtype = _describe_dtype(pipeline)
    identity = ModelIdentity(
        name=model_id,
        revision=verified["revision"],
        config_sha256=verified["config_sha256"],
        weights_sha256=verified["weights_sha256"],
        weights_bytes=verified["weights_bytes"],
        license=PINNED_LICENSE,
        license_source=PINNED_LICENSE_SOURCE,
        source_url=PINNED_MODEL_URL,
        trained_quantiles=_read_trained_quantiles(pipeline),
        model_context_length=int(pipeline.model_context_length),
        model_prediction_length=int(pipeline.model_prediction_length),
        snapshot_path=str(snapshot_path),
    )
    return LoadedModel(
        pipeline=pipeline,
        identity=identity,
        device=actual_device,
        dtype=actual_dtype,
    )
