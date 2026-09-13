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

Check 2 asks the Hub which commit the pinned name resolves to, which means it
needs the network. Availability and integrity are kept apart there (review round
2): a Hub that *answers* with a different commit is a supply-chain failure and
always fails the load, while a Hub that cannot be *reached* at all is an
availability failure — the load proceeds on the digests, which prove the
snapshot's content byte for byte on their own, and the exported provenance
records ``revision_confirmed_against_hub: false`` with the reason. Pass
``require_hub_confirmation=True`` to refuse that degraded mode; it is not the
default, so a cached, digest-matching snapshot still loads offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import HubUnavailableError, ModelIntegrityError, ModelSourceError

__all__ = [
    "PINNED_MODEL_ID",
    "PINNED_REVISION",
    "PINNED_WEIGHTS_SHA256",
    "PINNED_WEIGHTS_BYTES",
    "PINNED_CONFIG_SHA256",
    "PINNED_LICENSE",
    "PINNED_LICENSE_SOURCE",
    "PINNED_MODEL_URL",
    "PINNED_MODEL_KEY",
    "DEFAULT_WEIGHTS_DIR",
    "MANIFEST_NAME",
    "EXPECTED_TRAINED_QUANTILES",
    "WEIGHTS_FILENAME",
    "CONFIG_FILENAME",
    "REFUSED_WEIGHT_SUFFIXES",
    "ModelIdentity",
    "LoadedModel",
    "sha256_file",
    "check_model_source",
    "verify_snapshot",
    "stage_missing_files",
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

#: Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): the pinned files also live in a
#: repository-local snapshot directory named by this key, described by a committed
#: ``dimer-base-manifest.json``. The manifest is the parity anchor a standalone notebook
#: carries inline; the digest constants above are asserted equal to it on every load, so the
#: two can never disagree silently.
PINNED_MODEL_KEY = "chronos-2"
MANIFEST_NAME = "dimer-base-manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / PINNED_MODEL_KEY

#: Pickle-format checkpoints execute arbitrary code on load. If any appear in
#: the snapshot the load is refused rather than silently preferring safetensors.
REFUSED_WEIGHT_SUFFIXES: tuple[str, ...] = (".bin", ".pt", ".pth", ".ckpt", ".pkl")

_MUTABLE_REFS = frozenset({"main", "master", "latest", "head", "HEAD"})
_URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:[\\/]")
_SHA1_HEX = re.compile(r"^[0-9a-f]{40}$")

#: Exception type names that mean the request never got an answer out of the
#: Hub: no DNS, no route, no socket, a proxy or TLS failure, or a cache that has
#: been told not to go to the network at all (``HF_HUB_OFFLINE=1``).
_UNREACHABLE_EXC_NAMES = frozenset(
    {
        "OfflineModeIsEnabled",
        "LocalEntryNotFoundError",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectTimeoutError",
        "Timeout",
        "TimeoutError",
        "ProxyError",
        "SSLError",
        "ChunkedEncodingError",
        "socket.timeout",
    }
)

#: HTTP statuses that are the Hub failing to serve rather than the Hub
#: disagreeing. 401/403/404 are excluded on purpose: they are answers — the
#: repo is gated, or the revision is not there — and must fail closed.
_UNREACHABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _hub_unreachable_reason(exc: BaseException) -> str | None:
    """Say why the Hub was unreachable, or ``None`` if it actually answered.

    The distinction is the whole point of review round 2. ``None`` means the Hub
    (or its cache of an authoritative answer) responded and the response was not
    the pin — gated repo, missing repo, missing revision, malformed reply — which
    is a supply-chain event and must never be downgraded to "offline".
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        # An HTTP status means something answered, so only the serving failures
        # count as unreachable.
        if status in _UNREACHABLE_HTTP_STATUS:
            return f"the Hub responded HTTP {status} rather than serving the lookup"
        return None

    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & _UNREACHABLE_EXC_NAMES:
        return f"{type(exc).__name__}: {exc}"
    # requests' transport errors all descend from OSError; anything left that is
    # an OSError is a socket/filesystem failure, not a Hub answer.
    if isinstance(exc, OSError):
        return f"{type(exc).__name__}: {exc}"
    return None


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
    #: True only when a Hub lookup ran on *this* load and answered with the
    #: pinned commit. False whenever the Hub was not reached, so the exported
    #: metadata can never imply a check that did not happen. The default is the
    #: honest one: an identity built by hand has confirmed nothing.
    revision_confirmed_against_hub: bool = False
    #: How the revision was established on this load — the successful lookup, or
    #: the reason it could not run and what carried the integrity claim instead.
    revision_confirmation_note: str = "no Hub confirmation was attempted"
    snapshot_path: str = field(default="", compare=False)

    @property
    def model_id(self) -> str:
        """Alias for name for compatibility with downstream loaders."""
        return self.name

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
            "revision_confirmed_against_hub": self.revision_confirmed_against_hub,
            "revision_confirmation_note": self.revision_confirmation_note,
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
    HubUnavailableError
        The Hub was never reached — offline mode, no route, a proxy or TLS
        failure, a rate limit, a 5xx. Nothing is claimed about the commit; the
        caller decides whether the recorded digests are enough (they are, by
        default: see :func:`load_pinned_model`).
    ModelIntegrityError
        The Hub answered and its answer was not usable as a confirmation — a
        gated or missing repo, a revision it does not have, or a reply carrying
        no commit SHA. This always fails closed.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id=model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001 - classified, then re-raised
        unreachable = _hub_unreachable_reason(exc)
        if unreachable is not None:
            raise HubUnavailableError(
                f"Could not reach the Hugging Face Hub to confirm {model_id!r}@{revision}: "
                f"{unreachable}. No claim is made about the resolved commit."
            ) from exc
        raise ModelIntegrityError(
            f"The Hub refused to resolve {model_id!r}@{revision}: "
            f"{type(exc).__name__}: {exc}. The Hub answered and its answer was not the "
            f"pinned commit, so the load is refused."
        ) from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha:
        raise ModelIntegrityError(
            f"The Hub returned no commit SHA for {model_id!r}@{revision}; the resolved "
            f"revision cannot be verified against the pin."
        )
    return sha


def _read_manifest(root: Path, *, expected_revision: str) -> dict[str, Any]:
    """Load and identity-check ``<root>/dimer-base-manifest.json``."""
    manifest_path = root / MANIFEST_NAME
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelIntegrityError(f"Could not read snapshot manifest {manifest_path}: {exc}") from exc
    if manifest.get("modelId") != PINNED_MODEL_ID:
        raise ModelIntegrityError(
            f"Snapshot manifest names model {manifest.get('modelId')!r}, not the pinned "
            f"{PINNED_MODEL_ID!r} ({manifest_path})."
        )
    if manifest.get("revision") != expected_revision:
        raise ModelIntegrityError(
            f"Snapshot manifest names revision {manifest.get('revision')!r}, not the pinned "
            f"{expected_revision!r} ({manifest_path})."
        )
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ModelIntegrityError(f"Snapshot manifest lists no files: {manifest_path}")
    return manifest


def _verify_manifest_snapshot(
    root: Path,
    *,
    expected_revision: str,
    expected_config_sha256: str,
    expected_weights_sha256: str,
    expected_weights_bytes: int,
    resolved_revision: str | None,
) -> dict[str, Any]:
    """Manifest-driven verification of a snapshot directory named by ``PINNED_MODEL_KEY``.

    The directory name carries no revision here — the committed manifest does — so the
    revision assertion moves to the manifest and the per-file digests come from it. The
    package's own ``PINNED_*`` digest constants are then asserted **equal to** the manifest
    entries, so this path is never weaker than the revision-directory path.
    """
    if resolved_revision is not None and resolved_revision != expected_revision:
        raise ModelIntegrityError(
            f"The Hub resolved this model to revision {resolved_revision!r}, which does not "
            f"match the pinned revision {expected_revision!r}. The pinned commit is the primary "
            f"supply-chain invariant, so a different commit is refused whatever its contents."
        )
    manifest = _read_manifest(root, expected_revision=expected_revision)

    refused = sorted(
        p.name
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in REFUSED_WEIGHT_SUFFIXES
    )
    if refused:
        raise ModelIntegrityError(
            f"Refusing to load: pickle-format weight files are present in the snapshot and "
            f"would permit arbitrary code execution: {refused}. Only {WEIGHTS_FILENAME} is "
            f"acceptable."
        )

    digests: dict[str, str] = {}
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise ModelIntegrityError(f"Snapshot file listed in the manifest is missing: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != entry["bytes"]:
            raise ModelIntegrityError(
                f"{entry['path']} is {actual_bytes} bytes, expected {entry['bytes']} for "
                f"revision {expected_revision}."
            )
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ModelIntegrityError(
                f"{entry['path']} SHA-256 {digest} does not match the manifest digest "
                f"{entry['sha256']}."
            )
        digests[entry["path"]] = digest

    sizes = {entry["path"]: int(entry["bytes"]) for entry in manifest["files"]}
    for filename in (CONFIG_FILENAME, WEIGHTS_FILENAME):
        if filename not in digests:
            raise ModelIntegrityError(
                f"Snapshot manifest does not list {filename}, which the loader requires "
                f"({root / MANIFEST_NAME})."
            )
    if digests[CONFIG_FILENAME] != expected_config_sha256:
        raise ModelIntegrityError(
            f"Manifest {CONFIG_FILENAME} digest {digests[CONFIG_FILENAME]} does not match the "
            f"pinned digest {expected_config_sha256}."
        )
    if digests[WEIGHTS_FILENAME] != expected_weights_sha256:
        raise ModelIntegrityError(
            f"Manifest {WEIGHTS_FILENAME} digest {digests[WEIGHTS_FILENAME]} does not match the "
            f"pinned digest {expected_weights_sha256}."
        )
    if sizes[WEIGHTS_FILENAME] != expected_weights_bytes:
        raise ModelIntegrityError(
            f"Manifest {WEIGHTS_FILENAME} size {sizes[WEIGHTS_FILENAME]} does not match the "
            f"pinned byte count {expected_weights_bytes}."
        )

    return {
        "path": str(root),
        "revision": str(manifest["revision"]),
        "config_sha256": digests[CONFIG_FILENAME],
        "weights_sha256": digests[WEIGHTS_FILENAME],
        "weights_bytes": sizes[WEIGHTS_FILENAME],
        "files": list(manifest["files"]),
        "model_key": manifest.get("modelKey", PINNED_MODEL_KEY),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at ``PINNED_REVISION`` straight into ``root``."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(PINNED_MODEL_ID, relative_path, revision=PINNED_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | os.PathLike[str] | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent from the snapshot directory.

    A fresh clone commits the manifest and git-ignores the weights, so this is how the
    snapshot is populated. Only files named by the manifest are fetched, only at
    ``PINNED_REVISION``, and :func:`verify_snapshot` still re-hashes everything afterwards.
    Returns the relative paths fetched (empty when nothing was missing).
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root, expected_revision=PINNED_REVISION)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them "
            f"at {PINNED_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


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

    if (root / MANIFEST_NAME).is_file():
        # Fleet snapshot directory (``weights/<PINNED_MODEL_KEY>/``): the revision is carried by
        # the committed manifest rather than by the directory name, and the per-file digests come
        # from it. The pinned constants are asserted equal to the manifest inside.
        return _verify_manifest_snapshot(
            root,
            expected_revision=expected_revision,
            expected_config_sha256=expected_config_sha256,
            expected_weights_sha256=expected_weights_sha256,
            expected_weights_bytes=expected_weights_bytes,
            resolved_revision=resolved_revision if check_resolved_revision else None,
        )

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
    require_hub_confirmation: bool = False,
    weights_dir: str | os.PathLike[str] | None = None,
    allow_download: bool = False,
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
    require_hub_confirmation
        When ``True``, a Hub that cannot be reached is fatal. The default
        ``False`` is the available mode: an unreachable Hub lets the load
        continue on the recorded digests alone, and the resulting
        :class:`ModelIdentity` reports ``revision_confirmed_against_hub=False``
        with the reason. A Hub that *answers* with a different commit is refused
        in either mode.
    weights_dir
        A fleet snapshot directory holding ``dimer-base-manifest.json`` (normally
        ``weights/<PINNED_MODEL_KEY>/``). When given, nothing is resolved through
        ``snapshot_download``: missing manifest entries are staged with
        :func:`stage_missing_files`, :func:`verify_snapshot` re-hashes every entry against the
        manifest *and* against the pinned digest constants, and the same
        ``BaseChronosPipeline.from_pretrained`` call loads from that directory. Identical files,
        identical loader, a different place to keep them.
    allow_download
        Only meaningful with ``weights_dir``: permit :func:`stage_missing_files` to fetch the
        manifest entries that are absent, at ``PINNED_REVISION``. The default refuses.

    Raises
    ------
    ModelSourceError
        The source is not the pinned pair.
    HubUnavailableError
        ``require_hub_confirmation=True`` and the Hub could not be reached.
    ModelIntegrityError
        The Hub resolved the pin to another commit, or a file digest or the
        weight byte count does not match the pin.
    """
    check_model_source(model_id, revision)

    from chronos import BaseChronosPipeline

    hub_revision: str | None
    snapshot_path: str
    if weights_dir is not None:
        # Fleet snapshot path: the committed manifest is the revision oracle and the digests
        # prove the content, so no Hub lookup is performed for the identity.
        root = Path(weights_dir)
        stage_missing_files(root, allow_download=allow_download)
        verified = verify_snapshot(root)
        hub_revision = None
        confirmation_note = (
            f"the Hub was not consulted on this load; the snapshot's identity rests on the "
            f"committed {MANIFEST_NAME} at {root}, whose per-file SHA-256 digests were "
            f"re-hashed and asserted equal to the pinned {CONFIG_FILENAME} and "
            f"{WEIGHTS_FILENAME} digests and the weight byte count"
        )
        snapshot_path = str(root)
    else:
        from huggingface_hub import snapshot_download

        try:
            hub_revision = revision_resolver(model_id, revision)
        except HubUnavailableError as exc:
            if require_hub_confirmation:
                raise
            hub_revision = None
            confirmation_note = (
                f"the Hub was not consulted successfully on this load ({exc}); the "
                f"snapshot's identity rests on the recorded {CONFIG_FILENAME} and "
                f"{WEIGHTS_FILENAME} SHA-256 digests and the weight byte count, which "
                f"prove its content independently of the Hub"
            )
        else:
            if hub_revision != PINNED_REVISION:
                raise ModelIntegrityError(
                    f"The Hub resolved {model_id!r}@{revision} to commit {hub_revision!r}, not "
                    f"the pinned {PINNED_REVISION!r}. Nothing is downloaded and nothing is loaded."
                )
            confirmation_note = (
                f"the Hub resolved {model_id}@{revision} to {hub_revision} on this load"
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
        revision_confirmed_against_hub=hub_revision is not None,
        revision_confirmation_note=confirmation_note,
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
