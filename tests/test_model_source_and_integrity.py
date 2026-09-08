"""Loader source-rejection matrix and snapshot integrity checks. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronos2_pipeline import ModelIntegrityError, ModelSourceError
from chronos2_pipeline import model as model_mod
from chronos2_pipeline.model import (
    CONFIG_FILENAME,
    PINNED_CONFIG_SHA256,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    PINNED_WEIGHTS_BYTES,
    PINNED_WEIGHTS_SHA256,
    WEIGHTS_FILENAME,
    check_model_source,
    sha256_file,
    verify_snapshot,
)

OTHER_SHA = "0" * 40


# --------------------------------------------------------------------------
# The pins themselves
# --------------------------------------------------------------------------


def test_pinned_constants_are_the_values_recorded_in_the_rfc():
    assert PINNED_MODEL_ID == "amazon/chronos-2"
    assert PINNED_REVISION == "95a9710e2596287d08352589f42634fa5abdf0a7"
    assert PINNED_WEIGHTS_SHA256 == (
        "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
    )
    assert PINNED_WEIGHTS_BYTES == 477_930_472
    assert PINNED_CONFIG_SHA256 == (
        "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031"
    )


def test_expected_trained_quantiles_is_the_21_level_grid():
    grid = model_mod.EXPECTED_TRAINED_QUANTILES
    assert len(grid) == 21
    assert grid[0] == 0.01
    assert grid[-1] == 0.99
    assert list(grid) == sorted(grid)
    assert 0.5 in grid


def test_license_is_recorded_as_metadata_not_as_a_file():
    assert model_mod.PINNED_LICENSE == "apache-2.0"
    source = model_mod.PINNED_LICENSE_SOURCE
    assert "model card metadata" in source
    assert "no LICENSE file" in source
    assert PINNED_REVISION in source


# --------------------------------------------------------------------------
# Source rejection matrix
# --------------------------------------------------------------------------


def test_the_pinned_pair_is_accepted():
    check_model_source(PINNED_MODEL_ID, PINNED_REVISION)  # must not raise


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD"])
def test_mutable_refs_rejected(revision):
    with pytest.raises(ModelSourceError, match="mutable"):
        check_model_source(PINNED_MODEL_ID, revision)


def test_a_different_commit_sha_is_rejected():
    with pytest.raises(ModelSourceError) as exc:
        check_model_source(PINNED_MODEL_ID, OTHER_SHA)
    assert PINNED_REVISION in str(exc.value)


@pytest.mark.parametrize(
    "source",
    [
        "s3://my-bucket/chronos-2",
        "hf://amazon/chronos-2",
        "https://huggingface.co/amazon/chronos-2",
        "file:///models/chronos-2",
    ],
)
def test_uri_schemes_rejected(source):
    with pytest.raises(ModelSourceError, match="URI schemes"):
        check_model_source(source, PINNED_REVISION)


@pytest.mark.parametrize(
    "source",
    ["/models/chronos-2", "./chronos-2", "../chronos-2", r"C:\models\chronos-2", r"models\chronos"],
)
def test_local_paths_rejected(source):
    with pytest.raises(ModelSourceError, match="local filesystem paths"):
        check_model_source(source, PINNED_REVISION)


def test_an_existing_local_directory_named_like_a_repo_id_is_still_a_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "amazon").mkdir()
    (tmp_path / "amazon" / "chronos-3").mkdir()
    with pytest.raises(ModelSourceError, match="local filesystem paths"):
        check_model_source("amazon/chronos-3", PINNED_REVISION)


@pytest.mark.parametrize("source", ["amazon/chronos-bolt-base", "google/timesfm-1.0", "chronos-2"])
def test_other_repo_ids_rejected(source):
    with pytest.raises(ModelSourceError, match="pinned to"):
        check_model_source(source, PINNED_REVISION)


@pytest.mark.parametrize("revision", [None, "", "v2.0", 12345])
def test_non_commit_revisions_rejected(revision):
    with pytest.raises(ModelSourceError):
        check_model_source(PINNED_MODEL_ID, revision)


def test_model_source_error_is_a_value_error():
    with pytest.raises(ValueError):
        check_model_source(PINNED_MODEL_ID, "main")


def test_load_pinned_model_refuses_before_touching_the_network(monkeypatch):
    """A rejected source must never reach snapshot_download."""
    import huggingface_hub

    def explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("snapshot_download was called for a rejected source")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)
    with pytest.raises(ModelSourceError):
        model_mod.load_pinned_model(PINNED_MODEL_ID, "main")


# --------------------------------------------------------------------------
# Snapshot integrity
# --------------------------------------------------------------------------


def build_snapshot(
    root: Path,
    *,
    revision: str = PINNED_REVISION,
    config_bytes: bytes = b'{"ok": true}',
    weights_bytes: bytes = b"safetensors-stand-in",
) -> Path:
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / CONFIG_FILENAME).write_bytes(config_bytes)
    (snapshot / WEIGHTS_FILENAME).write_bytes(weights_bytes)
    return snapshot


def expectations(snapshot: Path) -> dict:
    return {
        "expected_config_sha256": sha256_file(snapshot / CONFIG_FILENAME),
        "expected_weights_sha256": sha256_file(snapshot / WEIGHTS_FILENAME),
        "expected_weights_bytes": (snapshot / WEIGHTS_FILENAME).stat().st_size,
    }


def test_a_matching_snapshot_verifies(tmp_path):
    snapshot = build_snapshot(tmp_path)
    result = verify_snapshot(snapshot, **expectations(snapshot))
    assert result["revision"] == PINNED_REVISION
    assert result["weights_bytes"] == len(b"safetensors-stand-in")


def test_a_snapshot_resolved_to_another_commit_is_refused(tmp_path):
    snapshot = build_snapshot(tmp_path, revision=OTHER_SHA)
    with pytest.raises(ModelIntegrityError, match="does not match the pinned revision"):
        verify_snapshot(snapshot, **expectations(snapshot))


def test_a_corrupted_config_is_refused(tmp_path):
    snapshot = build_snapshot(tmp_path)
    expected = expectations(snapshot)
    (snapshot / CONFIG_FILENAME).write_bytes(b'{"ok": false}')
    with pytest.raises(ModelIntegrityError) as exc:
        verify_snapshot(snapshot, **expected)
    assert CONFIG_FILENAME in str(exc.value)
    assert expected["expected_config_sha256"] in str(exc.value)


def test_a_corrupted_weight_file_of_the_same_size_is_refused(tmp_path):
    snapshot = build_snapshot(tmp_path)
    expected = expectations(snapshot)
    original = (snapshot / WEIGHTS_FILENAME).read_bytes()
    tampered = b"S" + original[1:]
    assert len(tampered) == len(original)
    (snapshot / WEIGHTS_FILENAME).write_bytes(tampered)
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        verify_snapshot(snapshot, **expected)


def test_a_truncated_weight_file_is_refused_on_size_first(tmp_path):
    snapshot = build_snapshot(tmp_path)
    expected = expectations(snapshot)
    (snapshot / WEIGHTS_FILENAME).write_bytes(b"short")
    with pytest.raises(ModelIntegrityError, match="bytes, expected"):
        verify_snapshot(snapshot, **expected)


@pytest.mark.parametrize("name", ["pytorch_model.bin", "weights.pt", "model.ckpt", "state.pkl"])
def test_pickle_format_weights_in_the_snapshot_are_refused(tmp_path, name):
    snapshot = build_snapshot(tmp_path)
    expected = expectations(snapshot)
    (snapshot / name).write_bytes(b"pickled")
    with pytest.raises(ModelIntegrityError, match="pickle-format"):
        verify_snapshot(snapshot, **expected)


def test_a_missing_weight_file_is_refused(tmp_path):
    snapshot = build_snapshot(tmp_path)
    expected = expectations(snapshot)
    (snapshot / WEIGHTS_FILENAME).unlink()
    with pytest.raises(ModelIntegrityError, match=f"Missing {WEIGHTS_FILENAME}"):
        verify_snapshot(snapshot, **expected)


def test_a_missing_config_file_is_refused(tmp_path):
    snapshot = build_snapshot(tmp_path)
    expected = expectations(snapshot)
    (snapshot / CONFIG_FILENAME).unlink()
    with pytest.raises(ModelIntegrityError, match=f"Missing {CONFIG_FILENAME}"):
        verify_snapshot(snapshot, **expected)


def test_a_nonexistent_snapshot_path_is_refused(tmp_path):
    with pytest.raises(ModelIntegrityError, match="not a directory"):
        verify_snapshot(tmp_path / "nope")


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    payload = json.dumps({"a": 1}).encode()
    path = tmp_path / "f.json"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_resolve_device_is_cpu_when_cuda_is_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert model_mod.resolve_device("auto") == "cpu"
    assert model_mod.resolve_device("cpu") == "cpu"


def test_resolve_device_is_cuda_when_cuda_is_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert model_mod.resolve_device("auto") == "cuda"
