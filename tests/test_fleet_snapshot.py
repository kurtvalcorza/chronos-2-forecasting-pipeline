"""Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): manifest-driven staging and verification.

Nothing here touches the network or the real weights: the snapshot directory is built from
stand-in bytes, the manifest is written by hand, and downloads go through an injected callable.
"""

# ruff: noqa: E501  -- offline fixtures and assertions are kept on single lines for readability
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronos2_pipeline import model as model_mod
from chronos2_pipeline.model import (
    CONFIG_FILENAME,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PINNED_LICENSE,
    PINNED_MODEL_ID,
    PINNED_MODEL_KEY,
    PINNED_REVISION,
    WEIGHTS_FILENAME,
    ModelIntegrityError,
    sha256_file,
    stage_missing_files,
    verify_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
OTHER_SHA = "0" * 40


def write_manifest(root: Path, files: dict[str, bytes], **overrides) -> dict:
    entries = []
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append({"path": name, "bytes": len(payload), "sha256": sha256_file(path)})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
        **overrides,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def stand_in_snapshot(root: Path) -> dict:
    return write_manifest(
        root,
        {"README.md": b"# stand-in\n", CONFIG_FILENAME: b'{"ok": true}', WEIGHTS_FILENAME: b"safetensors-stand-in"},
    )


def pinned_expectations(snapshot: Path) -> dict:
    return {
        "expected_config_sha256": sha256_file(snapshot / CONFIG_FILENAME),
        "expected_weights_sha256": sha256_file(snapshot / WEIGHTS_FILENAME),
        "expected_weights_bytes": (snapshot / WEIGHTS_FILENAME).stat().st_size,
    }


# --------------------------------------------------------------------------
# Identity constants
# --------------------------------------------------------------------------


def test_fleet_identity_names_alias_the_package_pins():
    assert (MODEL_ID, MODEL_REVISION, MODEL_LICENSE, MODEL_KEY) == (
        PINNED_MODEL_ID,
        PINNED_REVISION,
        PINNED_LICENSE,
        PINNED_MODEL_KEY,
    )
    assert model_mod.DEFAULT_WEIGHTS_DIR == ROOT / "weights" / MODEL_KEY


def test_committed_manifest_names_the_pinned_identity_and_digests():
    manifest = json.loads((ROOT / "weights" / MODEL_KEY / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    assert by_path[WEIGHTS_FILENAME]["sha256"] == model_mod.PINNED_WEIGHTS_SHA256
    assert by_path[WEIGHTS_FILENAME]["bytes"] == model_mod.PINNED_WEIGHTS_BYTES
    assert by_path[CONFIG_FILENAME]["sha256"] == model_mod.PINNED_CONFIG_SHA256
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])


# --------------------------------------------------------------------------
# stage_missing_files
# --------------------------------------------------------------------------


def test_stage_fetches_only_the_absent_entries_through_the_injected_downloader(tmp_path):
    stand_in_snapshot(tmp_path)
    (tmp_path / WEIGHTS_FILENAME).unlink()
    (tmp_path / "README.md").unlink()
    fetched_calls: list[tuple[str, Path]] = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched_calls.append((relative_path, root))
        (root / relative_path).write_bytes(b"safetensors-stand-in" if relative_path == WEIGHTS_FILENAME else b"# stand-in\n")

    fetched = stage_missing_files(tmp_path, allow_download=True, downloader=downloader)
    assert sorted(fetched) == sorted(["README.md", WEIGHTS_FILENAME])
    assert [call[0] for call in fetched_calls] == fetched
    assert all(call[1] == tmp_path for call in fetched_calls)
    assert stage_missing_files(tmp_path, allow_download=False, downloader=downloader) == []
    assert len(fetched_calls) == 2


def test_stage_refuses_to_download_by_default(tmp_path):
    stand_in_snapshot(tmp_path)
    (tmp_path / WEIGHTS_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)


def test_stage_refuses_a_manifest_with_the_wrong_identity(tmp_path):
    write_manifest(tmp_path, {CONFIG_FILENAME: b"{}", WEIGHTS_FILENAME: b"w"}, modelId="someone/else")
    with pytest.raises(ModelIntegrityError, match="not the pinned"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)
    write_manifest(tmp_path, {CONFIG_FILENAME: b"{}", WEIGHTS_FILENAME: b"w"}, revision=OTHER_SHA)
    with pytest.raises(ModelIntegrityError, match="not the pinned"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_stage_refuses_a_directory_without_a_manifest(tmp_path):
    with pytest.raises(ModelIntegrityError, match="Could not read snapshot manifest"):
        stage_missing_files(tmp_path)


def test_the_default_downloader_fetches_at_the_pinned_revision(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_hf_hub_download(repo_id, filename, *, revision, local_dir):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision, "local_dir": local_dir})

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    model_mod._hub_download(WEIGHTS_FILENAME, tmp_path)
    assert calls == [
        {"repo_id": MODEL_ID, "filename": WEIGHTS_FILENAME, "revision": MODEL_REVISION, "local_dir": str(tmp_path)}
    ]


# --------------------------------------------------------------------------
# verify_snapshot on a manifest-described directory
# --------------------------------------------------------------------------


def test_a_manifest_snapshot_verifies_and_reports_the_manifest(tmp_path):
    manifest = stand_in_snapshot(tmp_path)
    result = verify_snapshot(tmp_path, **pinned_expectations(tmp_path))
    assert result["revision"] == MODEL_REVISION
    assert result["files"] == manifest["files"]
    assert result["model_key"] == MODEL_KEY
    assert result["weights_bytes"] == len(b"safetensors-stand-in")


def test_a_tampered_manifest_digest_is_refused(tmp_path):
    manifest = stand_in_snapshot(tmp_path)
    expected = pinned_expectations(tmp_path)
    for entry in manifest["files"]:
        if entry["path"] == WEIGHTS_FILENAME:
            entry["sha256"] = "f" * 64
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="does not match the manifest digest"):
        verify_snapshot(tmp_path, **expected)


def test_a_tampered_file_is_refused_against_the_manifest(tmp_path):
    stand_in_snapshot(tmp_path)
    expected = pinned_expectations(tmp_path)
    (tmp_path / WEIGHTS_FILENAME).write_bytes(b"Safetensors-stand-in")  # same size, one byte flipped
    with pytest.raises(ModelIntegrityError, match="does not match the manifest digest"):
        verify_snapshot(tmp_path, **expected)


def test_a_manifest_that_disagrees_with_the_pinned_digest_constants_is_refused(tmp_path):
    """The package's own PINNED_* digests are asserted equal to the manifest, never replaced."""
    stand_in_snapshot(tmp_path)
    expected = pinned_expectations(tmp_path)
    expected["expected_weights_sha256"] = "e" * 64
    with pytest.raises(ModelIntegrityError, match="does not match the pinned digest"):
        verify_snapshot(tmp_path, **expected)


def test_a_manifest_snapshot_with_the_wrong_identity_is_refused(tmp_path):
    write_manifest(tmp_path, {CONFIG_FILENAME: b"{}", WEIGHTS_FILENAME: b"w"}, revision=OTHER_SHA)
    with pytest.raises(ModelIntegrityError, match="not the pinned"):
        verify_snapshot(tmp_path, **pinned_expectations(tmp_path))


def test_a_missing_manifest_entry_is_refused(tmp_path):
    stand_in_snapshot(tmp_path)
    expected = pinned_expectations(tmp_path)
    (tmp_path / "README.md").unlink()
    with pytest.raises(ModelIntegrityError, match="listed in the manifest is missing"):
        verify_snapshot(tmp_path, **expected)


def test_pickle_weights_next_to_a_manifest_are_refused(tmp_path):
    stand_in_snapshot(tmp_path)
    expected = pinned_expectations(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ModelIntegrityError, match="pickle-format"):
        verify_snapshot(tmp_path, **expected)


# --------------------------------------------------------------------------
# load_pinned_model(weights_dir=...) stages, verifies, then loads from that directory
# --------------------------------------------------------------------------


class _FakePipeline:
    quantiles = list(model_mod.EXPECTED_TRAINED_QUANTILES)
    model_context_length = 8192
    model_prediction_length = 1024
    model = None


def test_load_pinned_model_from_weights_dir_never_consults_the_hub(monkeypatch, tmp_path):
    import sys
    import types

    stand_in_snapshot(tmp_path)
    expected = pinned_expectations(tmp_path)
    monkeypatch.setattr(model_mod, "PINNED_CONFIG_SHA256", expected["expected_config_sha256"])
    monkeypatch.setattr(model_mod, "PINNED_WEIGHTS_SHA256", expected["expected_weights_sha256"])
    monkeypatch.setattr(model_mod, "PINNED_WEIGHTS_BYTES", expected["expected_weights_bytes"])
    loaded_from: list[str] = []

    class FakeBase:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded_from.append(path)
            return _FakePipeline()

    monkeypatch.setitem(sys.modules, "chronos", types.SimpleNamespace(BaseChronosPipeline=FakeBase))

    def explode(*args, **kwargs):
        raise AssertionError("the Hub must not be consulted on the weights_dir path")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)
    monkeypatch.setattr(model_mod, "resolve_hub_revision", explode)

    def verify_with_test_digests(path, **kwargs):
        return original_verify(path, **{**expected, **kwargs})

    original_verify = model_mod.verify_snapshot
    monkeypatch.setattr(model_mod, "verify_snapshot", verify_with_test_digests)
    loaded = model_mod.load_pinned_model(device="cpu", weights_dir=tmp_path)
    assert loaded_from == [str(tmp_path)]
    assert loaded.identity.revision == MODEL_REVISION
    assert loaded.identity.revision_confirmed_against_hub is False
    assert MANIFEST_NAME in loaded.identity.revision_confirmation_note
    assert loaded.identity.snapshot_path == str(tmp_path)


def test_load_pinned_model_from_weights_dir_refuses_to_download_by_default(monkeypatch, tmp_path):
    import sys
    import types

    stand_in_snapshot(tmp_path)
    (tmp_path / WEIGHTS_FILENAME).unlink()
    monkeypatch.setitem(sys.modules, "chronos", types.SimpleNamespace(BaseChronosPipeline=object))
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        model_mod.load_pinned_model(device="cpu", weights_dir=tmp_path)
