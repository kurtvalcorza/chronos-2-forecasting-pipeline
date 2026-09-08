"""Hub *unavailability* vs Hub *disagreement* (review round 2). No network.

Round 1 made the pinned-revision check real by asking the Hub which commit the
pin resolves to. That coupled every load to Hub reachability: with
``HF_HUB_OFFLINE=1`` and a fully cached, digest-matching snapshot the load
failed with ``ModelIntegrityError``. These tests fix the distinction in place —
a Hub that cannot be reached is tolerated and recorded honestly in provenance,
a Hub that answers with anything other than the pin is refused.

Everything here is mocked: no Hub call, no weights, no torch device.
"""

from __future__ import annotations

from pathlib import Path

import chronos
import huggingface_hub
import pytest

from chronos2_pipeline import ModelIntegrityError
from chronos2_pipeline import model as model_mod
from chronos2_pipeline.errors import HubUnavailableError
from chronos2_pipeline.model import (
    CONFIG_FILENAME,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    WEIGHTS_FILENAME,
    ModelIdentity,
    sha256_file,
)

try:  # huggingface_hub >= 0.23 exposes it here; older layouts moved it around
    from huggingface_hub.errors import OfflineModeIsEnabled
except ImportError:  # pragma: no cover - depends on the installed hub version

    class OfflineModeIsEnabled(ConnectionError):  # type: ignore[no-redef]
        """Local stand-in with the same base class the real one uses."""


OTHER_SHA = "0" * 40


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _HubHTTPError(OSError):
    """Shaped like ``HfHubHTTPError``, which descends from ``OSError``.

    That inheritance is exactly the trap: a 404 must not be read as "the network
    is down" merely because the exception is an ``OSError``.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = _Response(status_code)


class _FakePipeline:
    quantiles = model_mod.EXPECTED_TRAINED_QUANTILES
    model_context_length = 8192
    model_prediction_length = 1024


def _build_snapshot(root: Path, revision: str = PINNED_REVISION) -> Path:
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / CONFIG_FILENAME).write_bytes(b'{"ok": true}')
    (snapshot / WEIGHTS_FILENAME).write_bytes(b"safetensors-stand-in")
    return snapshot


def _install_cached_snapshot(monkeypatch, tmp_path, *, model_info) -> Path:
    """Simulate "weights already cached" with ``model_info`` as the only variable.

    ``verify_snapshot`` keeps its real digest logic; only the expected digests
    are swapped for the stand-in file's own, so a later corruption of that file
    still fails for the right reason.
    """
    snapshot = _build_snapshot(tmp_path)
    expected = {
        "expected_config_sha256": sha256_file(snapshot / CONFIG_FILENAME),
        "expected_weights_sha256": sha256_file(snapshot / WEIGHTS_FILENAME),
        "expected_weights_bytes": (snapshot / WEIGHTS_FILENAME).stat().st_size,
    }
    real_verify = model_mod.verify_snapshot
    monkeypatch.setattr(
        model_mod,
        "verify_snapshot",
        lambda path, **kwargs: real_verify(path, **{**expected, **kwargs}),
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot)
    )
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(
        chronos.BaseChronosPipeline,
        "from_pretrained",
        lambda *args, **kwargs: _FakePipeline(),
    )
    return snapshot


def _offline(self, **kwargs):
    raise OfflineModeIsEnabled(
        "Offline mode is enabled. To disable it, unset HF_HUB_OFFLINE."
    )


def _serves(sha: str):
    class _Info:
        pass

    def model_info(self, **kwargs):
        info = _Info()
        info.sha = sha
        return info

    return model_info


def _forbid_download(monkeypatch):
    def explode(**kwargs):  # pragma: no cover - must not run
        raise AssertionError("snapshot_download ran despite a refused load")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)


# --------------------------------------------------------------------------
# The classifier: an answer is never an outage
# --------------------------------------------------------------------------


def test_only_transport_failures_and_serving_failures_count_as_unreachable():
    reason = model_mod._hub_unreachable_reason
    assert reason(OfflineModeIsEnabled("offline")) is not None
    assert reason(ConnectionError("no route to host")) is not None
    assert reason(TimeoutError("timed out")) is not None
    assert reason(_HubHTTPError("rate limited", 429)) is not None
    assert reason(_HubHTTPError("bad gateway", 502)) is not None
    # The Hub answered in each of these; they are supply-chain events.
    assert reason(_HubHTTPError("gated repo", 403)) is None
    assert reason(_HubHTTPError("no such revision", 404)) is None
    assert reason(_HubHTTPError("unauthorized", 401)) is None
    assert reason(ValueError("malformed reply")) is None


def test_a_hub_that_answers_404_is_a_hard_failure_not_an_outage(monkeypatch, tmp_path):
    def not_found(self, **kwargs):
        raise _HubHTTPError("404 Client Error: revision not found", 404)

    _install_cached_snapshot(monkeypatch, tmp_path, model_info=not_found)
    _forbid_download(monkeypatch)
    with pytest.raises(ModelIntegrityError) as excinfo:
        model_mod.load_pinned_model(device="cpu")
    assert not isinstance(excinfo.value, HubUnavailableError)
    assert "The Hub refused to resolve" in str(excinfo.value)


# --------------------------------------------------------------------------
# Unreachable Hub + matching digests: the load proceeds, and says so
# --------------------------------------------------------------------------


def test_offline_load_succeeds_when_every_digest_matches(monkeypatch, tmp_path):
    _install_cached_snapshot(monkeypatch, tmp_path, model_info=_offline)

    loaded = model_mod.load_pinned_model(device="cpu")

    assert loaded.identity.revision == PINNED_REVISION
    assert loaded.identity.revision_confirmed_against_hub is False
    note = loaded.identity.revision_confirmation_note
    assert "OfflineModeIsEnabled" in note
    assert WEIGHTS_FILENAME in note and CONFIG_FILENAME in note


def test_the_exported_metadata_records_that_the_hub_check_did_not_run(
    monkeypatch, tmp_path
):
    """Provenance must never imply a check that did not happen."""
    _install_cached_snapshot(monkeypatch, tmp_path, model_info=_offline)
    block = model_mod.load_pinned_model(device="cpu").identity.as_dict()
    assert block["revision_confirmed_against_hub"] is False
    assert "not consulted successfully" in block["revision_confirmation_note"]


def test_offline_load_still_refuses_a_corrupted_digest(monkeypatch, tmp_path):
    snapshot = _install_cached_snapshot(monkeypatch, tmp_path, model_info=_offline)
    (snapshot / WEIGHTS_FILENAME).write_bytes(b"tampered-safetensors")

    with pytest.raises(ModelIntegrityError) as excinfo:
        model_mod.load_pinned_model(device="cpu")
    assert "SHA-256" in str(excinfo.value)


def test_require_hub_confirmation_refuses_an_unreachable_hub(monkeypatch, tmp_path):
    _install_cached_snapshot(monkeypatch, tmp_path, model_info=_offline)
    _forbid_download(monkeypatch)

    with pytest.raises(HubUnavailableError):
        model_mod.load_pinned_model(device="cpu", require_hub_confirmation=True)

    # ... and the strict mode is opt-in, or the regression is simply relabelled.
    import inspect

    default = inspect.signature(model_mod.load_pinned_model).parameters[
        "require_hub_confirmation"
    ].default
    assert default is False


# --------------------------------------------------------------------------
# Reachable Hub: disagreement is still fatal, agreement is recorded as such
# --------------------------------------------------------------------------


def test_a_reachable_hub_serving_another_commit_is_still_refused(monkeypatch, tmp_path):
    _install_cached_snapshot(monkeypatch, tmp_path, model_info=_serves(OTHER_SHA))
    _forbid_download(monkeypatch)

    with pytest.raises(ModelIntegrityError) as excinfo:
        model_mod.load_pinned_model(device="cpu")
    assert OTHER_SHA in str(excinfo.value)
    assert not isinstance(excinfo.value, HubUnavailableError)


def test_confirmation_is_reported_true_only_when_the_hub_answered_with_the_pin(
    monkeypatch, tmp_path
):
    _install_cached_snapshot(
        monkeypatch, tmp_path, model_info=_serves(PINNED_REVISION)
    )

    identity = model_mod.load_pinned_model(device="cpu").identity

    assert identity.revision_confirmed_against_hub is True
    assert PINNED_REVISION in identity.revision_confirmation_note
    assert PINNED_MODEL_ID in identity.revision_confirmation_note


def test_a_hand_built_identity_claims_no_confirmation_by_default():
    """The dataclass default must be the honest one, not the flattering one."""
    identity = ModelIdentity(
        name=PINNED_MODEL_ID,
        revision=PINNED_REVISION,
        config_sha256="c" * 64,
        weights_sha256="w" * 64,
        weights_bytes=1,
        license="apache-2.0",
        license_source="metadata",
        source_url="https://example.invalid",
        trained_quantiles=(0.5,),
        model_context_length=8192,
        model_prediction_length=1024,
    )
    assert identity.revision_confirmed_against_hub is False
    assert identity.as_dict()["revision_confirmed_against_hub"] is False
