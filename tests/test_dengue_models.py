import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "dengue_models", Path(__file__).parents[1] / "tools" / "dengue_models.py"
)
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


def test_cached_snapshot_tampering_refused(tmp_path):
    payload = b"verified file"
    manifest = {
        "revision": "a" * 40,
        "modelId": "owner/model",
        "files": [
            {
                "path": "config.json",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    root = tmp_path / manifest["revision"]
    root.mkdir()
    (root / "config.json").write_bytes(payload)
    assert models.stage_snapshot(manifest, tmp_path) == root
    (root / "config.json").write_bytes(b"X" * len(payload))
    with pytest.raises(ValueError, match="integrity"):
        models.stage_snapshot(manifest, tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/file", "x\\y"])
def test_snapshot_path_refusal(tmp_path, name):
    with pytest.raises(ValueError):
        models.stage_snapshot(
            {"revision": "a" * 40, "modelId": "owner/model", "files": [{"path": name}]}, tmp_path
        )


class Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_chronos_shapes_and_crossing():
    class Pipeline:
        crossing = False

        def predict_quantiles(self, inputs, **kwargs):
            assert np.array_equal(inputs[0], [1, 2, 3])
            assert kwargs["quantile_levels"] == [0.1, 0.5, 0.9]
            assert kwargs["context_length"] == 3
            row = [1, 2, 3] if not self.crossing else [1, 3, 2]
            return [Tensor(np.array([[row] * kwargs["prediction_length"]]))], None

    pipeline = Pipeline()
    assert models.chronos_predict(pipeline, [1, 2, 3], 4).shape == (4, 3)
    pipeline.crossing = True
    with pytest.raises(ValueError, match="crossing"):
        models.chronos_predict(pipeline, [1, 2, 3], 4)


def test_mitra_context_is_fresh_and_query_excluded(monkeypatch):
    preprocessors = []

    class Preprocessor:
        def fit(self, X, y):
            self.X, self.y = X.copy(), y.copy()

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.preprocessor = Preprocessor()
            preprocessors.append(self.preprocessor)

        def post_fit_optimize(self):
            pass

        def predict(self, X, y, query):
            np.testing.assert_array_equal(X, self.preprocessor.X)
            return np.repeat(self.preprocessor.y.mean(), len(query))

    class Regressor:
        def __init__(self, **kwargs):
            assert kwargs["fine_tune"] is False
            assert kwargs["random_mirror_x"] is False

        def _create_config(self, task, output):
            return SimpleNamespace(hyperparams={}), None

    monkeypatch.setattr(models, "_seed", lambda: None)
    module = SimpleNamespace(MitraRegressor=Regressor, TrainerFinetune=Trainer)
    monkeypatch.setitem(sys.modules, "autogluon.tabular.models.mitra.sklearn_interface", module)
    network = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    support = np.array([[1, 2], [2, 3]])
    first = models.mitra_predict(network, support, [1, 3], [[9999, 9999]])
    second = models.mitra_predict(network, support, [4, 6], [[9999, 9999]])
    assert first[0] == 2 and second[0] == 5
    assert preprocessors[0] is not preprocessors[1]
    np.testing.assert_array_equal(preprocessors[0].X, support)


def test_nonfinite_inputs_refused_before_model_import():
    with pytest.raises(ValueError, match="finite"):
        models.mitra_predict(None, [[1], [2]], [1, 2], [[np.nan]])
