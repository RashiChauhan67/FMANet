import json

import numpy as np
import pytest

from fmanet.model import build_fmanet, compile_fmanet
from fmanet.training import TrainConfig, evaluate_model, train_and_evaluate

SIZE = (32, 32, 3)
CLASSES = ("c0", "c1", "c2")


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = rng.random((48, *SIZE), dtype=np.float32)
    labels = np.repeat(np.arange(3), 16)
    y = np.eye(3, dtype=np.float32)[labels]
    return x[:36], y[:36], x[36:], y[36:]


def _model():
    return compile_fmanet(build_fmanet(input_shape=SIZE, num_classes=3))


def test_evaluate_model_returns_expected_fields(toy_data):
    _, _, x_test, y_test = toy_data
    result = evaluate_model(_model(), x_test, y_test, CLASSES, batch_size=4)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["num_samples"] == 12
    assert np.array(result["confusion_matrix"]).sum() == 12
    assert set(CLASSES) <= set(result["classification_report"])


@pytest.mark.parametrize("protocol", ["notebook", "holdout"])
def test_train_and_evaluate_runs_and_writes_outputs(toy_data, tmp_path, protocol):
    x_train, y_train, x_test, y_test = toy_data
    cfg = TrainConfig(epochs=1, batch_size=8, protocol=protocol, val_fraction=0.25)

    result = train_and_evaluate(_model(), x_train, y_train, x_test, y_test, CLASSES, cfg, tmp_path)

    assert result["protocol"] == protocol and result["epochs_run"] == 1
    assert (tmp_path / "history.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["protocol"] == protocol


def test_unknown_protocol_raises(toy_data):
    x_train, y_train, x_test, y_test = toy_data
    with pytest.raises(ValueError):
        train_and_evaluate(
            _model(), x_train, y_train, x_test, y_test, CLASSES, TrainConfig(protocol="bogus")
        )
