import keras
import numpy as np
import pytest

from fmanet.model import ConstraintWeights, CosineSimilarity, build_fmanet, compile_fmanet


@pytest.mark.parametrize(
    ("num_classes", "expected"),
    [
        (8, 5_389_832),
        (9, 5_389_897),
    ],  # model.summary() totals printed by the two original notebooks
)
def test_parameter_count_matches_original_notebooks(num_classes, expected):
    assert build_fmanet(num_classes=num_classes).count_params() == expected


def test_attention_modules_are_not_in_the_notebook_graph():
    """The notebooks create the attention modules but never connect them to the output."""
    plain = build_fmanet().count_params()
    wired = build_fmanet(use_attention_modules=True).count_params()
    assert wired > plain


def test_forward_pass_gives_a_probability_distribution():
    model = build_fmanet(input_shape=(32, 32, 3), num_classes=4)
    out = model.predict(np.random.rand(3, 32, 32, 3).astype("float32"), verbose=0)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_constraint_weights_sum_to_one():
    a = np.array([[0.2], [0.6]], dtype="float32")
    b = np.array([[0.6], [0.2]], dtype="float32")
    w1, w2 = ConstraintWeights()([a, b])
    np.testing.assert_allclose(np.asarray(w1) + np.asarray(w2), 1.0, atol=1e-6)


def test_cosine_similarity_is_tanh_of_cosine():
    a = np.array([[1.0, 0.0]], dtype="float32")
    b = np.array([[1.0, 1.0]], dtype="float32")
    expected = np.tanh(1 / np.sqrt(2))
    np.testing.assert_allclose(np.asarray(CosineSimilarity()([a, b])), [[expected]], atol=1e-5)


def test_save_and_load_round_trip(tmp_path):
    model = compile_fmanet(build_fmanet(input_shape=(32, 32, 3), num_classes=3))
    x = np.random.rand(2, 32, 32, 3).astype("float32")
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-5)
