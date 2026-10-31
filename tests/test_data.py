import cv2
import numpy as np
import pytest

from fmanet.data import DATASETS, holdout_split, load_image_folder, split_dataset


@pytest.fixture
def image_root(tmp_path):
    """Three tiny class folders with 10 random images each (plus a non-image file)."""
    rng = np.random.default_rng(0)
    for name in ("a", "b", "c"):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(10):
            cv2.imwrite(str(folder / f"{i}.png"), rng.integers(0, 255, (20, 24, 3), dtype=np.uint8))
        (folder / "notes.txt").write_text("not an image")
    return tmp_path


def test_load_image_folder_shapes_labels_and_scaling(image_root):
    images, labels = load_image_folder(image_root, ("a", "b", "c"), image_size=(16, 16))

    assert images.shape == (30, 16, 16, 3)
    assert images.dtype == np.float32
    assert 0.0 <= images.min() and images.max() <= 1.0
    assert labels.tolist() == [0] * 10 + [1] * 10 + [2] * 10  # label follows class_dirs order


def test_load_image_folder_missing_class_raises(image_root):
    with pytest.raises(FileNotFoundError):
        load_image_folder(image_root, ("a", "missing"))


def test_split_dataset_sizes_one_hot_and_determinism(image_root):
    images, labels = load_image_folder(image_root, ("a", "b", "c"), image_size=(16, 16))

    x_train, x_test, y_train, y_test = split_dataset(images, labels, 3, test_size=0.2)
    again = split_dataset(images, labels, 3, test_size=0.2)

    assert (len(x_train), len(x_test)) == (24, 6)
    assert y_train.shape == (24, 3) and set(np.unique(y_train)) <= {0.0, 1.0}
    assert np.array_equal(x_test, again[1])  # same seed -> same split


def test_holdout_split_is_stratified_and_leaves_test_untouched(image_root):
    images, labels = load_image_folder(image_root, ("a", "b", "c"), image_size=(16, 16))
    x_train, x_test, y_train, y_test = split_dataset(images, labels, 3, test_size=0.2)

    x_fit, x_val, y_fit, y_val = holdout_split(x_train, y_train, val_fraction=0.25)

    assert len(x_fit) + len(x_val) == len(x_train)
    assert set(y_val.argmax(axis=1)) == {0, 1, 2}  # every class appears in validation
    assert len(x_test) == 6


def test_dataset_specs_are_consistent():
    for spec in DATASETS.values():
        assert len(spec.class_dirs) == len(spec.class_names) == spec.num_classes
        assert 0.0 < spec.test_size < 1.0
