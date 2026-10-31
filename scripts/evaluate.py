"""Evaluate a saved model on the test split of a dataset.

The test split is regenerated from the data folder with the same seed, so it is the same
split the model was trained against.

Example:
    python scripts/evaluate.py --data-dir /path/to/dataset --dataset kather-5000 \
        --model runs/default/best_model.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras

import fmanet.model  # noqa: F401  (registers any custom layers for loading)
from fmanet.data import DATASETS, SPLIT_SEED, load_image_folder, split_dataset
from fmanet.training import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="kather-5000")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    images, labels = load_image_folder(args.data_dir, spec.class_dirs)
    _, x_test, _, y_test = split_dataset(
        images, labels, spec.num_classes, spec.test_size, args.seed
    )

    model = keras.models.load_model(args.model)
    result = evaluate_model(model, x_test, y_test, spec.class_names)
    print(json.dumps({k: result[k] for k in ("accuracy", "loss", "num_samples")}, indent=2))
    print(json.dumps(result["confusion_matrix"]))


if __name__ == "__main__":
    main()
