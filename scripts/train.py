"""Train FMANet on an image-folder dataset and evaluate it on the test split.

Example:
    python scripts/train.py --data-dir /path/to/dataset --dataset kather-5000 --protocol holdout
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import keras

from fmanet.data import DATASETS, load_image_folder, split_dataset
from fmanet.model import build_fmanet, compile_fmanet
from fmanet.training import TrainConfig, train_and_evaluate

logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Folder holding one sub-folder per class."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="kather-5000")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument(
        "--protocol",
        choices=("notebook", "holdout"),
        default="notebook",
        help="'notebook' reproduces the original (test split also used for early stopping); "
        "'holdout' keeps the test set untouched until the end.",
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override the notebook default."
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=TrainConfig.val_fraction,
        help="Share of the training data used for validation (holdout protocol only).",
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keras.utils.set_random_seed(args.seed)
    spec = DATASETS[args.dataset]

    images, labels = load_image_folder(args.data_dir, spec.class_dirs)
    logger.info(
        "Loaded %d images from %s (%d classes)", len(images), args.data_dir, spec.num_classes
    )
    x_train, x_test, y_train, y_test = split_dataset(
        images, labels, spec.num_classes, spec.test_size, args.seed
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        protocol=args.protocol,
        val_fraction=args.val_fraction,
        seed=args.seed,
        patience=spec.patience or TrainConfig.patience,
    )
    model = compile_fmanet(build_fmanet(num_classes=spec.num_classes), args.learning_rate)
    result = train_and_evaluate(
        model, x_train, y_train, x_test, y_test, spec.class_names, cfg, args.output_dir
    )
    logger.info("Test accuracy (%s protocol): %.4f", args.protocol, result["accuracy"])


if __name__ == "__main__":
    main()
