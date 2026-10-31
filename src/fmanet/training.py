"""Training and evaluation helpers.

Two protocols are supported:

* ``notebook`` (default): the test split is also passed as ``validation_data``, so early
  stopping and best-checkpoint selection are driven by the test set. This is what the
  original notebooks did, and it makes the reported test accuracy optimistic.
* ``holdout``: a validation set is taken from the training data and the test set is used
  only once, for the final evaluation. Use this to produce numbers that are not biased by
  checkpoint selection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .data import SPLIT_SEED, holdout_split


@dataclass
class TrainConfig:
    """Training settings. Defaults match the original notebooks."""

    epochs: int = 50
    batch_size: int = 16
    monitor: str = "val_accuracy"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.001
    protocol: str = "notebook"
    val_fraction: float = 0.1
    seed: int = SPLIT_SEED


def make_callbacks(cfg: TrainConfig, checkpoint_path: Path | None = None) -> list:
    """Early stopping (restoring the best weights) plus an optional best-model checkpoint."""
    callbacks: list = [
        keras.callbacks.EarlyStopping(
            monitor=cfg.monitor,
            mode=cfg.mode,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            restore_best_weights=True,
        )
    ]
    if checkpoint_path is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path), monitor=cfg.monitor, mode=cfg.mode, save_best_only=True
            )
        )
    return callbacks


def evaluate_model(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    class_names: tuple[str, ...] | list[str],
    batch_size: int = 32,
) -> dict:
    """Accuracy, loss, per-class report and confusion matrix on ``(x, y)``."""
    loss, accuracy = model.evaluate(x, y, batch_size=batch_size, verbose=0)[:2]
    y_true = y.argmax(axis=1)
    y_pred = model.predict(x, batch_size=batch_size, verbose=0).argmax(axis=1)
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "num_samples": int(len(y)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_and_evaluate(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: tuple[str, ...] | list[str],
    cfg: TrainConfig,
    output_dir: Path | None = None,
) -> dict:
    """Fit the model under the chosen protocol and evaluate it on the test split."""
    if cfg.protocol == "notebook":
        validation = (x_test, y_test)
    elif cfg.protocol == "holdout":
        x_train, x_val, y_train, y_val = holdout_split(x_train, y_train, cfg.val_fraction, cfg.seed)
        validation = (x_val, y_val)
    else:
        raise ValueError(f"Unknown protocol {cfg.protocol!r}; use 'notebook' or 'holdout'.")

    checkpoint = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "best_model.keras"

    history = model.fit(
        x_train,
        y_train,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        validation_data=validation,
        callbacks=make_callbacks(cfg, checkpoint),
        shuffle=True,
        verbose=2,
    )

    result = evaluate_model(model, x_test, y_test, class_names, cfg.batch_size)
    result["protocol"] = cfg.protocol
    result["epochs_run"] = len(history.history["loss"])
    result["config"] = asdict(cfg)

    if output_dir is not None:
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        (output_dir / "history.json").write_text(
            json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()}, indent=2)
        )
    return result
