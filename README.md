# FMANet

**Colorectal tissue classification from histopathological images, using fused multi-level attention features and constraint fusion.**

[![CI](https://github.com/RashiChauhan67/FMANet/actions/workflows/ci.yml/badge.svg)](https://github.com/RashiChauhan67/FMANet/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)

> **Paper.** R. Chauhan, K. Mohan, P. Singh. *Fused multi-level attention features with a constraint fusion network for colorectal tissue classification using histopathological images.* **Computers and Electrical Engineering**, vol. 124, article 110296, 2025. DOI: [10.1016/j.compeleceng.2025.110296](https://doi.org/10.1016/j.compeleceng.2025.110296)
>
> **Related repository.** Code released by the paper's co-author: [KarnatiMOHAN/FMANet_colorectal-tissue_classification](https://github.com/KarnatiMOHAN/FMANet_colorectal-tissue_classification).

[Overview](#overview) · [Architecture](#architecture) · [Quick start](#quick-start) · [Datasets](#datasets) · [Results](#results) · [Notes](#notes) · [Citation](#citation)

## Overview

FMANet classifies 128 x 128 H&E tissue tiles. A **low-level** and a **high-level** feature stream, each refined by a dilated residual-inception block, are pooled and embedded by shared dense layers. **Constraint fusion** then weights each stream by its cosine similarity to the combined features, and the weighted features feed a softmax classifier.

This repository has a cleaned, tested Keras 3 implementation, the original experiment notebooks, and the recorded results.

| | Kather 5,000 | CRC-VAL-HE-7K |
|---|---|---|
| Classes | 8 | 9 |
| Test tiles | 500 | 718 |
| Test accuracy | **95.0%** | **97.35%** |
| Macro F1 | 0.95 | 0.97 |
| Parameters | 5.39 M | 5.39 M |

Accuracies are the values recorded in the notebooks (see [Notes](#notes)).

## Architecture

![FMANet architecture](docs/figures/fmanet_architecture.png)

*Schematic of the full network. Layer widths in `src/fmanet/model.py` are authoritative.*

![FMANet modules](docs/figures/fmanet_modules.png)

*The dilated residual-inception block, channel attention and spatial attention modules.*

The graph built by `src/fmanet/model.py`:

```mermaid
flowchart LR
    I["Input<br/>128 x 128 x 3"] --> LL["Low-level stream<br/>3 x (Conv 64 + BN), concat"]
    LL --> A1["Dilated residual-inception<br/>192 filters, rates 1-2-3<br/>-> A1"]
    LL --> HL["High-level stream<br/>3 x (Conv 64 + BN), concat"]
    HL --> A2["Dilated residual-inception<br/>192 filters, rates 1-2-3<br/>-> A2"]
    A1 --> G1["GAP -> Dense 384"]
    A2 --> G2["GAP -> Dense 384"]
    A1 --> CAT["concat -> GAP"]
    A2 --> CAT
    G1 --> S["Shared Dense 768 x 2"]
    G2 --> S
    CAT --> S
    S --> W["tanh cosine similarity to the<br/>combined feature -> weights a_l, a_g<br/>(a_l + a_g = 1)"]
    W --> F["weighted concat<br/>Dense 64 -> Dense N, softmax"]
```

## Quick start

```bash
git clone https://github.com/RashiChauhan67/FMANet.git
cd FMANet
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Train and evaluate:

```bash
# Original protocol: the test split also drives early stopping (see Notes)
python scripts/train.py --data-dir /path/to/Kather_texture_2016_image_tiles_5000 \
    --dataset kather-5000 --output-dir runs/kather

# Leakage-free protocol: validation is taken from the training data, the test set is used once
python scripts/train.py --data-dir /path/to/CRC-VAL-HE-7K --dataset crc-val-he-7k \
    --protocol holdout --output-dir runs/crc_val_holdout

# Evaluate a saved model on the regenerated test split
python scripts/evaluate.py --data-dir /path/to/CRC-VAL-HE-7K --dataset crc-val-he-7k \
    --model runs/crc_val_holdout/best_model.keras
```

Defaults: Nadam (1e-4), batch size 16, up to 50 epochs, early stopping on validation accuracy (patience 10, best weights restored). Each run writes `best_model.keras`, `history.json` and `metrics.json`.

Use the model directly:

```python
from fmanet.model import build_fmanet, compile_fmanet

model = compile_fmanet(build_fmanet(num_classes=8))  # 5,389,832 parameters
```

## Datasets

| Key | Dataset | Classes | Tiles | Test split |
|---|---|---|---|---|
| `kather-5000` | Colorectal histology (Kather et al., 2016) | 8 | 5,000 | 10% |
| `crc-val-he-7k` | CRC-VAL-HE-7K (Kather et al., 2018) | 9 | 7,180 | 10% |

Both are on Zenodo under CC BY 4.0. Data is not included; see [`data/README.md`](data/README.md) for links, folder layout and citations. Images are resized to 128 x 128, scaled to [0, 1] and read with OpenCV.

## Results

Recorded in the original notebooks. **These were not re-run for this repository.** Per-class tables are in [`results/recorded_results.md`](results/recorded_results.md).

| Dataset | Test tiles | Accuracy | Macro F1 | Epochs run | Per-class AUC |
|---|---|---|---|---|---|
| Kather 5,000 | 500 | **0.950** | 0.95 | 25 | 0.991 to 0.9999 |
| CRC-VAL-HE-7K | 718 | **0.9735** | 0.97 | 9 | 0.988 to 1.000 |

| Kather 5,000 | CRC-VAL-HE-7K |
|---|---|
| ![](results/figures/kather_confusion_matrix_normalized.png) | ![](results/figures/crc_val_confusion_matrix_normalized.png) |
| ![](results/figures/kather_roc.png) | ![](results/figures/crc_val_roc.png) |
| ![](results/figures/kather_radar.png) | ![](results/figures/crc_val_radar.png) |

t-SNE of the features before and after training:

| | Original features | Trained features |
|---|---|---|
| **Kather 5,000** | ![](results/figures/kather_tsne_original_features.png) | ![](results/figures/kather_tsne_trained_features.png) |
| **CRC-VAL-HE-7K** | ![](results/figures/crc_val_tsne_original_features.png) | ![](results/figures/crc_val_tsne_trained_features.png) |

Class indices: Kather 0 tumor, 1 stroma, 2 complex, 3 lympho, 4 debris, 5 mucosa, 6 adipose, 7 empty. CRC-VAL-HE-7K 0 ADI, 1 BACK, 2 DEB, 3 LYM, 4 MUC, 5 MUS, 6 NORM, 7 STR, 8 TUM.

## Notes

**Implementation.**
- The default model reproduces the graph trained in the notebooks. In it, the spatial- and channel-attention modules are defined but their outputs are not connected to the classifier (5,389,832 parameters for 8 classes).
- `build_fmanet(use_attention_modules=True)` connects them as shown in the architecture figure (5,432,145 parameters). That variant has not been trained here.
- The parameter counts match the notebooks' `model.summary()` output; the tests check them.

**Evaluation.**
- The notebooks pass the test split as `validation_data`, and early stopping and checkpoint selection monitor it, so the recorded accuracies equal the best validation accuracy (0.9500 and 0.9735) and are optimistic. Use `--protocol holdout` for numbers that are not selected on the test set.
- Splits are random (not stratified) at the image level with a fixed seed. No framework-level seeds are set, so a re-run will not reproduce the recorded numbers exactly. This port reads files in sorted order; the notebooks used `os.listdir` order, so the exact test tiles can differ slightly.
- The FLOPs figure in the notebooks (about 139 GFLOPs) comes from a hand-written per-layer estimator and is approximate.

**Notebooks.** `colorectal-tissue_dataset_1.ipynb` contains some stray radar charts from other datasets near its end. The second notebook is included only up to its colorectal section (`colorectal-tissue_dataset_2_crc_only.ipynb`); an unrelated experiment that followed it was removed.

## Repository structure

```
.
├── src/fmanet/          # model, data loading, training and evaluation
├── scripts/             # train.py, evaluate.py
├── tests/               # unit tests (synthetic data; no dataset needed)
├── notebooks/           # original experiment notebooks
├── docs/figures/        # architecture figures
├── results/             # recorded notebook results and figures
└── data/README.md       # dataset instructions
```

Development: `pip install -e ".[dev]"`, then `ruff check .` and `pytest`.

## Citation

If you use this code, please cite the paper (a `CITATION.cff` is provided):

```bibtex
@article{chauhan2025fmanet,
  author  = {Chauhan, Rashi and Karnati, Mohan and Singh, Pradeep},
  title   = {Fused multi-level attention features with a constraint fusion network for colorectal tissue classification using histopathological images},
  journal = {Computers and Electrical Engineering},
  volume  = {124},
  pages   = {110296},
  year    = {2025},
  doi     = {10.1016/j.compeleceng.2025.110296}
}
```

## License

MIT. See [LICENSE](LICENSE). The datasets have their own licenses; check them before reuse.

**Figures.** The architecture figures in `docs/figures/` are reproduced from the paper by its authors. Reuse of those figures is subject to the publisher's terms (Elsevier); the MIT license covers the code and the result plots generated from the notebooks.
