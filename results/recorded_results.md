# Recorded results

Everything on this page is copied from the saved outputs of the original notebooks in [`../notebooks/`](../notebooks/). **None of it was re-run** for this repository.

Protocol of the recorded runs: random 90% / 10% split (seed 2); the 10% split was also used as `validation_data`, with early stopping and best-checkpoint selection on validation accuracy. The test accuracy is therefore the best validation accuracy on the same data (see the README).

Model: 5,389,832 parameters (8 classes) or 5,389,897 (9 classes). Model size printed by the notebooks: 20.56 MB. Hand-estimated cost printed by the notebooks: about 139 GFLOPs / 69.5 GMACs (approximate).

## Kather 5,000 (`colorectal-tissue_dataset_1.ipynb`)

Test loss 0.2004, test accuracy 0.9500. Training ran 25 epochs; best validation accuracy 0.9500.

| Class | Name | Precision | Recall | F1 | Support | AUC |
|---|---|---|---|---|---|---|
| 0 | tumor | 0.96 | 0.99 | 0.97 | 77 | 0.9986 |
| 1 | stroma | 0.90 | 0.98 | 0.94 | 61 | 0.9979 |
| 2 | complex | 0.89 | 0.87 | 0.88 | 55 | 0.9911 |
| 3 | lympho | 1.00 | 0.90 | 0.95 | 72 | 0.9974 |
| 4 | debris | 0.98 | 0.92 | 0.95 | 51 | 0.9988 |
| 5 | mucosa | 0.93 | 0.98 | 0.96 | 58 | 0.9978 |
| 6 | adipose | 0.94 | 1.00 | 0.97 | 59 | 0.9999 |
| 7 | empty | 1.00 | 0.94 | 0.97 | 67 | 0.9998 |
| | accuracy | | | 0.95 | 500 | |
| | macro avg | 0.95 | 0.95 | 0.95 | 500 | |
| | weighted avg | 0.95 | 0.95 | 0.95 | 500 | |

## CRC-VAL-HE-7K (`colorectal-tissue_dataset_2_crc_only.ipynb`)

Test loss 0.1159, test accuracy 0.9735. Training ran 9 epochs; best validation accuracy 0.9735.

| Class | Name | Precision | Recall | F1 | Support | AUC |
|---|---|---|---|---|---|---|
| 0 | ADI | 0.98 | 0.95 | 0.97 | 131 | 0.9991 |
| 1 | BACK | 1.00 | 1.00 | 1.00 | 79 | 1.0000 |
| 2 | DEB | 0.97 | 1.00 | 0.98 | 31 | 0.9999 |
| 3 | LYM | 1.00 | 1.00 | 1.00 | 63 | 1.0000 |
| 4 | MUC | 1.00 | 0.97 | 0.98 | 100 | 1.0000 |
| 5 | MUS | 0.99 | 0.97 | 0.98 | 80 | 0.9997 |
| 6 | NORM | 0.83 | 1.00 | 0.91 | 49 | 0.9997 |
| 7 | STR | 0.95 | 0.89 | 0.92 | 44 | 0.9879 |
| 8 | TUM | 0.99 | 0.98 | 0.98 | 141 | 0.9978 |
| | accuracy | | | 0.97 | 718 | |
| | macro avg | 0.97 | 0.97 | 0.97 | 718 | |
| | weighted avg | 0.98 | 0.97 | 0.97 | 718 | |

AUC values are read from the legends of the ROC plots in the notebooks.
