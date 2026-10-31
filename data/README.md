# Data

No data is stored in this repository. Download the datasets yourself and check each dataset's license and terms before use.

## Expected layout

`scripts/train.py` expects a folder with one sub-folder per class. The sub-folder names must match the ones below (they are the names used by the original notebooks).

### `kather-5000`: colorectal histology (8 classes, 5,000 tiles)

```
Kather_texture_2016_image_tiles_5000/
├── 01_TUMOR/      (label 0)
├── 02_STROMA/     (label 1)
├── 03_COMPLEX/    (label 2)
├── 04_LYMPHO/     (label 3)
├── 05_DEBRIS/     (label 4)
├── 06_MUCOSA/     (label 5)
├── 07_ADIPOSE/    (label 6)
└── 08_EMPTY/      (label 7)
```

### `crc-val-he-7k`: CRC-VAL-HE-7K (9 classes, 7,180 tiles)

```
CRC-VAL-HE-7K/
├── ADI/   (0)   ├── BACK/  (1)   ├── DEB/ (2)
├── LYM/   (3)   ├── MUC/   (4)   ├── MUS/ (5)
├── NORM/  (6)   ├── STR/   (7)   └── TUM/ (8)
```

## Preprocessing (as in the notebooks)

- Read with OpenCV (BGR channel order), resized to 128 x 128, scaled to [0, 1].
- Shuffled with seed 2, then split 90% / 10% (train / test) at random, without stratification.

## Sources (as given in the paper)

| Dataset | Source | License on Zenodo |
|---|---|---|
| Colorectal histology (paper's Dataset-1) | Kather, Zöllner, Bianconi, Melchers, Schad, Gaiser et al. *Collection of textures in colorectal cancer histology.* Zenodo, 2016. [doi:10.5281/zenodo.53169](https://zenodo.org/records/53169). Collected at the University Medical Center Mannheim (UMM). | CC BY 4.0 |
| CRC-VAL-HE-7K (paper's Dataset-2, called NCT-CRC-HE-100K) | Kather, Halama, Marx. *100,000 histological images of human colorectal cancer and healthy tissue.* Zenodo, 2018. [doi:10.5281/zenodo.1214456](https://zenodo.org/records/1214456). The 7,180-image set from fifty patients. | CC BY 4.0 |

Licenses were read from the Zenodo records. Please confirm them on the record page before reuse and cite the datasets when you publish results.

Related paper for the second dataset: Kather, J. N. et al. *Predicting survival from colorectal cancer histology slides using deep learning: a retrospective multicenter study.* PLOS Medicine 16(1), e1002730 (2019).
