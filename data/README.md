# Data

No data is stored in this repository. Download the datasets yourself and check each dataset's license and terms before use.

## Expected layout

`scripts/train.py` expects a folder with one sub-folder per class. The sub-folder names must match the keys below (they are the names used by the original notebooks).

### `kather-5000`: Kather 2016 colorectal histology (8 classes, 5,000 tiles)

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

### `crc-val-he-7k`: CRC-VAL-HE-7K (9 classes)

```
CRC-VAL-HE-7K/
├── ADI/   (0)   ├── BACK/  (1)   ├── DEB/ (2)
├── LYM/   (3)   ├── MUC/   (4)   ├── MUS/ (5)
├── NORM/  (6)   ├── STR/   (7)   └── TUM/ (8)
```

## Preprocessing (as in the notebooks)

- Read with OpenCV (BGR channel order), resized to 128 x 128, scaled to [0, 1].
- Shuffled with seed 2, then split 90% / 10% (train / test) at random, without stratification.

## Sources and citations

Please verify the exact source and license of the copy you use.

- Kather, J. N. et al. *Multi-class texture analysis in colorectal cancer histology.* Scientific Reports 6, 27988 (2016).
- Kather, J. N. et al. *Predicting survival from colorectal cancer histology slides using deep learning: a retrospective multicenter study.* PLOS Medicine 16(1), e1002730 (2019). This introduced NCT-CRC-HE-100K and the CRC-VAL-HE-7K validation set.

The paper reports its second dataset as NCT-CRC-HE-100K, while the recorded notebook run used CRC-VAL-HE-7K.
