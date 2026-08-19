# 🌴 PLANTEUR 🌴 — Panoptic overseas crop mapping from satellite time series

PLANTEUR is a benchmark dataset for **panoptic and semantic segmentation of agricultural
parcels** in the **French overseas territories** (DROM: Guadeloupe, Martinique, Guyane,
Réunion, Mayotte). It is the tropical/overseas counterpart to
[PASTIS](https://github.com/VSainteuf/pastis-benchmark), and is distributed in the **exact
same file format** so that existing PASTIS / PASTIS-R dataloaders, collate functions, and
panoptic metrics work **unchanged**.

> **Note (pre-release):** items marked ⚠️ **TODO** need to be finalized before public
> release (dataset host/DOI, citation, license holder, leaderboard numbers).

## The PLANTEUR Dataset

- **Presentation**

Each patch is a Sentinel-2 (and Sentinel-1) multispectral image time series of variable
length, with per-pixel panoptic annotations (instance index + semantic label) derived from
the French Registre Parcellaire Graphique (RPG). Crops are grouped into **8 classes** built
around the crops that actually occur across the DROM (sugarcane, banana, pineapple, tropical
tuber, …). We provide an official spatial-block, territory-stratified train/val/test split.

- **Dataset in numbers**

| | | |
|---|---|---|
| ▶️ 2,750 time series | ▶️ 60,576 parcels | ▶️ 8 crop classes |
| ▶️ **128×128 px / image** | ▶️ **~80–92 S2 acquisitions / series** | ▶️ **10 m / pixel** |
| ▶️ **10 S2 + 2 S1 bands** | ▶️ **5 overseas territories** | ▶️ **Sept 2018 – Dec 2019** |

- 🛰️ **Multimodal (optical + radar).** Every patch has aligned Sentinel-1 (VV/VH)
  observations in addition to Sentinel-2, for optical–radar fusion experiments.

## Usage

- **Download**

The dataset (~115 GB) will be hosted on the Hugging Face Hub:

> **https://huggingface.co/datasets/allenai/planteur-benchmark**  ⚠️ *(upload pending)*

Internal (pre-release) location on Weka: `/weka/dfive-default/piperw/pastis2_drom_pastis_format`.

- **Data loading**

This repository contains a PyTorch dataset class in `code/dataloader.py` that loads PLANTEUR
for semantic and panoptic segmentation (identical API to PASTIS). `code/collate.py` provides
`pad_collate` for temporally padding variable-length series. For the pixel-set format use
`code/dataloader_pixelset.py`.

```python
import sys; sys.path.append("code")
from dataloader import PASTIS_Dataset
train = PASTIS_Dataset(folder="<PLANTEUR_ROOT>", sats=["S2"], folds=[1])
test  = PASTIS_Dataset(folder="<PLANTEUR_ROOT>", sats=["S2", "S1"], target="instance", folds=[5])
```

- **Metrics**

`code/panoptic_metrics.py` provides the PyTorch panoptic metrics (SQ/RQ/PQ). For semantic
segmentation, report mean IoU over the 8 classes.

- **Visualization**

See `demo.ipynb` for loading and visualizing samples. Per-island class distributions are in
`images/`.

## Splits

| Fold | role | patches |
|--:|---|--:|
| 1 | train | 1,896 |
| 4 | val | 411 |
| 5 | test | 443 |

## Documentation

The 8-class nomenclature and full format specification (arrays, shapes, metadata, provenance)
are in [`documentation/planteur-documentation.md`](documentation/planteur-documentation.md).
Class names: `documentation/label_names.json`; visualization colors: `documentation/colormap.txt`.

![pixels per class per island](images/pixels_per_class_per_island.png)

## References

⚠️ **TODO — citation.**
```
@misc{planteur2026,
  title  = {PLANTEUR: Panoptic overseas crop mapping from satellite image time series},
  author = {TODO},
  year   = {2026}
}
```
This dataset follows the format of PASTIS; if you use it, please also cite PASTIS:
```
@article{garnot2021panoptic,
  title   = {Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks},
  author  = {Sainte Fare Garnot, Vivien and Landrieu, Loic},
  journal = {ICCV},
  year    = {2021}
}
```

## Credits

- **Imagery**: Sentinel-2 / Sentinel-1 (Copernicus) retrieved via the
  [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/).
- **Annotations**: the French [Registre Parcellaire Graphique (RPG 2019)](https://www.data.gouv.fr/en/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/), produced by IGN.
- **Format**: adapted from [PASTIS](https://github.com/VSainteuf/pastis-benchmark) (V. Sainte Fare Garnot & L. Landrieu), MIT-licensed.
