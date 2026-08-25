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
| ▶️ 2,750 time series | ▶️ 60,576 parcels | ▶️ 8 classes + void |
| ▶️ **128×128 px / image** | ▶️ **40–150 S2 acquisitions / series** | ▶️ **10 m / pixel** |
| ▶️ **10 S2 + 2 S1 bands** | ▶️ **6 territories (5 DROM + Saint-Martin)** | ▶️ **Sept 2018 – Dec 2019** |

- 🛰️ **Multimodal (optical + radar).** Every patch has aligned Sentinel-1 (VV/VH)
  observations in addition to Sentinel-2, for optical–radar fusion experiments.

## Usage

- **Download**

The dataset (~115 GB) will be hosted on the Hugging Face Hub:

> **https://huggingface.co/datasets/allenai/planteur-benchmark**

Internal (pre-release) location on Weka: `/weka/dfive-default/piperw/pastis2_drom_pastis_format`.

- **Data loading**

This repository contains a PyTorch dataset class in `code/dataloader.py` that loads PLANTEUR
for semantic and panoptic segmentation (identical API to PASTIS). `code/collate.py` provides
`pad_collate` for temporally padding variable-length series. For the pixel-set format use
`code/dataloader_pixelset.py`.

```python
import sys; sys.path.append("code")
from dataloader import PLANTEUR_Dataset
train = PLANTEUR_Dataset(folder="<PLANTEUR_ROOT>", sats=["S2"], folds=[1])
test  = PLANTEUR_Dataset(folder="<PLANTEUR_ROOT>", sats=["S2", "S1"], target="instance", folds=[5])
```

- **Metrics**

`code/panoptic_metrics.py` provides the PyTorch panoptic metrics (SQ/RQ/PQ). Void (255)
must be excluded from both loss and metrics: for semantic segmentation report mean IoU
over the 8 classes with void ignored, and for panoptic use
`PanopticMeter(num_classes=8, void_label=255, background_label=0)`, which scores PQ over
the 7 crop classes (background and void both excluded).

- **Visualization**

See `demo.ipynb` for loading and visualizing samples. Per-island class distributions are in
`images/`.

## Splits

### Default (spatial-block, territory-stratified)

| Fold | role | patches |
|--:|---|--:|
| 1 | train | 1,896 |
| 4 | val | 411 |
| 5 | test | 443 |

### Leave-one-island-out (LOIO)

`splits/loio.json` holds five additional configurations. For each held-out territory,
`test` is all of its patches; `train`/`val` come from the same spatial-block,
territory-stratified 70/15/15 assignment over the remaining territories.

| held-out (test) | test | val | train |
|---|--:|--:|--:|
| Guadeloupe | 735 | 265 | 1,750 |
| Martinique | 550 | 250 | 1,950 |
| Guyane | 506 | 284 | 1,960 |
| Réunion | 813 | 217 | 1,720 |
| Mayotte | 140 | 314 | 2,296 |

Per-patch territory is in `metadata.geojson.Island`. Saint-Martin (6 patches) is
labelled but never held out — it stays in `train` for every fold.

## Documentation

The 8-class nomenclature and full format specification (arrays, shapes, metadata, provenance)
are in [`documentation/planteur-documentation.md`](documentation/planteur-documentation.md).
Class names: `documentation/label_names.json`; the full 26-class → bg8void mapping:
`documentation/bg8void_class_map.json`; visualization colors: `documentation/colormap.txt`.

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
