# PLANTEUR — dataset documentation

PLANTEUR is a crop-type **semantic + panoptic** segmentation benchmark over the
five French overseas *départements et régions d'outre-mer* (DROM): **Guadeloupe,
Martinique, Guyane, Réunion, Mayotte**. It is packaged in the **exact original
[PASTIS](https://github.com/VSainteuf/pastis-benchmark) file format** so that PASTIS /
PASTIS-R dataloaders, collate functions and panoptic metrics work unchanged.

## 1. Contents

2,750 patches, 128×128 px @ 10 m/px, Sept 2018 – Dec 2019.

| Path | Shape / dtype | Description |
|---|---|---|
| `DATA_S2/S2_<id>.npy` | `(T_s2, 10, 128, 128)` uint16 | Sentinel-2 L2A time series. Bands B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12. Variable T (~80–92 unique dates). |
| `DATA_S1/S1_<id>.npy` | `(T_s1, 2, 128, 128)` float32 | Sentinel-1 (VV, VH) time series, raw backscatter (not dB). Variable T (~120–150). |
| `ANNOTATIONS/TARGET_<id>.npy` | `(1, 128, 128)` | Semantic labels, 8 classes (below). |
| `INSTANCE_ANNOTATIONS/INSTANCES_<id>.npy` | `(128, 128)` int32 | Per-pixel parcel/instance id (0 = background). IDs are globally unique. |
| `INSTANCE_ANNOTATIONS/HEATMAP_<id>.npy` | `(128, 128)` float32 | Object-centerness heatmap (PaPs convention). |
| `INSTANCE_ANNOTATIONS/ZONES_<id>.npy` | `(128, 128)` int32 | Pixel → nearest-center instance (Voronoi zones). |
| `metadata.geojson` | 2,750 rows | Per-patch `ID_PATCH`, `dates-S2`, `dates-S1` (`{step: YYYYMMDD}`), `Fold`, `geometry` (real patch polygon, EPSG:4326). |
| `NORM_S2_patch.json`, `NORM_S1_patch.json` | — | Per-band train mean/std, keyed `Fold_1`..`Fold_5`. |

Same-day acquisitions are mosaicked to a single observation, so `dates-*` are unique
and chronological and their length matches the array T (PASTIS's one-observation-per-date rule).

## 2. Classes (8)

| id | class | id | class |
|--:|---|--:|---|
| 0 | Background | 4 | Sugarcane |
| 1 | Meadow | 5 | Banana |
| 2 | Fruits, vegetables, flowers | 6 | Pineapple |
| 3 | Orchard | 7 | Tropical tuber |

Background (0) is non-agricultural land plus agricultural parcels whose crop type is
outside this taxonomy. Colors for visualization are in `colormap.txt` (index-aligned to
the class ids, with a trailing white entry for nodata/void).

## 3. Splits

Spatial-block, territory-stratified 5-fold-style split written in `metadata.geojson.Fold`:

| Fold | role | patches |
|--:|---|--:|
| 1 | train | 1,896 |
| 4 | val   | 411 |
| 5 | test  | 443 |

Whole ~11 km spatial blocks are assigned to a single split (no spatial leakage), and each
territory (hence every crop class) is represented across train/val/test.

## 4. Per-territory composition

Patch counts: Réunion 813 · Guadeloupe 741 · Martinique 550 · Guyane 506 · Mayotte 140.
Total unique parcels: **60,576**. Per-class distributions (pixels / windows / parcels) per
island are shown in `../images/{pixels,windows,parcels}_per_class_per_island.png`.
Island crop signatures: sugarcane dominates Réunion & Guadeloupe; banana dominates
Martinique & Mayotte; Guyane is meadow/orchard-heavy.

## 5. Provenance

- **Imagery**: Sentinel-2 L2A and Sentinel-1 retrieved from the Microsoft Planetary
  Computer (Copernicus data; Sen2Cor L2A). This is the one intentional departure from
  PASTIS, which used THEIA/MAJA.
- **Annotations**: the French *Registre Parcellaire Graphique* (RPG 2019), produced by
  IGN, rasterized to the patch grid and mapped to the 8-class taxonomy.

## 6. Loading

```python
import sys; sys.path.append("code")
from dataloader import PASTIS_Dataset
ds = PASTIS_Dataset(folder="<PLANTEUR_ROOT>", sats=["S2"], folds=[1])      # train, S2
# ds = PASTIS_Dataset(folder=..., sats=["S2", "S1"], folds=[5])            # test, S2+S1
# ds = PASTIS_Dataset(folder=..., target="instance", folds=[5])           # panoptic
(data, dates), target = ds[0]
```
