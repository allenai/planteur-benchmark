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
| `DATA_S2/S2_<id>.npy` | `(T_s2, 10, 128, 128)` uint16 | Sentinel-2 L2A time series. Bands B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12. Variable T (40–150 unique dates, median 87). |
| `DATA_S1/S1_<id>.npy` | `(T_s1, 2, 128, 128)` float32 | Sentinel-1 (VV, VH) time series, raw linear backscatter (not dB). `0` = no valid observation for that pixel/date (0.3% of pixels, mostly at swath edges). Variable T (40–150, median 123). |
| `DATA_LS/LS_<id>.npy` | `(T_ls, 10, 64, 64)` uint16 | Landsat-8 OLI/TIRS time series. Bands B1,B2,B3,B4,B5,B6,B7,B9,B10,B11. Resampled from Landsat's native 30 m onto a 20 m grid, so 64&times;64 covers the same 1,280 m patch footprint as the 128&times;128 S2/S1 arrays. Variable T (28&ndash;61 unique dates, median 30). |
| `DATA_LS_PAN/LSPAN_<id>.npy` | `(T_ls, 1, 128, 128)` uint16 | Landsat-8 panchromatic B8, resampled from its native 15 m onto the 10 m grid, hence a separate array from the multispectral bands. Same T and same dates as `DATA_LS`. |
| `ANNOTATIONS/TARGET_<id>.npy` | `(1, 128, 128)` uint8 | Semantic labels: 8 classes + 255 void (below). |
| `ANNOTATIONS_GEOTIFF/TARGET_<id>.tif` | `(128, 128)` uint8 | The same semantic labels as georeferenced GeoTIFFs, carrying each patch's CRS and affine transform, `nodata=255`. Pixel values identical to the `.npy`. Patches span five UTM zones (32620/32621/32622/32738/32740); counts are in `ANNOTATIONS_GEOTIFF/_crs_summary.json`. |
| `INSTANCE_ANNOTATIONS/INSTANCES_<id>.npy` | `(128, 128)` int32 | Per-pixel parcel/instance id (0 = background). IDs are globally unique. |
| `INSTANCE_ANNOTATIONS/HEATMAP_<id>.npy` | `(128, 128)` float32 | Object-centerness heatmap (PaPs convention). |
| `INSTANCE_ANNOTATIONS/ZONES_<id>.npy` | `(128, 128)` int32 | Pixel → nearest-center instance (Voronoi zones). |
| `metadata.geojson` | 2,750 rows | Per-patch `ID_PATCH`, `dates-S2`, `dates-S1`, `dates-LS` (`{step: YYYYMMDD}`), `Fold`, `Island`, `geometry` (real patch polygon, EPSG:4326). |
| `NORM_S2_patch.json`, `NORM_S1_patch.json` | — | Per-band train mean/std, keyed `Fold_1`..`Fold_5`. S1 statistics are computed over valid (non-zero) pixels only. |
| `sample/sample.parquet` | 25 rows | Preview split for the Hugging Face dataset viewer: S2 true-colour and colourised label PNGs plus per-patch metadata (island, split, CRS, bounds, sequence lengths, classes present). Random draw, seed 0. Previews only — train from the arrays above. |

Same-day acquisitions are mosaicked to a single observation, so `dates-*` are unique
and chronological and their length matches the array T (PASTIS's one-observation-per-date rule).

Landsat keeps its two source resolutions as two arrays rather than being upsampled into one:
the multispectral bands are 64&times;64 (20 m) and the panchromatic band 128&times;128 (10 m), both
covering the same 1,280 m footprint. `DATA_LS` and `DATA_LS_PAN` share one date list per patch
(`dates-LS`), and a patch's Landsat count is independent of its S2/S1 counts &mdash; patches in a
WRS-2 sidelap see about twice as many Landsat passes (T clusters near 30 for single coverage
and near 60 for sidelap; 2,331 of 2,750 patches are in the 30s).

## 2. Classes (8 + void)

| id | class | id | class |
|--:|---|--:|---|
| 0 | Background | 4 | Sugarcane |
| 1 | Meadow | 5 | Banana |
| 2 | Fruits, vegetables, flowers | 6 | Pineapple |
| 3 | Orchard | 7 | Tropical tuber |

Void is **255**. Agricultural parcels whose crop type is outside this taxonomy (Corn,
Grapevine, Beet, Potatoes, Leguminous fodder, Soybeans, Vanilla, Ylang-ylang,
Coffee/Cacao, plus temperate classes absent from the DROM) are mapped to void and must
be excluded from loss and metrics — **not** folded into Background. Background (0) is
therefore genuine non-agricultural land only, and remains a real trained class. Void
covers 92,145 px (0.20% of pixels; 0.25% of the Background class) across 966 of the
60,576 parcels. The full 26-class → class-id mapping is in `class_map.json`.

**Representing void in a trainer.** Void is `255` on disk, which metric code handles
directly. A trainer that allocates one output channel per class cannot represent 255 —
e.g. UTAE-PaPs `train_panoptic.py`, whose `--out_conv "[32, N]"` sizes the head by
`num_classes`. For those, remap `255 -> 8` and use `num_classes 9`, `void_label 8`. The
eight trained class ids are unchanged either way.


Colors for visualization are in `colormap.txt` (index-aligned to
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


## 4. Per-territory composition

Patch counts: Réunion 813 · Guadeloupe 741 · Martinique 550 · Guyane 506 · Mayotte 140.
The Guadeloupe figure includes 6 patches from Saint-Martin (a COM, not a DROM), which
`metadata.geojson.Island` labels separately as `Saint-Martin`.
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
from dataloader import PLANTEUR_Dataset
ds = PLANTEUR_Dataset(folder="<PLANTEUR_ROOT>", sats=["S2"], folds=[1])      # train, S2
# ds = PLANTEUR_Dataset(folder=..., sats=["S2", "S1"], folds=[5])            # test, S2+S1
# ds = PLANTEUR_Dataset(folder=..., target="instance", folds=[5])           # panoptic
(data, dates), target = ds[0]
```
