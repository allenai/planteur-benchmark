# PLANTEUR — dataset documentation

PLANTEUR is a crop-type **semantic + panoptic** segmentation benchmark over the
five French overseas *départements et régions d'outre-mer* (DROM): **Guadeloupe,
Martinique, Guyane, Réunion, Mayotte**, plus 6 patches from Saint-Martin, an overseas
collectivity (*collectivité d'outre-mer*, COM). It is packaged in almost the **exact same
[PASTIS](https://github.com/VSainteuf/pastis-benchmark) file format** so that existing PASTIS
dataloaders work mostly unchanged with the Sentinel-2, Sentinel-1, and label arrays
(Landsat-8, which PASTIS does not have, needs the loader in this repository).

## 1. Contents

2,750 patches, 128×128 px @ 10 m/px, Sept 2018 – Dec 2019.

| Path | Shape / dtype | Description |
|---|---|---|
| `DATA_S2/S2_<id>.npy` | `(T_s2, 10, 128, 128)` uint16 | Sentinel-2 L2A time series. Bands B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12. Variable T (40–150 unique dates, median 87). |
| `DATA_S1/S1_<id>.npy` | `(T_s1, 2, 128, 128)` float32 | Sentinel-1 (VV, VH) time series, radiometrically terrain-corrected (RTC) linear backscatter (not dB). `0` = no valid observation for that pixel/date (0.3% of pixels, mostly at swath edges). Variable T (40–150, median 123). |
| `DATA_LS/LS_<id>.npy` | `(T_ls, 10, 64, 64)` uint16 | Landsat-8 OLI/TIRS Collection 2 Level-1 time series (scaled digital numbers, not surface reflectance). Bands B1,B2,B3,B4,B5,B6,B7,B9,B10,B11. Resampled from Landsat's native 30 m onto a 20 m grid, so 64&times;64 covers the same 1,280 m patch footprint as the 128&times;128 S2/S1 arrays. Variable T (28&ndash;61 unique dates, median 30). |
| `DATA_LS_PAN/LSPAN_<id>.npy` | `(T_ls, 1, 128, 128)` uint16 | Landsat-8 panchromatic B8, resampled from its native 15 m onto the 10 m grid, hence a separate array from the multispectral bands. Same T and same dates as `DATA_LS`. |
| `ANNOTATIONS/TARGET_<id>.npy` | `(1, 128, 128)` uint8 | Semantic labels: 8 classes + 255 void (below). |
| `ANNOTATIONS_GEOTIFF/TARGET_<id>.tif` | `(128, 128)` uint8 | The same semantic labels as georeferenced GeoTIFFs, carrying each patch's CRS and affine transform, `nodata=255`. Pixel values identical to the `.npy`. Patches span five UTM zones (32620/32621/32622/32738/32740); counts are in `ANNOTATIONS_GEOTIFF/_crs_summary.json`. |
| `INSTANCE_ANNOTATIONS/INSTANCES_<id>.npy` | `(128, 128)` int32 | Per-pixel parcel/instance id (0 = no parcel: Background or void). IDs are globally unique, so a parcel crossing a patch border has the same ID in each patch. |
| `INSTANCE_ANNOTATIONS/HEATMAP_<id>.npy` | `(128, 128)` float32 | Object-centerness heatmap (PaPs convention). |
| `INSTANCE_ANNOTATIONS/ZONES_<id>.npy` | `(128, 128)` int32 | Pixel → nearest-center instance (Voronoi zones). |
| `metadata.geojson` | 2,750 rows | Per-patch `ID_PATCH`, `dates-S2`, `dates-S1`, `dates-LS` (`{step: YYYYMMDD}`), `orbits-S1` (`{step: "A"\|"D"\|"AD"}`, the Sentinel-1 orbit direction of each `dates-S1` step; `"AD"` = same-day ascending and descending passes averaged), `Fold`, `split` (`train`/`val`/`test`, same assignment as `Fold`), `Island`, `geometry` (real patch polygon, EPSG:4326). |
| `NORM_S2_patch.json`, `NORM_S1_patch.json`, `NORM_LS_patch.json`, `NORM_LS_PAN_patch.json` | — | Per-band train mean/std, keyed `Fold_1`..`Fold_5`. S1 and Landsat statistics are computed over valid (non-zero) pixels only. |
| `sample/sample.parquet` | 25 rows | Preview split for the Hugging Face dataset viewer: S2 true-colour and colourised label PNGs plus per-patch metadata (island, split, CRS, bounds, sequence lengths, classes present). Random draw, seed 0. Previews only — train from the arrays above. |
| `DATA_S2_MONTHLY/S2_<id>.npy` | `(12, 12, 128, 128)` uint16 | The paper's monthly inputs: 12 monthly 2019 mosaics (month m = 2019-01-01 + 30m days), the `DATA_S2` bands then B01,B09. Same for `DATA_S1_MONTHLY`, `DATA_LS_MONTHLY`, `DATA_LS_PAN_MONTHLY`; `MONTHLY/months.json` lists which months have a scene. |
| `NORM_S2_MONTHLY_patch.json` | — | Train mean/std of the 10 `DATA_S2` bands of the monthly mosaics. |
| `ANNOTATIONS_BALANCED/TARGET_<id>.npy` | `(1, 128, 128)` uint8 | Class-balanced training labels: in `train` patches, Background pixels set to void until Background equals the crop pixels; `val`/`test` identical to `ANNOTATIONS`. |
| `splits/fewshot_pixels.json` | — | Few-shot training pixels, `[row, col, class]` per patch, for X = 10/25/100/1000 per class. |
| `EMB_*/<name>_<id>.npy` | `(C, 128, 128)` float32 | Precomputed embeddings on the label grid (AlphaEarth, Tessera v1/v1.1/v2, OlmoEarth); `EMB_MISSING.json` lists patches without one. No-data pixels: Tessera v1 all-zero vectors (149 patches, 7 entirely zero: 291, 934, 989, 1015, 1038, 1058, 1980) or `NaN` (6 patches); Tessera v1.1 `NaN` (4 patches); AEF would use `-1` but has none; Tessera v2 and OlmoEarth v1.3 have none. |

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
be excluded from loss and metrics — **not** folded into Background. Background (0) is land
outside the RPG plus RPG parcels whose crop code has no class in the 26-class source
nomenclature (e.g. grain pulses, spring cereal variants, aromatic plants, unspecified
crops), and remains a real trained class. Void covers 92,145 px (0.20% of pixels) in 385
patches. Only crop parcels (classes 1–7) are instances: no instance in
`INSTANCE_ANNOTATIONS` contains Background or void pixels, and every parcel has a single
class.
The full 26-class → class-id mapping is in `class_map.json`.

**Representing void in a trainer.** Void is `255` on disk, which metric code handles
directly. A trainer that allocates one output channel per class cannot represent 255 —
e.g. UTAE-PaPs `train_panoptic.py`, whose `--out_conv "[32, N]"` sizes the head by
`num_classes`. For those, remap `255 -> 8` and use `num_classes 9`, `void_label 8`. The
eight trained class ids are unchanged either way.


Colors for visualization are in `colormap.txt`, one `<class id> <R> <G> <B>` line per
trained class (0–7), matching the map in the README. Void has no entry; render it as dark
grey (43 52 56), as the map does.

## 3. Splits

Spatial-block, territory-stratified split, written in `metadata.geojson.Fold` with PASTIS-style fold ids:

| Fold | role | patches |
|--:|---|--:|
| 1 | train | 1,896 |
| 4 | val   | 411 |
| 5 | test  | 443 |

Whole 0.1° (~11 km) spatial blocks are assigned to a single split, so no block is shared
between splits, and each territory (hence every crop class) is represented across
train/val/test. The 70/15/15 target is applied per territory, but because blocks are
indivisible the realised ratios vary, most for Mayotte (10 blocks):

| Territory | train | val | test |
|---|--:|--:|--:|
| Guadeloupe | 526 (72%) | 140 (19%) | 69 (9%) |
| Martinique | 426 (77%) | 66 (12%) | 58 (11%) |
| Guyane | 359 (71%) | 67 (13%) | 80 (16%) |
| Réunion | 524 (64%) | 117 (14%) | 172 (21%) |
| Mayotte | 55 (39%) | 21 (15%) | 64 (46%) |
| Saint-Martin | 6 | 0 | 0 |

Parcels crossing a block edge can still reach two splits: 592 of the 47,256 parcels have
pixels in patches of more than one split (321 train/test, 240 train/val, 30 val/test, 1 in
all three). Each patch only holds the part of the parcel inside it.

### Leave-one-island-out (LOIO)

`splits/loio.json` holds five additional configurations. For each held-out territory,
`test` is all of its patches; `train`/`val` split the remaining territories' patches
(about 88/12), as used in the paper. Unlike the default split, this train/val split is not
block-aligned: train and val share 0.1° blocks, so val scores are optimistic. `test` never
shares a block with `train` or `val`.

| held-out (test) | train | val | test |
|---|--:|--:|--:|
| Guadeloupe | 1,750 | 265 | 735 |
| Martinique | 1,950 | 250 | 550 |
| Guyane | 1,960 | 284 | 506 |
| Réunion | 1,720 | 217 | 813 |
| Mayotte | 2,296 | 314 | 140 |

Per-patch territory is in `metadata.geojson.Island`. Saint-Martin (6 patches) is
labelled but never held out — it stays in `train` for every fold.


## 4. Per-territory composition

Patch counts: Réunion 813 · Guadeloupe 735 · Martinique 550 · Guyane 506 · Mayotte 140 ·
Saint-Martin 6 (a COM, not a DROM, labelled separately in `metadata.geojson.Island`).
Total unique parcels: **47,256** (58,489 parcel–patch instances, since 9,799 parcels cross a
patch border and keep the same instance ID in each patch). Per-class distributions (pixels / windows / parcels) per
island are shown in `../images/{pixels,windows,parcels}_per_class_per_island.png`.
Island crop signatures: sugarcane dominates Réunion & Guadeloupe; banana dominates
Martinique & Mayotte; Guyane is meadow/orchard-heavy.

## 5. Provenance

- **Imagery**: Sentinel-2 L2A (Sen2Cor) and Sentinel-1 RTC retrieved from the Microsoft
  Planetary Computer (Copernicus data). Using Sen2Cor L2A rather than THEIA/MAJA is the one
  intentional departure from PASTIS. Landsat-8 OLI/TIRS Collection 2 Level-1 from the USGS
  `usgs-landsat` bucket on AWS Open Data.
- **Annotations**: the French *Registre Parcellaire Graphique* (RPG 2019), produced by
  IGN, rasterized to the patch grid and mapped to the 8-class taxonomy.

## 6. Loading

```python
import sys; sys.path.append("code")
from dataloader import PLANTEUR_Dataset
ds = PLANTEUR_Dataset(folder="<PLANTEUR_ROOT>", sats=["S2"], folds=[1])      # train, S2
# ds = PLANTEUR_Dataset(folder=..., sats=["S2", "S1"], folds=[5])            # test, S2+S1
# ds = PLANTEUR_Dataset(folder=..., target="instance", folds=[5])           # panoptic
# ds = PLANTEUR_Dataset(folder=..., sats=["LS", "LS_PAN"], folds=[1])       # Landsat
(data, dates), target = ds[0]
```

`sats` takes any of `"S2"`, `"S1"`, `"LS"`, `"LS_PAN"` and `"S2_MONTHLY"`; each returns its own
time series and dates (`LS` and `LS_PAN` share `dates-LS`; `S2_MONTHLY` returns the 10 `DATA_S2`
bands of the 12 monthly mosaics, dated mid-month). `annotations="ANNOTATIONS_BALANCED"` loads
the class-balanced training labels.

For parcel classification, `dataloader_pixelset.PLANTEUR_Dataset_PixelSet` yields PASTIS-R-style
pixel sets: one sample per parcel instance per patch, labelled with its semantic class (void and
Background excluded; labels 1–7 shifted to 0–6 by the default `label_offset=1`), with the four
Pixel-Set Encoder geometric features computed from the instance mask. Pixel sets are extracted
on the fly from the patch arrays, so no parcel-level files are needed.
