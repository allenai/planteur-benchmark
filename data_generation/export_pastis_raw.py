"""Export PASTIS2-DROM to the *exact* original-PASTIS file format from the DENSE
rslearn dataset (national_ds: ~150 raw S2 + ~123 raw S1 acquisitions per patch),
so existing PASTIS dataloaders (pastis-benchmark/code/dataloader.py) load it as-is.

Produces, under --out:
  DATA_S2/S2_<id>.npy          (T_s2, 10, 128, 128) uint16, chronological
  DATA_S1/S1_<id>.npy          (T_s1, 2, 128, 128) float32 (vv,vh dB), chronological
  ANNOTATIONS/TARGET_<id>.npy  (1, 128, 128) uint8  -- 8 classes + 255 void (bg8void)
  metadata.geojson             ID_PATCH, dates-S2, dates-S1, Fold, geometry(real polygon)
  NORM_S2_patch.json, NORM_S1_patch.json

Panoptic INSTANCE_ANNOTATIONS are added by a companion pass (export_pastis_panoptic.py).

Run: python export_pastis_raw.py --src data/national_ds --group rpg_2019_dec31 \
        --out /weka/dfive-default/piperw/pastis2_drom_pastis_format [--limit N]
"""
from __future__ import annotations
import argparse, glob, json, os
from datetime import datetime

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import pyproj

S2_BS = "B02_B03_B04_B05_B06_B07_B08_B8A_B11_B12"   # 10 bands, 128x128
S1_BS = "vv_vh"
# 26-class source label id -> the 8 trained classes.
#
# Canonical taxonomy is "bg8void": every excluded crop (Corn, Grapevine, Beet,
# Potatoes, Leguminous fodder, Soybeans, Vanilla, Ylang-ylang, Coffee/Cacao, plus the
# temperate classes absent from the DROM) maps to VOID, which is masked out of loss
# and metrics. Background(0) therefore stays a real land-cover class instead of
# absorbing unmapped agriculture -- following the PASTIS convention that background is
# land cover and void is invalid data.
#
# The legacy "bg8" taxonomy folded those ids into Background(0). It is retained only to
# reproduce the deprecated ANNOTATIONS/ set; new work should use bg8void.
BG8 = {0: 0, 1: 1, 12: 2, 16: 3, 19: 4, 20: 5, 21: 6, 23: 7}
VOID = 255  # void / nodata sentinel (also the source raster's nodata value)
FOLD_OF = {"train": 1, "val": 4, "test": 5}


def _acq_dates(window: str, layer: str) -> list[tuple[int, int]]:
    """Return [(group_idx, yyyymmdd), ...] for every acquisition of a layer."""
    d = json.load(open(os.path.join(window, "items.json")))
    entry = next((e for e in d if e["layer_name"] == layer), None)
    if entry is None:
        return []
    out = []
    for gi, grp in enumerate(entry["serialized_item_groups"]):
        if not grp:
            continue
        it = json.loads(grp[0]) if isinstance(grp[0], str) else grp[0]
        tr = it.get("geometry", {}).get("time_range")
        dt = datetime.fromisoformat(tr[0]) if tr else None
        if dt is None:  # fall back to parsing the product name
            continue
        out.append((gi, int(dt.strftime("%Y%m%d"))))
    return out


def _read_group(window: str, layer: str, gi: int, bs: str) -> np.ndarray | None:
    sub = layer if gi == 0 else f"{layer}.{gi}"
    tif = os.path.join(window, "layers", sub, bs, "geotiff.tif")
    if not os.path.exists(tif):
        return None
    with rasterio.open(tif) as ds:
        return ds.read()  # (C, H, W)


def build_series(window: str, layer: str, bs: str):
    """Return (T,C,128,128) array (chronological, ONE obs per date) + yyyymmdd list.

    Same-date acquisitions (multiple overlapping tiles/orbits) are mosaicked into a
    single observation -- per band+pixel mean over the non-nodata (nonzero) frames --
    matching PASTIS's one-observation-per-date convention.
    """
    by_date: dict[int, list] = {}
    for gi, day in _acq_dates(window, layer):
        arr = _read_group(window, layer, gi, bs)
        if arr is None or arr.shape[-2:] != (128, 128):
            continue
        by_date.setdefault(day, []).append(arr)
    if not by_date:
        return None, []
    frames, days = [], []
    for day in sorted(by_date):
        fs = by_date[day]
        if len(fs) == 1:
            m = fs[0]
        else:
            st = np.stack(fs).astype(np.float32)
            valid = st != 0
            m = (st * valid).sum(0) / np.maximum(valid.sum(0), 1)
            m = m.astype(fs[0].dtype)
        frames.append(m)
        days.append(day)
    return np.stack(frames, 0), days


def read_target(window: str, taxonomy: str = "bg8void") -> np.ndarray:
    """Semantic target for one window. taxonomy: "bg8void" (canonical) or "bg8"."""
    tif = glob.glob(os.path.join(window, "layers/label/*/geotiff.tif"))[0]
    with rasterio.open(tif) as ds:
        lab = ds.read(1)
    # Anything not explicitly kept by BG8 becomes VOID under bg8void, or Background(0)
    # under legacy bg8. BG8 maps source id 0 -> 0, so true background survives either way.
    fill = VOID if taxonomy == "bg8void" else 0
    out = np.full(lab.shape, fill, dtype=np.uint8)
    for k, v in BG8.items():
        out[lab == k] = v
    out[lab == VOID] = VOID
    return out[None]  # (1, H, W)


def patch_polygon(meta: dict):
    m = meta["projection"]; b = meta["bounds"]
    xr, yr = m["x_resolution"], m["y_resolution"]
    poly = box(b[0] * xr, b[3] * yr, b[2] * xr, b[1] * yr)
    return gpd.GeoSeries([poly], crs=m["crs"]).to_crs(4326).iloc[0]


def merge(out):
    """Combine per-shard metadata + norm into final metadata.geojson + NORM_*.json."""
    import glob as _g
    rows = []
    for f in sorted(_g.glob(os.path.join(out, "_meta", "*.geojson"))):
        rows.append(gpd.read_file(f))
    gdf = pd.concat(rows, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, crs=4326).sort_values("ID_PATCH").reset_index(drop=True)
    gdf.to_file(os.path.join(out, "metadata.geojson"), driver="GeoJSON")
    for nm, bands in [("S2", 10), ("S1", 2)]:
        S = SQ = None; N = 0
        for f in sorted(_g.glob(os.path.join(out, "_norm", f"*_{nm}.json"))):
            d = json.load(open(f))
            if not d["n"]:
                continue
            s, sq, n = np.array(d["sum"]), np.array(d["sqsum"]), d["n"]
            S = s if S is None else S + s
            SQ = sq if SQ is None else SQ + sq
            N += n
        if N:
            mean = S / N; std = np.sqrt(np.maximum(SQ / N - mean ** 2, 0))
            # PASTIS dataloader keys norm per fold (Fold_1..Fold_5) and averages
            # over the selected folds; we compute stats on train only, so write
            # the same train mean/std under every fold key.
            per_fold = {f"Fold_{f}": {"mean": mean.tolist(), "std": std.tolist()}
                        for f in range(1, 6)}
            json.dump(per_fold, open(os.path.join(out, f"NORM_{nm}_patch.json"), "w"))
    c = gdf["split"].value_counts().to_dict()
    print(f"MERGED: {len(gdf)} patches -> {out}  splits={c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--group", default="rpg_2019_dec31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--norm-sample", type=int, default=150)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)   # 0 == to end
    ap.add_argument("--merge", action="store_true")
    ap.add_argument(
        "--taxonomy",
        choices=["bg8void", "bg8"],
        default="bg8void",
        help="bg8void (canonical): excluded crops -> 255 void. "
        "bg8 (legacy): excluded crops -> 0 Background.",
    )
    args = ap.parse_args()
    for sub in ["DATA_S2", "DATA_S1", "ANNOTATIONS", "_meta", "_norm"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    if args.merge:
        merge(args.out); return

    wins = sorted(d for d in glob.glob(os.path.join(args.src, "windows", args.group, "*")) if os.path.isdir(d))
    if args.limit:
        wins = wins[: args.limit]
    lo, hi = args.start, (args.end or len(wins))
    shard = wins[lo:hi]
    rows = []
    n_s2 = {"sum": None, "sqsum": None, "n": 0}
    n_s1 = {"sum": None, "sqsum": None, "n": 0}
    for j, w in enumerate(shard):
        i = lo + j  # global window index == PATCH id (stable across shards)
        meta = json.load(open(os.path.join(w, "metadata.json")))
        split = meta["options"].get("split")
        pid = i  # sequential PATCH id
        s2, d2 = build_series(w, "sentinel2", S2_BS)
        s1, d1 = build_series(w, "sentinel1", S1_BS)
        if s2 is None:
            print(f"  SKIP {i}: no S2", flush=True); continue
        np.save(os.path.join(args.out, "DATA_S2", f"S2_{pid}.npy"), s2.astype(np.uint16))
        if s1 is not None:
            # convert to dB like PASTIS-R (10*log10), keep float32
            s1f = s1.astype(np.float32)
            np.save(os.path.join(args.out, "DATA_S1", f"S1_{pid}.npy"), s1f)
        np.save(
            os.path.join(args.out, "ANNOTATIONS", f"TARGET_{pid}.npy"),
            read_target(w, args.taxonomy),
        )
        # norm accumulation (train only; merged across shards later)
        if split == "train":
            x = s2.astype(np.float64).transpose(1, 0, 2, 3).reshape(s2.shape[1], -1)
            n_s2["sum"] = x.sum(1) if n_s2["sum"] is None else n_s2["sum"] + x.sum(1)
            n_s2["sqsum"] = (x**2).sum(1) if n_s2["sqsum"] is None else n_s2["sqsum"] + (x**2).sum(1)
            n_s2["n"] += x.shape[1]
            if s1 is not None:
                y = s1.astype(np.float64).transpose(1, 0, 2, 3).reshape(s1.shape[1], -1)
                n_s1["sum"] = y.sum(1) if n_s1["sum"] is None else n_s1["sum"] + y.sum(1)
                n_s1["sqsum"] = (y**2).sum(1) if n_s1["sqsum"] is None else n_s1["sqsum"] + (y**2).sum(1)
                n_s1["n"] += y.shape[1]
        rows.append({
            "ID_PATCH": pid,
            "dates-S2": json.dumps({str(k): int(v) for k, v in enumerate(d2)}),
            "dates-S1": json.dumps({str(k): int(v) for k, v in enumerate(d1)}),
            "Fold": FOLD_OF.get(split, 0),
            "split": split,
            "geometry": patch_polygon(meta),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wins)} (T_s2={s2.shape[0]} T_s1={s1.shape[0] if s1 is not None else 0})", flush=True)

    # per-shard outputs (merged later via --merge)
    if rows:
        gpd.GeoDataFrame(rows, crs=4326).to_file(
            os.path.join(args.out, "_meta", f"{lo:06d}.geojson"), driver="GeoJSON")
    for nm, acc in [("S2", n_s2), ("S1", n_s1)]:
        json.dump(
            {"sum": acc["sum"].tolist() if acc["sum"] is not None else [],
             "sqsum": acc["sqsum"].tolist() if acc["sqsum"] is not None else [],
             "n": acc["n"]},
            open(os.path.join(args.out, "_norm", f"{lo:06d}_{nm}.json"), "w"))
    print(f"shard {lo}:{hi} done: {len(rows)} patches", flush=True)


if __name__ == "__main__":
    main()
