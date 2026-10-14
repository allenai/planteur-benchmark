"""Add PASTIS INSTANCE_ANNOTATIONS (panoptic) to the raw PASTIS-format export.

For each exported patch, rasterize the RPG parcels onto the patch grid to get
per-pixel instance IDs, keep only the parcels labelled with one of the 7 crop
classes, then compute the centerness HEATMAP and ZONES exactly like
pastis-benchmark/code/centerness_ground_truth.py. Writes, under <out>:

  INSTANCE_ANNOTATIONS/INSTANCES_<id>.npy  (128,128) int32   -- per-pixel parcel/instance id (0=Background or void)
  INSTANCE_ANNOTATIONS/HEATMAP_<id>.npy    (128,128) float32 -- object centerness
  INSTANCE_ANNOTATIONS/ZONES_<id>.npy      (128,128) int32   -- pixel->nearest-center instance

Uses the imagery pass's metadata.geojson (ID_PATCH == sorted-window index) so the
patch ids line up with DATA_S2/ANNOTATIONS. Run AFTER export_planteur_raw.py:
  python export_planteur_panoptic.py --src <RSLEARN_DATASET> --group rpg_2019_dec31 \
      --out <PLANTEUR_ROOT> --rpg <RPG_DIR>
"""
from __future__ import annotations
import argparse, glob, json, os

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import box


def bbox(img):
    rows = np.any(img, axis=1); cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
    if rmin > 0: rmin -= 1
    if cmin > 0: cmin -= 1
    rmax = rmax + 1 if rmax < img.shape[1] - 1 else rmax
    cmax = cmax + 1 if cmax < img.shape[1] - 1 else cmax
    return rmin, rmax, cmin, cmax


def splat(shape, center, dims, r=3, alpha=0.7):
    h, w = shape; hc, wc = center; height, width = dims
    hcords = np.stack([np.arange(h) for _ in range(w)], axis=1)
    wcords = np.stack([np.arange(w) for _ in range(h)], axis=0)
    sigma_h = (1 - alpha) / (1 + alpha) * max(height, 1) / r
    sigma_w = (1 - alpha) / (1 + alpha) * max(width, 1) / r
    out = (hcords - hc) ** 2 / (2 * sigma_h ** 2) + (wcords - wc) ** 2 / (2 * sigma_w ** 2)
    return np.exp(-out)


def rasterize_parcels(window_meta, parcels_4326, sindex):
    m = window_meta["projection"]; b = window_meta["bounds"]
    xr, yr = m["x_resolution"], m["y_resolution"]
    minx, miny, maxx, maxy = b
    transform = Affine(xr, 0.0, minx * xr, 0.0, yr, miny * yr)
    bb = box(minx * xr, maxy * yr, maxx * xr, miny * yr)
    bb4 = gpd.GeoSeries([bb], crs=m["crs"]).to_crs(4326).iloc[0]
    idx = list(sindex.query(bb4, predicate="intersects"))
    if not idx:
        return np.zeros((maxy - miny, maxx - minx), dtype=np.int32)
    sub = parcels_4326.iloc[idx].to_crs(m["crs"])
    shapes = [(geom, int(gid)) for geom, gid in zip(sub.geometry, sub["pid_uid"])]
    return rasterio.features.rasterize(
        shapes, out_shape=(maxy - miny, maxx - minx), transform=transform,
        fill=0, dtype="int32",
    )


def crop_mask(semantic):
    return (semantic >= 1) & (semantic <= 7)


def compute_heatmap_zones(parcel_ids, semantic):
    instance = parcel_ids * crop_mask(semantic)
    paid = np.unique(instance)
    mapp = {int(pid): k for k, pid in enumerate(paid)}  # 0 (bg) -> code 0
    inst_codes = np.vectorize(lambda x: mapp[int(x)])(instance)
    splats, codes = [], []
    for pid, code in mapp.items():
        if code == 0:
            continue
        mask = inst_codes == code
        rmin, rmax, cmin, cmax = bbox(mask)
        cr = rmin + (rmax - rmin) // 2; cc = cmin + (cmax - cmin) // 2
        splats.append(splat(mask.shape, (cr, cc), (rmax - rmin, cmax - cmin)))
        codes.append(pid)  # store the global ParcelID for the zone value
    if not splats:
        h, w = semantic.shape
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.int32)
    S = np.stack(splats)
    heatmap = S.max(axis=0).astype(np.float32)
    zi = np.argmax(S, axis=0)
    zones = np.vectorize(lambda x: codes[x])(zi).astype(np.int32)
    zones = zones * crop_mask(semantic)
    return heatmap, zones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--group", default="rpg_2019_dec31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rpg", required=True, help="directory of RPG 2019 .gpkg files")
    args = ap.parse_args()
    ia = os.path.join(args.out, "INSTANCE_ANNOTATIONS")
    os.makedirs(ia, exist_ok=True)

    wins = sorted(d for d in glob.glob(os.path.join(args.src, "windows", args.group, "*")) if os.path.isdir(d))
    meta = gpd.read_file(os.path.join(args.out, "metadata.geojson"))
    ids = sorted(int(x) for x in meta["ID_PATCH"])

    gdfs = []
    for g in sorted(glob.glob(os.path.join(args.rpg, "*.gpkg"))):
        d = gpd.read_file(g); col = "class_id" if "class_id" in d.columns else d.columns[0]
        gdfs.append(d[[col, "geometry"]].rename(columns={col: "class_id"}).to_crs(4326))
    P = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=4326)
    P["pid_uid"] = np.arange(1, len(P) + 1)  # unique instance id, 0 reserved for bg
    sindex = P.sindex

    for n, pid in enumerate(ids):
        w = wins[pid]
        wmeta = json.load(open(os.path.join(w, "metadata.json")))
        pids = rasterize_parcels(wmeta, P, sindex)
        semantic = np.load(os.path.join(args.out, "ANNOTATIONS", f"TARGET_{pid}.npy"))[0]
        heat, zones = compute_heatmap_zones(pids, semantic)
        instances = (pids * crop_mask(semantic)).astype(np.int32)
        np.save(os.path.join(ia, f"INSTANCES_{pid}.npy"), instances)
        np.save(os.path.join(ia, f"HEATMAP_{pid}.npy"), heat)
        np.save(os.path.join(ia, f"ZONES_{pid}.npy"), zones)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(ids)}", flush=True)
    print(f"done: panoptic annotations for {len(ids)} patches -> {ia}")


if __name__ == "__main__":
    main()
