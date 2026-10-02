"""Rebuild metadata.geojson with DEDUPED dates that exactly match the (already
deduped) DATA_S2/DATA_S1 .npy arrays, without re-reading any geotiffs.

The per-shard _meta/*.geojson were written by an older code path and still list
the raw (pre-mosaic) acquisition dates (e.g. 150 S2 entries with only 81 unique),
which breaks pastis-benchmark's dataloader (duplicate date-table labels). The
DATA_*.npy were re-exported with same-date mosaicking, so the correct date list
per patch is simply the sorted unique acquisition dates -- provided every raw
group had a valid 128x128 tile (no skips). We verify that invariant by comparing
the unique-date count to the actual array frame count; any mismatch is reported
so it can be re-exported the slow way.

Run: python rebuild_meta.py --src data/national_ds --group rpg_2019_dec31 \
        --out <PLANTEUR_ROOT>
"""
from __future__ import annotations
import argparse, glob, json, os

import numpy as np
import geopandas as gpd
from export_pastis_raw import _acq_dates, patch_polygon, FOLD_OF


def dedup_days(window: str, layer: str) -> list[int]:
    days = [day for _, day in _acq_dates(window, layer)]
    return sorted(set(days))


def npy_frames(out: str, sat: str, pid: int) -> int | None:
    p = os.path.join(out, f"DATA_{sat}", f"{sat}_{pid}.npy")
    if not os.path.exists(p):
        return None
    return int(np.load(p, mmap_mode="r").shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--group", default="rpg_2019_dec31")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wins = sorted(d for d in glob.glob(os.path.join(args.src, "windows", args.group, "*")) if os.path.isdir(d))
    rows, mism = [], []
    for pid, w in enumerate(wins):
        if npy_frames(args.out, "S2", pid) is None:
            continue  # patch was skipped during export (no S2)
        meta = json.load(open(os.path.join(w, "metadata.json")))
        split = meta["options"].get("split")
        d2 = dedup_days(w, "sentinel2")
        d1 = dedup_days(w, "sentinel1")
        t2, t1 = npy_frames(args.out, "S2", pid), npy_frames(args.out, "S1", pid)
        if t2 != len(d2) or (t1 is not None and t1 != len(d1)):
            mism.append((pid, len(d2), t2, len(d1), t1))
        rows.append({
            "ID_PATCH": pid,
            "dates-S2": json.dumps({str(k): int(v) for k, v in enumerate(d2)}),
            "dates-S1": json.dumps({str(k): int(v) for k, v in enumerate(d1)}),
            "Fold": FOLD_OF.get(split, 0),
            "split": split,
            "geometry": patch_polygon(meta),
        })
        if (pid + 1) % 250 == 0:
            print(f"  {pid+1}/{len(wins)} rows={len(rows)} mismatches={len(mism)}", flush=True)

    gdf = gpd.GeoDataFrame(rows, crs=4326).sort_values("ID_PATCH").reset_index(drop=True)
    dst = os.path.join(args.out, "metadata.geojson")
    tmp = dst + ".tmp"
    gdf.to_file(tmp, driver="GeoJSON")
    os.replace(tmp, dst)
    c = gdf["split"].value_counts().to_dict()
    print(f"REBUILT: {len(gdf)} patches -> {dst}  splits={c}")
    print(f"MISMATCHES (pid,nd2,t2,nd1,t1): {len(mism)}")
    for m in mism[:40]:
        print("  ", m)


if __name__ == "__main__":
    main()
