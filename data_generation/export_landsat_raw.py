"""Export materialized Landsat-8 to the PLANTEUR/HF layout.

  DATA_LS/LS_<id>.npy         (T, 10, 64, 64)  uint16   B1-B7,B9,B10,B11 (native)
  DATA_LS_PAN/LSPAN_<id>.npy  (T,  1,128,128)  uint16   B8 panchromatic (native)
  _dates_ls/<id>.json         {step: YYYYMMDD} -> merged into metadata.geojson as dates-LS

ID_PATCH is the index into sorted(windows) -- verified against DATA_S2 and the
existing dates-S2 before this ran.  Same-day acquisitions are mosaicked (mean over
non-zero frames) and frames are chronological, matching export_pastis_raw.py.

Landsat is NOT resampled: the multispectral bands stay on their native 64x64 grid
and B8 on its 128x128 grid, so nothing is invented.  (export_pastis_raw.build_series
hard-skips any frame that is not 128x128, which would have silently produced empty
Landsat arrays -- that guard is why this is a separate exporter.)
"""
import glob, json, os, sys
from datetime import datetime
from multiprocessing import Pool
import numpy as np, rasterio

SRC = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2/data/national_ds"
OUT = "/weka/dfive-default/piperw/pastis2_drom_pastis_format"
MS  = "B1_B2_B3_B4_B5_B6_B7_B9_B10_B11"
PAN = "B8"
WINS = sorted(d for d in glob.glob(f"{SRC}/windows/rpg_2019_dec31/*") if os.path.isdir(d))


def acq_dates(window):
    d = json.load(open(f"{window}/items.json"))
    e = next((x for x in d if x["layer_name"] == "landsat"), None)
    if e is None:
        return []
    out = []
    for gi, grp in enumerate(e["serialized_item_groups"]):
        if not grp:
            continue
        it = json.loads(grp[0]) if isinstance(grp[0], str) else grp[0]
        tr = it.get("geometry", {}).get("time_range")
        if tr:
            out.append((gi, int(datetime.fromisoformat(tr[0]).strftime("%Y%m%d"))))
    return out


def read_group(window, gi, bs, hw):
    sub = "landsat" if gi == 0 else f"landsat.{gi}"
    tif = f"{window}/layers/{sub}/{bs}/geotiff.tif"
    if not os.path.exists(tif):
        return None
    with rasterio.open(tif) as ds:
        a = ds.read()
    return a if a.shape[-2:] == hw else None


def series(window, bs, hw):
    by = {}
    for gi, day in acq_dates(window):
        a = read_group(window, gi, bs, hw)
        if a is not None:
            by.setdefault(day, []).append(a)
    if not by:
        return None, []
    frames, days = [], []
    for day in sorted(by):
        fs = by[day]
        if len(fs) == 1:
            m = fs[0]
        else:
            st = np.stack(fs).astype(np.float32)
            v = st != 0
            m = ((st * v).sum(0) / np.maximum(v.sum(0), 1)).astype(fs[0].dtype)
        frames.append(m)
        days.append(day)
    return np.stack(frames, 0), days


def one(i):
    w = WINS[i]
    ms, d1 = series(w, MS, (64, 64))
    pan, d2 = series(w, PAN, (128, 128))
    if ms is None or pan is None:
        return (i, "EMPTY", 0, 0)
    if d1 != d2:
        return (i, f"DATE-MISMATCH ms={len(d1)} pan={len(d2)}", len(d1), len(d2))
    np.save(f"{OUT}/DATA_LS/LS_{i}.npy", ms.astype(np.uint16))
    np.save(f"{OUT}/DATA_LS_PAN/LSPAN_{i}.npy", pan.astype(np.uint16))
    json.dump({str(k): int(v) for k, v in enumerate(d1)},
              open(f"{OUT}/_dates_ls/{i}.json", "w"))
    return (i, "ok", len(d1), ms.shape[1])


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    for sub in ("DATA_LS", "DATA_LS_PAN", "_dates_ls"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    idx = list(range(lo, min(hi, len(WINS))))
    bad = 0
    with Pool(12) as p:
        for n, (i, st, t, c) in enumerate(p.imap_unordered(one, idx, chunksize=4), 1):
            if st != "ok":
                bad += 1
                print(f"  BAD id={i}: {st}", flush=True)
            if n % 100 == 0:
                print(f"  {n}/{len(idx)} done, {bad} bad", flush=True)
    print(f"EXPORT_RANGE_DONE {lo}:{hi} bad={bad}", flush=True)
