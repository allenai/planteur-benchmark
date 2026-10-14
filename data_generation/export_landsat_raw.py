"""Export materialized Landsat-8 to the PLANTEUR/HF layout.

  DATA_LS/LS_<id>.npy         (T, 10, 64, 64)  uint16   B1-B7,B9,B10,B11 (20 m)
  DATA_LS_PAN/LSPAN_<id>.npy  (T,  1,128,128)  uint16   B8 panchromatic (10 m)
  _dates_ls/<id>.json         {step: YYYYMMDD} -> merged into metadata.geojson as dates-LS
  NORM_LS_patch.json, NORM_LS_PAN_patch.json  (run with `norm` after dates-LS is merged)

ID_PATCH is the index into sorted(windows), as in export_planteur_raw.py.  Same-day
acquisitions are mosaicked (mean over non-zero frames) and frames are chronological,
matching export_planteur_raw.py.

Resampling happens when rslearn materializes the data, not here: the multispectral
bands on a 20 m 64x64 grid (zoom_offset -1, from native 30 m) and B8 on the 10 m
128x128 grid (from native 15 m).  (export_planteur_raw.build_series
hard-skips any frame that is not 128x128, which would have silently produced empty
Landsat arrays -- that guard is why this is a separate exporter.)
"""
import glob, json, os, sys
from datetime import datetime
from multiprocessing import Pool
import numpy as np, rasterio

# Set before running: the rslearn dataset with materialized Landsat, and the dataset root.
SRC = "<RSLEARN_DATASET>"
OUT = "<PLANTEUR_ROOT>"
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


def _norm_stats(pid):
    """Per-band sum / sum of squares / count over valid (non-zero) pixels of one patch."""
    out = {}
    for nm, path in (("LS", f"{OUT}/DATA_LS/LS_{pid}.npy"), ("LS_PAN", f"{OUT}/DATA_LS_PAN/LSPAN_{pid}.npy")):
        a = np.load(path).astype(np.float64)
        v = a.transpose(1, 0, 2, 3).reshape(a.shape[1], -1)
        ok = v != 0
        out[nm] = ((v * ok).sum(1), (v**2 * ok).sum(1), ok.sum(1))
    return out


def write_norm():
    """NORM_LS_patch.json / NORM_LS_PAN_patch.json: train (Fold 1) per-band mean/std over
    valid pixels (0 = nodata, which occurs in the thermal bands B10/B11), written under
    every Fold_<k> key as the loader expects (same convention as NORM_S2/S1)."""
    meta = json.load(open(f"{OUT}/metadata.geojson"))
    train = [f["properties"]["ID_PATCH"] for f in meta["features"] if f["properties"]["Fold"] == 1]
    with Pool(32) as p:
        stats = p.map(_norm_stats, train, chunksize=8)
    for nm in ("LS", "LS_PAN"):
        S, SQ, N = (sum(st[nm][k] for st in stats) for k in range(3))
        mean = S / N
        std = np.sqrt(np.maximum(SQ / N - mean**2, 0))
        json.dump({f"Fold_{f}": {"mean": mean.tolist(), "std": std.tolist()} for f in range(1, 6)},
                  open(f"{OUT}/NORM_{nm}_patch.json", "w"))
        print(f"NORM_{nm}_patch.json: {len(train)} train patches, mean={mean.round(1).tolist()}")


if __name__ == "__main__":
    if sys.argv[1:] == ["norm"]:  # python export_landsat_raw.py norm
        write_norm()
        sys.exit()
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
