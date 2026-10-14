"""Add `orbits-S1` (per-observation Sentinel-1 orbit direction) to metadata.geojson.

DATA_S1 holds every Sentinel-1 RTC acquisition of a patch in date order, ascending and
descending passes together; export_planteur_raw.py averages same-day acquisitions into one
observation. For each `dates-S1` step this records "A" (ascending), "D" (descending) or "AD"
(a same-day average of both), read from the scenes of the source rslearn dataset's
`sentinel1` layer. Scenes are grouped by date exactly as export_planteur_raw.py does, and
patch ids follow its sorted-window order.

Sentinel-1 scene names carry the UTC pass time, and over the DROM the ascending and
descending passes fall at distinct hours, so the direction is read from it; --check-stac N
compares N random scenes with the Planetary Computer STAC `sat:orbit_state`.

Run: python add_orbits_s1.py --src <RSLEARN_DATASET> --group rpg_2019_dec31 \
        --metadata <PLANTEUR_ROOT>/metadata.geojson --out <OUT>/metadata.geojson [--check-stac 60]
"""
from __future__ import annotations
import argparse, collections, glob, json, os, random, urllib.request
from datetime import datetime

ASC_HOURS, DESC_HOURS = {14, 15, 21, 22}, {1, 2, 9, 10}  # UTC pass hours over the DROM
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc/items/{}"


def orbit_of(scene: str) -> str:
    hour = int(scene.split("_")[4][9:11])
    if hour in ASC_HOURS:
        return "A"
    if hour in DESC_HOURS:
        return "D"
    raise ValueError(f"unmapped pass hour {hour} in {scene}")


def scenes_by_date(window: str) -> dict[int, list[str]]:
    """Scene names of the `sentinel1` layer, keyed by yyyymmdd as in export_planteur_raw.py."""
    d = json.load(open(os.path.join(window, "items.json")))
    entry = next((e for e in d if e["layer_name"] == "sentinel1"), None)
    out = collections.defaultdict(list)
    for grp in (entry or {}).get("serialized_item_groups", []):
        if not grp:
            continue
        it = json.loads(grp[0]) if isinstance(grp[0], str) else grp[0]
        tr = it.get("geometry", {}).get("time_range")
        if tr:
            out[int(datetime.fromisoformat(tr[0]).strftime("%Y%m%d"))].append(it["name"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--group", default="rpg_2019_dec31")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-stac", type=int, default=0)
    args = ap.parse_args()

    wins = sorted(d for d in glob.glob(os.path.join(args.src, "windows", args.group, "*")) if os.path.isdir(d))
    meta = json.load(open(args.metadata))
    assert len(wins) == len(meta["features"]), (len(wins), len(meta["features"]))

    counts, scenes = collections.Counter(), []
    for feat in meta["features"]:
        p = feat["properties"]
        by_date = scenes_by_date(wins[p["ID_PATCH"]])
        orbits = {}
        for step, day in p["dates-S1"].items():
            names = by_date.get(int(day))
            assert names, f"patch {p['ID_PATCH']}: dates-S1 {day} has no source scene"
            orbits[step] = "".join(sorted({orbit_of(n) for n in names}))
            counts[orbits[step]] += 1
            scenes.extend(names)
        p["orbits-S1"] = orbits

    if args.check_stac:
        for n in random.Random(0).sample(sorted(set(scenes)), args.check_stac):
            with urllib.request.urlopen(STAC.format(n), timeout=30) as r:
                stac = json.load(r)["properties"]["sat:orbit_state"]
            assert stac[0].upper() == orbit_of(n), (n, stac)

    with open(args.out, "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    print("observations per orbit:", dict(counts))


if __name__ == "__main__":
    main()
