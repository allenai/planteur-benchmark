"""Per-island class distributions of the released dataset -> images/*.png.

  pixels_per_class_per_island.png   # pixels per class (void excluded)
  windows_per_class_per_island.png  # patches containing each class
  parcels_per_class_per_island.png  # unique parcels (instance IDs) per crop class

Run: python plot_class_distributions.py <PLANTEUR_ROOT> [--out ../images]
"""
import argparse, json, os
from collections import defaultdict
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ISLANDS = ["Guadeloupe", "Martinique", "Réunion", "Guyane", "Mayotte", "Saint-Martin"]
CLASSES = ["Background", "Meadow", "Fruits, vegetables, flowers", "Orchard", "Sugarcane", "Banana",
           "Pineapple", "Tropical tuber"]
COLORS = ["#dceaf5", "#3ccf6e", "#ffb02e", "#e8433a", "#f5e54a", "#3da5f4", "#b072e8", "#ff7fc8"]


def patch_stats(args):
    root, pid = args
    t = np.load(f"{root}/ANNOTATIONS/TARGET_{pid}.npy")[0]
    ins = np.load(f"{root}/INSTANCE_ANNOTATIONS/INSTANCES_{pid}.npy")
    pixels = np.bincount(t[t != 255].ravel(), minlength=8)
    parcels = {}
    for i in np.unique(ins):
        if i:
            parcels[int(i)] = int(np.bincount(t[ins == i]).argmax())
    return pid, pixels, parcels


def plot(counts, n_patches, ylabel, title, classes, path):
    fig, axes = plt.subplots(2, 3, figsize=(20, 9.2))
    for ax, isl in zip(axes.ravel(), ISLANDS):
        v = [counts[isl][c] for c in classes]
        ax.bar(range(len(classes)), v, color=[COLORS[c] for c in classes],
               edgecolor="#444444", linewidth=0.6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([CLASSES[c] for c in classes], rotation=40, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{isl}  (n={n_patches[isl]} patches)", fontweight="bold")
    fig.suptitle(f"PLANTEUR — {title}, per island (log scale)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "images"))
    args = ap.parse_args()
    meta = json.load(open(f"{args.root}/metadata.geojson"))
    island = {f["properties"]["ID_PATCH"]: f["properties"]["Island"] for f in meta["features"]}

    with Pool(32) as p:
        stats = p.map(patch_stats, [(args.root, pid) for pid in sorted(island)], chunksize=8)

    n_patches = defaultdict(int)
    pixels = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    windows = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    parcels = defaultdict(dict)
    for pid, px, pc in stats:
        isl = island[pid]
        n_patches[isl] += 1
        pixels[isl] += px
        windows[isl] += px > 0
        parcels[isl].update(pc)
    parcel_counts = {isl: np.bincount(list(d.values()) or [0], minlength=8) if d else np.zeros(8)
                     for isl, d in parcels.items()}
    for isl in ISLANDS:
        parcel_counts.setdefault(isl, np.zeros(8))

    plot(pixels, n_patches, "# pixels", "# pixels per class", range(8),
         os.path.join(args.out, "pixels_per_class_per_island.png"))
    plot(windows, n_patches, "# patches", "# patches containing each class", range(8),
         os.path.join(args.out, "windows_per_class_per_island.png"))
    plot(parcel_counts, n_patches, "# parcels", "# parcels per crop class", range(1, 8),
         os.path.join(args.out, "parcels_per_class_per_island.png"))


if __name__ == "__main__":
    main()
