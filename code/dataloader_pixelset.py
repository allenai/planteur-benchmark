"""
Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
License MIT

Adapted for PLANTEUR: pixel sets are extracted on the fly from the patch arrays and
INSTANCE_ANNOTATIONS, so no separate parcel-level export is needed.
"""

import json
import os

from datetime import datetime

import geopandas as gpd
import numpy as np
import torch
import torch.utils.data as data
from scipy.interpolate import interp1d
from torch.nn import functional as F

from dataloader import SATS

VOID = 255
BACKGROUND = 0


class PLANTEUR_Dataset_PixelSet(data.Dataset):
    def __init__(
            self,
            folder,
            norm=True,
            cache=False,
            sats=["S2"],
            reference_date="2018-09-01",
            n_pixel=32,
            geom_features=True,
            folds=None,
            ignore_label=None,
            label_offset=1,
            class_mapping=None,
            interpolate=False,
            drop_temp_s2=0,
            drop_temp_s1=0,
    ):
        """
        Pytorch Dataset class to load PLANTEUR in pixel-set format for parcel-based
        classification.

        One sample is one parcel within one patch: every instance of
        INSTANCE_ANNOTATIONS/INSTANCES_<id>.npy is a sample, its pixels are gathered
        from the patch time series, and its label is the parcel's semantic class from
        ANNOTATIONS/TARGET_<id>.npy. Void parcels (class 255) and Background are not
        samples. A parcel crossing a patch border appears once per patch it touches,
        with only the pixels inside that patch.

        The Dataset yields a data dictionary for each sample.
        For example, assuming the dataset was set to load
        Sentinel-2 observations (sats=['S2']),
        it will have the following keys:
            'S2': A tuple containing the Sentinel-2 time series of pixel-sets and
                the mask of true and repeated pixels. Formally, this key contains
                a tuple (X, M) with X a tensor of shape T_s2 x C_s2 x n_pixel and
                M a tensor of shape n_pixel. (T: number of available observations for S2,
                and C: number of spectral channels of S2)
            'dates-S2': A tensor containing the T dates of observation of the loaded
                S2 sequence. The dates are expressed as number of days since the
                reference date.
            'geomfeat' : A vector containing the parcel's 4 geometric features used
                by the Pixel-Set Encoder: area (pixels), perimeter (pixel edges),
                perimeter / area, and area / bounding-box area. Computed on the part
                of the parcel inside the patch.
            'label' : Crop type label of the parcel.

        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using the pre-computed
                channel-wise means and standard deviations of NORM_<sat>_patch.json.
            cache (bool): If True, the loaded samples stay in RAM, default False.
                Without it every sample reads its whole patch, so for training either
                enable it or iterate patch by patch (the most recently read patch is
                kept, and samples are ordered by patch).
            sats (list): defines the satellites to use, any of 'S2', 'S1', 'LS' and
                'LS_PAN' (see dataloader.PLANTEUR_Dataset). For 'LS' (64x64 grid) a
                Landsat pixel belongs to the parcel when at least 2 of the 4 10 m
                pixels it covers do.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            n_pixel (int): number of pixels randomly sampled from each parcel (default 32).,
                geom_features (bool): If False no geometric descriptors of the parcel's shape
                are used in the Pixel-Set Encoder (default True).
            folds (list, optional): List of ints specifying which folds to load:
                1 = train, 4 = val, 5 = test. By default (when None is specified) all
                folds are loaded.
            ignore_label (int): If not None, the parcels annotated with this label are discarded from
                the dataset.
            label_offset (int): If not None this offset is substracted from the labels contained in the dataset
                (default to 1: parcel classes are 1..7 since Background is not a parcel class,
                so labels become 0..6).
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 8-class nomenclature and another class grouping, optional.
                Applied before label_offset.
            interpolate (bool): if True, the non-S2 observations are temporally interpolated to the
                dates of Sentinel-2 acquisitions (default False).
            drop_temp_s2 (float): probability of temporal dropout for Sentinel-2 time series (default 0.0),
            drop_temp_s1: probability of temporal dropout for Sentinel-1 time series (default 0.0),
        """
        unknown = [s for s in sats if s not in SATS]
        assert not unknown, "Unknown satellite {} (available: {})".format(unknown, list(SATS))

        super(PLANTEUR_Dataset_PixelSet, self).__init__()
        self.folder = folder
        self.sats = sats
        self.norm = norm
        self.n_pixel = n_pixel
        self.geom_features = geom_features
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.label_offset = label_offset
        self.cache = cache
        self.memory = {}
        self.class_mapping = class_mapping
        self.interpolate = interpolate
        self.drop_temp = {"S2": drop_temp_s2, "S1": drop_temp_s1}
        self._last_patch = (None, None)

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        if folds is not None:
            self.meta_patch = self.meta_patch[self.meta_patch["Fold"].isin(folds)]

        self.dates = {
            s: {
                pid: prepare_dates(d if not isinstance(d, str) else json.loads(d),
                                   self.reference_date)
                for pid, d in self.meta_patch[SATS[s][2]].items()
            }
            for s in sats
        }

        # Build the parcel index from the instance and semantic annotations
        print("Indexing parcels . . .")
        self.samples, self.labels, geomfeat = [], [], []
        for pid in self.meta_patch.index:
            instances = np.load(os.path.join(
                folder, "INSTANCE_ANNOTATIONS", "INSTANCES_{}.npy".format(pid)))
            semantic = np.load(os.path.join(
                folder, "ANNOTATIONS", "TARGET_{}.npy".format(pid)))[0]
            for id_parcel in np.unique(instances):
                if id_parcel == 0:
                    continue
                mask = instances == id_parcel
                classes, counts = np.unique(semantic[mask], return_counts=True)
                label = int(classes[counts.argmax()])
                if label in (VOID, BACKGROUND) or label == ignore_label:
                    continue
                self.samples.append((int(pid), int(id_parcel)))
                self.labels.append(label)
                geomfeat.append(geometric_features(mask))
        print("Done.")

        self.len = len(self.samples)
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                        os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = {s: None for s in sats}

        # Geometric features and their normalisation values for pixel-set
        if geom_features:
            self.geomfeat = np.array(geomfeat, dtype=np.float64).reshape(-1, 4)
            self.geom_m = self.geomfeat.mean(axis=0)
            self.geom_s = self.geomfeat.std(axis=0)

        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.dates[sat][id_patch]

    def _load_patch(self, id_patch):
        """Patch arrays and instance map; the most recent patch is kept."""
        if self._last_patch[0] != id_patch:
            arrays = {
                s: np.load(os.path.join(
                    self.folder, SATS[s][0], "{}_{}.npy".format(SATS[s][1], id_patch)))
                for s in self.sats
            }
            instances = np.load(os.path.join(
                self.folder, "INSTANCE_ANNOTATIONS", "INSTANCES_{}.npy".format(id_patch)))
            self._last_patch = (id_patch, (arrays, instances))
        return self._last_patch[1]

    def __getitem__(self, item):
        id_patch, id_parcel = self.samples[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            arrays, instances = self._load_patch(id_patch)
            mask = instances == id_parcel
            data = {}
            for s, a in arrays.items():
                m = mask
                if a.shape[-2:] != mask.shape:  # LS: 64x64, 2x coarser grid
                    f = mask.shape[0] // a.shape[-2]
                    m = mask.reshape(a.shape[-2], f, a.shape[-1], f).sum(axis=(1, 3)) >= 2
                data[s] = torch.from_numpy(a[:, :, m].astype(np.float32))  # T x C x N
            data = {
                s: repeat_pixel_and_norm(a, self.n_pixel, norm=self.norm[s])
                for s, a in data.items()
            }

            if self.cache:
                self.memory[item] = data

        else:
            data = self.memory[item]

        data = {
            sat: (sample_pixels(tensor, self.n_pixel), mask)
            for sat, (tensor, mask) in data.items()
        }  # Random pixel sampling

        # Retrieve date sequences
        dates = {
            "dates-{}".format(s): torch.from_numpy(self.get_dates(id_patch, s))
            for s in self.sats
        }

        # Temporal dropout
        for sat in self.sats:
            p = self.drop_temp.get(sat, 0)
            if p > 0:
                keep = np.random.rand(data[sat][0].shape[0]) > p
                data[sat] = (data[sat][0][keep], data[sat][1])
                dates["dates-{}".format(sat)] = dates["dates-{}".format(sat)][keep]

        data.update(dates)

        if self.interpolate and len(self.sats) > 1 and "S2" in self.sats:
            for sat in self.sats:
                if sat == "S2":
                    continue
                interpolation = interp1d(
                    data["dates-{}".format(sat)].numpy(),
                    data[sat][0].numpy(),
                    axis=0,
                    fill_value="extrapolate",
                )
                data[sat] = (
                    torch.from_numpy(interpolation(data["dates-S2"].numpy())).float(),
                    data[sat][1],
                )
                data["dates-{}".format(sat)] = data["dates-S2"]

        # Retrieve label
        label = self.labels[item]
        if self.class_mapping is not None:
            label = self.class_mapping[label]
        data["label"] = torch.from_numpy(np.array(label - self.label_offset, dtype=int))

        # Retrieve geometric features
        if self.geom_features:
            gf = (self.geomfeat[item] - self.geom_m) / self.geom_s
            data["geomfeat"] = torch.from_numpy(gf).float()

        return data


def geometric_features(mask):
    """Area, perimeter, perimeter/area and bounding-box cover ratio of a binary mask."""
    area = mask.sum()
    padded = np.pad(mask, 1)
    perimeter = (
        (padded[1:, :] != padded[:-1, :]).sum() + (padded[:, 1:] != padded[:, :-1]).sum()
    )
    rows, cols = np.where(mask)
    bbox = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
    return [area, perimeter, perimeter / area, area / bbox]


def repeat_pixel_and_norm(pixels, n_pixel, norm=None):
    """
    Repeats a pixel if the parcel has fewer pixels than n_pixel.
    Normalises the channels.
    """
    if pixels.shape[-1] < n_pixel:
        if pixels.shape[-1] == 0:
            x = torch.zeros((*pixels.shape[:2], n_pixel))
            mask = np.array([0 for _ in range(n_pixel)])
            mask[0] = 1
        else:
            x = F.pad(pixels, [0, n_pixel - pixels.shape[-1]], mode="replicate")
            mask = np.array(
                [1 for _ in range(pixels.shape[-1])]
                + [0 for _ in range(pixels.shape[-1], n_pixel)]
            )
    else:
        x = pixels
        mask = np.array([1 for _ in range(n_pixel)])

    if norm is not None:
        m, s = norm
        x = (x - m[None, :, None]) / s[None, :, None]  # channelwise normalisation

    return (x, torch.from_numpy(mask))


def sample_pixels(pixels, n_pixel):
    """
    Random sampling of pixels within a parcel.
    """
    if pixels.shape[-1] > n_pixel:
        idx = np.random.choice(
            list(range(pixels.shape[-1])), size=n_pixel, replace=False
        )
        x = pixels[:, :, idx]
    else:
        x = pixels
    return x


def prepare_dates(date_dict, reference_date):
    """Transforms date of observation to number of days since
    the reference date.
    """
    return np.array([
        (datetime.strptime(str(date_dict[k]), "%Y%m%d") - reference_date).days
        for k in sorted(date_dict, key=int)
    ])



# Backward-compatible alias.
PASTIS_Dataset_PixelSet = PLANTEUR_Dataset_PixelSet
