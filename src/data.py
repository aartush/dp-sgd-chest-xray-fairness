"""Dataset loading and controlled imbalance construction.

PneumoniaMNIST is binary (0 = normal, 1 = pneumonia) and natively imbalanced
(1,214 normal vs 3,494 pneumonia in train). We treat *normal* as the minority
class, which is the clinically interesting direction: missing a normal scan is
a false alarm, whereas a normal scan being called pneumonia is the costly error
in a screening setting, so recall on the minority class is the metric that maps
onto harm.

X rays are downloaded once into ./data by medmnist itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Split:
    images: np.ndarray  # (N, 1, 28, 28) float32 in [-1, 1]
    labels: np.ndarray  # (N,) int64 for binary, (N, 14) int64 for multilabel
    name: str

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, idx: np.ndarray) -> "Split":
        return Split(self.images[idx], self.labels[idx], self.name)


class ArrayDataset(Dataset):
    def __init__(self, split: Split):
        self.images = torch.from_numpy(split.images)
        self.labels = torch.from_numpy(split.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.images[i], self.labels[i]


def _to_float(images: np.ndarray) -> np.ndarray:
    # medmnist returns uint8 NHWC; normalise to zero-mean unit-range tensors
    x = images.astype(np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    if x.ndim == 3:
        x = x[:, None, :, :]
    return x.astype(np.float32)


def load_pneumoniamnist() -> dict[str, Split]:
    from medmnist import PneumoniaMNIST

    out = {}
    for split in ("train", "val", "test"):
        ds = PneumoniaMNIST(split=split, download=True, root=str(DATA_ROOT), size=28)
        labels = np.asarray(ds.labels, dtype=np.int64).reshape(-1)
        out[split] = Split(_to_float(np.asarray(ds.imgs)), labels, split)
    return out


def load_chestmnist(max_train: int | None = None, seed: int = 0) -> dict[str, Split]:
    from medmnist import ChestMNIST

    out = {}
    for split in ("train", "val", "test"):
        ds = ChestMNIST(split=split, download=True, root=str(DATA_ROOT), size=28)
        labels = np.asarray(ds.labels, dtype=np.int64)
        out[split] = Split(_to_float(np.asarray(ds.imgs)), labels, split)

    if max_train is not None and max_train < len(out["train"]):
        # unstratified-but-seeded cap keeps rare labels represented approximately
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(out["train"]), max_train, replace=False))
        out["train"] = out["train"].subset(idx)
    return out


def binary_class_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def make_imbalanced_train(
    split: Split,
    ratio: float,
    minority_class: int = 0,
    seed: int = 0,
) -> Split:
    """Downsample the majority class to reach majority:minority == ratio.

    The minority class is left intact so that every imbalance level is compared
    on the same set of minority examples; only the majority pool shrinks. This
    isolates imbalance from sample size for the minority group.
    """
    labels = split.labels
    minority = np.flatnonzero(labels == minority_class)
    majority = np.flatnonzero(labels != minority_class)
    target_majority = int(round(len(minority) * ratio))
    if target_majority >= len(majority):
        return split
    rng = np.random.default_rng(seed)
    keep_majority = rng.choice(majority, target_majority, replace=False)
    idx = np.sort(np.concatenate([minority, keep_majority]))
    return split.subset(idx)


def class_balanced_weights(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Inverse-frequency class weights, normalised to mean 1.

    Public constants derived from the training label distribution; using them
    inside the loss does not consume privacy budget because they carry no
    per-example information beyond the class.
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


def multilabel_pos_weights(labels: np.ndarray) -> np.ndarray:
    pos = labels.sum(axis=0).astype(np.float64)
    neg = len(labels) - pos
    pos[pos == 0] = 1.0
    w = neg / pos
    return (w / w.mean()).astype(np.float64)
