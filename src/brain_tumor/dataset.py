"""Dataset that reads one split of manifest.csv."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from brain_tumor.classes import CLASS_TO_IDX
from brain_tumor.splits import SPLITS


class ManifestDataset(Dataset):
    """Rows of `manifest` (a DataFrame with path, label, split) for one split.
    Paths are relative to `root`. Yields (image tensor, class index)."""

    def __init__(self, manifest, root, split, transform=None):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        unknown = set(manifest["label"]) - set(CLASS_TO_IDX)
        if unknown:
            raise ValueError(f"unknown label(s) in manifest: {sorted(unknown)}")
        rows = manifest[manifest["split"] == split]
        self.root = Path(root)
        self.paths = list(rows["path"])
        self.targets = [CLASS_TO_IDX[label] for label in rows["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as img:
            img = img.convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.targets[i], dtype=torch.long)
