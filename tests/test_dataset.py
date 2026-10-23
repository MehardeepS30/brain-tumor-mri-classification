"""Manifest-backed dataset: class mapping, split filtering, batch format."""

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from brain_tumor.classes import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES
from brain_tumor.dataset import ManifestDataset
from brain_tumor.models import BaselineCNN
from brain_tumor.transforms import build_transforms


def test_class_mapping_is_fixed_and_matches_kaggle_folders():
    assert CLASS_NAMES == ("glioma", "meningioma", "notumor", "pituitary")
    assert CLASS_TO_IDX == {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
    assert NUM_CLASSES == len(CLASS_NAMES) == 4


@pytest.fixture
def tiny_manifest(tmp_path):
    rows = []
    for i, (cls, split) in enumerate(
        [("glioma", "train"), ("meningioma", "train"), ("notumor", "val"),
         ("pituitary", "test"), ("pituitary", "train"), ("glioma", "train")]
    ):
        rel = f"{cls}/img{i}.jpg"
        (tmp_path / cls).mkdir(exist_ok=True)
        Image.new("L", (300 + 10 * i, 250), 40 + 20 * i).save(tmp_path / rel)
        rows.append({"path": rel, "label": cls, "split": split, "dup_group_id": i})
    return tmp_path, pd.DataFrame(rows)


def test_dataset_filters_by_split(tiny_manifest):
    root, df = tiny_manifest
    ds = ManifestDataset(df, root, split="train", transform=build_transforms(train=False))
    assert len(ds) == 4


def test_dataset_returns_tensor_and_label_index(tiny_manifest):
    root, df = tiny_manifest
    ds = ManifestDataset(df, root, split="test", transform=build_transforms(train=False))
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert y == CLASS_TO_IDX["pituitary"]


def test_loader_batch_feeds_model_to_four_logits(tiny_manifest):
    root, df = tiny_manifest
    ds = ManifestDataset(df, root, split="train", transform=build_transforms(train=True))
    xb, yb = next(iter(DataLoader(ds, batch_size=4)))
    assert xb.shape == (4, 3, 224, 224)
    assert yb.dtype == torch.long
    assert BaselineCNN(num_classes=NUM_CLASSES).eval()(xb).shape == (4, NUM_CLASSES)


def test_unknown_label_is_rejected(tiny_manifest):
    root, df = tiny_manifest
    df.loc[0, "label"] = "tumour"
    with pytest.raises(ValueError, match="unknown label"):
        ManifestDataset(df, root, split="train", transform=build_transforms(train=False))


def test_unknown_split_is_rejected(tiny_manifest):
    root, df = tiny_manifest
    with pytest.raises(ValueError):
        ManifestDataset(df, root, split="training", transform=None)
