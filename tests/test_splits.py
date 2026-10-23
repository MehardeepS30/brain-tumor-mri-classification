"""Group-aware stratified split tests."""

from collections import Counter

import numpy as np
import pytest

from brain_tumor.classes import CLASS_NAMES
from brain_tumor.splits import group_stratified_split

FRACTIONS = (0.70, 0.15, 0.15)


def _synthetic(n_per_class=(1600, 1600, 2000, 1750), seed=0):
    """Labels and duplicate-group ids; ~10% of images sit in groups of 2-4."""
    rng = np.random.default_rng(seed)
    labels, groups, gid = [], [], 0
    for cls, n in zip(CLASS_NAMES, n_per_class):
        left = n
        while left:
            size = int(rng.choice([1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4]))
            size = min(size, left)
            labels += [cls] * size
            groups += [gid] * size
            gid += 1
            left -= size
    return labels, groups


def test_every_item_gets_a_split():
    labels, groups = _synthetic()
    splits = group_stratified_split(labels, groups, FRACTIONS, seed=42)
    assert len(splits) == len(labels)
    assert set(splits) == {"train", "val", "test"}


def test_no_group_spans_two_splits():
    labels, groups = _synthetic()
    splits = group_stratified_split(labels, groups, FRACTIONS, seed=42)
    per_group = {}
    for g, s in zip(groups, splits):
        per_group.setdefault(g, set()).add(s)
    assert all(len(v) == 1 for v in per_group.values())


def test_class_proportions_match_within_one_percent():
    labels, groups = _synthetic()
    splits = group_stratified_split(labels, groups, FRACTIONS, seed=42)
    overall = Counter(labels)
    n = len(labels)
    for split in ("train", "val", "test"):
        sub = Counter(l for l, s in zip(labels, splits) if s == split)
        m = sum(sub.values())
        for cls in CLASS_NAMES:
            assert sub[cls] / m == pytest.approx(overall[cls] / n, abs=0.01)


def test_split_sizes_match_fractions_within_one_percent():
    labels, groups = _synthetic()
    splits = group_stratified_split(labels, groups, FRACTIONS, seed=42)
    c = Counter(splits)
    n = len(labels)
    for split, f in zip(("train", "val", "test"), FRACTIONS):
        assert c[split] / n == pytest.approx(f, abs=0.01)


def test_same_seed_same_split_different_seed_different_split():
    labels, groups = _synthetic()
    a = group_stratified_split(labels, groups, FRACTIONS, seed=42)
    b = group_stratified_split(labels, groups, FRACTIONS, seed=42)
    c = group_stratified_split(labels, groups, FRACTIONS, seed=7)
    assert a == b
    assert a != c


def test_group_with_mixed_labels_is_rejected():
    with pytest.raises(ValueError, match="mixed labels"):
        group_stratified_split(["glioma", "meningioma"], [0, 0], FRACTIONS, seed=0)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        group_stratified_split(["glioma"], [0], (0.7, 0.2, 0.2), seed=0)
