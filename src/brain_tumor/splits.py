"""Stratified train/val/test split that never puts one duplicate group in two splits."""

import math
from collections import defaultdict

import numpy as np

SPLITS = ("train", "val", "test")


def group_stratified_split(labels, groups, fractions=(0.70, 0.15, 0.15), seed=42):
    """Return a split name per item.

    Within each class, groups are shuffled (by `seed`), then placed largest first
    into whichever split is furthest below its target count for that class.
    """
    if len(fractions) != len(SPLITS) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError(f"fractions must be three values summing to 1, got {fractions}")
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")

    members = defaultdict(list)
    group_label = {}
    for i, (label, g) in enumerate(zip(labels, groups)):
        if group_label.setdefault(g, label) != label:
            raise ValueError(f"group {g!r} has mixed labels: {group_label[g]!r} and {label!r}")
        members[g].append(i)

    by_class = defaultdict(list)
    for g, label in group_label.items():
        by_class[label].append(g)

    rng = np.random.default_rng(seed)
    out = [None] * len(labels)
    for label in sorted(by_class):
        gs = by_class[label]
        order = rng.permutation(len(gs))
        # stable sort keeps the shuffled order among equal sizes
        gs = sorted((gs[k] for k in order), key=lambda g: -len(members[g]))
        total = sum(len(members[g]) for g in gs)
        targets = np.array(fractions) * total
        counts = np.zeros(len(SPLITS))
        for g in gs:
            s = int(np.argmax(targets - counts))
            counts[s] += len(members[g])
            for i in members[g]:
                out[i] = SPLITS[s]
    return out
