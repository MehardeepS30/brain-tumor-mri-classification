"""Step 3 of the data stage: apply the duplicate rules, split, and write the manifest.

Rules (evidence in reports/data from scripts/duplicate_study.py):
  1. Drop the dataset author's augmented copies (file names containing "-aug-").
  2. Copies: same SHA-256, or flip-aware pHash distance <= COPY_MAX_DIST with
     aligned pixel correlation >= COPY_MIN_CORR. A copy cluster with more than one
     label is dropped entirely (the true label is unknown); otherwise one image
     (the largest) is kept. Pairs just short of the copy rule that are still
     near-identical (corr >= 0.95 or SSIM >= 0.9) but carry different labels are
     dropped too.
  3. Related groups (patient proxy, deliberately over-inclusive): pHash distance
     <= 4, or distance <= 6 with SSIM >= 0.6, or SSIM >= 0.7 (distance <= 8).
     Only same-label pairs are linked; cross-label pairs are listed for review.
  4. Group-aware stratified 70/15/15 split over the related groups.

Writes data/manifest.csv, data/excluded.csv and reports/data/{conflicts.csv,
large_groups.csv, data_stats.json}.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain_tumor.classes import CLASS_NAMES
from brain_tumor.dedupe import group_duplicates, representatives
from brain_tumor.splits import SPLITS, group_stratified_split

COPY_MAX_DIST = 2
COPY_MIN_CORR = 0.99
NEAR_MIN_CORR = 0.95
NEAR_MIN_SSIM = 0.9
LARGE_GROUP = 5
FRACTIONS = (0.70, 0.15, 0.15)


def related(pairs):
    return (pairs.dist <= 4) | ((pairs.dist <= 6) & (pairs.ssim >= 0.6)) | (pairs.ssim >= 0.7)


def near_identical(pairs):
    """High-confidence same-image pairs: in the study, fewer than 0.3% of such pairs
    carry different labels, against 75% for random pairs."""
    return (pairs["corr"] >= NEAR_MIN_CORR) | (pairs.ssim >= NEAR_MIN_SSIM)


def _edges(pairs):
    return list(pairs[["i", "j", "dist"]].itertuples(index=False, name=None))


def class_weights(labels):
    counts = pd.Series(labels).value_counts().reindex(CLASS_NAMES, fill_value=0)
    return {c: float(len(labels) / (len(CLASS_NAMES) * n)) for c, n in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--data", default="data")
    ap.add_argument("--reports", default="reports/data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    interim, data, reports = Path(args.interim), Path(args.data), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(interim / "images.csv")
    pairs = pd.read_csv(interim / "candidate_pairs.csv")
    n = len(df)
    reason = pd.Series("", index=df.index, dtype=object)
    conflicts = []

    # 1. author-augmented copies
    aug = df.path.str.contains("-aug-")
    reason[aug] = "dataset-author augmented copy"

    # 2. copy clusters (computed over all non-augmented images, any label)
    live = ~aug.values
    pairs_live = pairs[live[pairs.i] & live[pairs.j]]
    copy_pairs = pairs_live[(pairs_live.dist <= COPY_MAX_DIST) & (pairs_live["corr"] >= COPY_MIN_CORR)]
    shas = [s if ok else f"excluded-{k}" for k, (s, ok) in enumerate(zip(df.sha256, live))]
    copy_id = np.array(group_duplicates([0] * n, shas, COPY_MAX_DIST, pairs=_edges(copy_pairs)))
    df["copy_cluster_id"] = copy_id
    mixed = df[live].groupby("copy_cluster_id").label.nunique()
    mixed = set(mixed[mixed > 1].index)
    for cid in sorted(mixed):
        members = df[(copy_id == cid) & live]
        conflicts.append({"kind": "copy cluster with different labels (all dropped)", "id": int(cid),
                          "paths": ";".join(members.path), "labels": ";".join(members.label)})
    in_mixed = np.isin(copy_id, list(mixed)) & live
    reason[in_mixed] = "label conflict: same image under different labels"

    live = (reason == "").values
    pixels = (df.width * df.height).where(live, -1).tolist()
    keep = np.array(representatives(copy_id, pixels)) & live
    rep_of = {cid: df.path[i] for i, cid in enumerate(copy_id) if keep[i]}
    dropped_copy = live & ~keep
    reason[dropped_copy] = ["copy of " + rep_of[c] for c in copy_id[dropped_copy]]

    # 2b. near-identical images under different labels that just miss the copy rule
    # (e.g. a mirrored re-save at corr 0.98): the label is unknown, so drop both
    kept = (reason == "").values
    pk = pairs[kept[pairs.i] & kept[pairs.j]]
    hc = pk[pk.cross_label & near_identical(pk)]
    for r in hc.itertuples():
        conflicts.append({"kind": "near-identical pair with different labels (both dropped)", "id": -1,
                          "paths": f"{r.path_i};{r.path_j}", "labels": f"{r.label_i};{r.label_j}",
                          "dist": r.dist, "ssim": round(r.ssim, 3), "corr": r.corr})
    both = np.unique(np.concatenate([hc.i.values, hc.j.values])).astype(int)
    reason.iloc[both] = "label conflict: near-identical image under another label"

    # 3. related groups over the kept images
    kept = (reason == "").values
    pk = pairs[kept[pairs.i] & kept[pairs.j] & related(pairs)]
    for r in pk[pk.cross_label].itertuples():
        conflicts.append({"kind": "similar pair with different labels (not linked)", "id": -1,
                          "paths": f"{r.path_i};{r.path_j}", "labels": f"{r.label_i};{r.label_j}",
                          "dist": r.dist, "ssim": round(r.ssim, 3), "corr": r.corr})
    shas_kept = [s if ok else f"excluded-{k}" for k, (s, ok) in enumerate(zip(df.sha256, kept))]
    group = np.array(group_duplicates([0] * n, shas_kept, 99, pairs=_edges(pk[~pk.cross_label])))

    man = df[kept].copy()
    man["dup_group_id"] = pd.factorize(group[kept])[0]
    man["split"] = group_stratified_split(man.label.tolist(), man.dup_group_id.tolist(), FRACTIONS, seed=args.seed)
    man["patient_id"] = ""
    cols = ["path", "label", "split", "sha256", "phash", "dup_group_id", "copy_cluster_id",
            "patient_id", "source_split", "width", "height"]
    man[cols].to_csv(data / "manifest.csv", index=False)
    pd.DataFrame({"path": df.path[~kept], "label": df.label[~kept], "reason": reason[~kept]}).to_csv(
        data / "excluded.csv", index=False)
    pd.DataFrame(conflicts).to_csv(reports / "conflicts.csv", index=False)

    sizes = man.groupby("dup_group_id").size()
    large = sizes[sizes > LARGE_GROUP]
    (man[man.dup_group_id.isin(large.index)].groupby("dup_group_id")
     .agg(size=("path", "size"), label=("label", "first"), split=("split", "first"),
          example=("path", "first")).sort_values("size", ascending=False)
     .to_csv(reports / "large_groups.csv"))

    # leakage checks: every group in one split, and no near-identical or related
    # pair of kept images (any labels) sits in two different splits
    assert (man.groupby("dup_group_id").split.nunique() == 1).all()
    split_of = dict(zip(man.path, man.split))
    kp = pairs[pairs.path_i.isin(split_of) & pairs.path_j.isin(split_of)]
    across = kp[kp.path_i.map(split_of) != kp.path_j.map(split_of)]
    leaks = across[near_identical(across) | (related(across) & ~across.cross_label)]
    assert leaks.empty, leaks
    cross_split_stats = {"candidate_pairs_across_splits": int(len(across)),
                         "max_ssim_across_splits": round(float(across.ssim.max()), 3),
                         "max_corr_across_splits": round(float(across["corr"].max()), 3),
                         "near_identical_or_related_across_splits": 0}

    by_reason = pd.Series(np.where(reason[~kept].str.startswith("copy of"), "copy of a kept image",
                                   reason[~kept])).value_counts()
    stats = {
        "raw_images": n,
        "raw_by_class": df.label.value_counts().reindex(CLASS_NAMES).to_dict(),
        "excluded_by_reason": by_reason.to_dict(),
        "kept_images": int(kept.sum()),
        "kept_by_class": man.label.value_counts().reindex(CLASS_NAMES).to_dict(),
        "copy_rule": f"sha256 equal, or pHash<= {COPY_MAX_DIST} and corr>= {COPY_MIN_CORR}",
        "related_groups": int(len(sizes)),
        "images_in_multi_image_groups": int(sizes[sizes > 1].sum()),
        "largest_groups": sizes.sort_values(ascending=False).head(5).tolist(),
        "groups_larger_than_5": int(len(large)),
        "conflicts": pd.Series([c["kind"] for c in conflicts]).value_counts().to_dict() if conflicts else {},
        "split_sizes": man.split.value_counts().reindex(SPLITS).to_dict(),
        "split_by_class": {s: man[man.split == s].label.value_counts().reindex(CLASS_NAMES).to_dict() for s in SPLITS},
        "class_weights_train": class_weights(man.label[man.split == "train"].tolist()),
        "groups_spanning_splits": 0,
        **cross_split_stats,
        "seed": args.seed,
    }
    (reports / "data_stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()
