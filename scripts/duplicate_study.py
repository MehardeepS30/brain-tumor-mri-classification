"""Step 2 of the data stage: evidence for the duplicate rules.

For every pair within flip/rotation-aware pHash distance 8, computes the aligned
SSIM, then tabulates pair counts and the cross-label rate by (distance, SSIM).
True copies almost never carry different labels, so a high cross-label rate in a
cell means that cell is mostly false matches. Random pairs give the reference.

Writes data/interim/{phash_variants.npy, candidate_pairs.csv} and figures/tables
under reports/data/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from brain_tumor.dedupe import (
    aligned_corr_prepped,
    aligned_ssim_prepped,
    corr_prep,
    distance_histogram,
    near_pairs_invariant,
    phash_variants,
    ssim_prep,
)

SSIM_BINS = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def load_variants(df, raw, cache):
    if not cache.exists():
        variants = []
        for p in tqdm(df.path, desc="pHash x8", mininterval=10):
            with Image.open(raw / p) as img:
                variants.append(phash_variants(img.convert("L")))
        np.save(cache, np.array(variants, dtype=np.uint64))
    return np.load(cache).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--reports", default="reports/data")
    ap.add_argument("--max-dist", type=int, default=8)
    ap.add_argument("--random-pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raw, interim, reports = Path(args.raw), Path(args.interim), Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(interim / "images.csv")

    variants = load_variants(df, raw, interim / "phash_variants.npy")
    plain_hist = distance_histogram([v[0] for v in variants])
    pairs = pd.DataFrame(near_pairs_invariant(variants, args.max_dist), columns=["i", "j", "dist"])

    prepped = [ssim_prep(Image.open(raw / p)) for p in tqdm(df.path, desc="SSIM prep", mininterval=10)]
    pairs["ssim"] = [aligned_ssim_prepped(prepped[i], prepped[j])
                     for i, j in tqdm(zip(pairs.i, pairs.j), total=len(pairs), desc="SSIM", mininterval=10)]
    need = sorted(set(pairs.i[pairs.dist <= 4]) | set(pairs.j[pairs.dist <= 4]))
    cprep = {i: corr_prep(Image.open(raw / df.path[i])) for i in tqdm(need, desc="corr prep", mininterval=10)}
    pairs["corr"] = [aligned_corr_prepped(cprep[i], cprep[j]) if d <= 4 else np.nan
                     for i, j, d in zip(pairs.i, pairs.j, pairs.dist)]
    del cprep
    for k in ("path", "label"):
        pairs[f"{k}_i"] = df[k].values[pairs.i]
        pairs[f"{k}_j"] = df[k].values[pairs.j]
    pairs["cross_label"] = pairs.label_i != pairs.label_j
    pairs.to_csv(interim / "candidate_pairs.csv", index=False)

    rng = np.random.default_rng(args.seed)
    rnd = []
    for _ in range(args.random_pairs):
        i, j = rng.choice(len(df), 2, replace=False)
        rnd.append((aligned_ssim_prepped(prepped[i], prepped[j]), df.label[i] != df.label[j]))
    rnd = pd.DataFrame(rnd, columns=["ssim", "cross_label"])

    pairs["ssim_bin"] = pd.cut(pairs.ssim, SSIM_BINS, right=False)
    counts = pairs.pivot_table(index="dist", columns="ssim_bin", values="cross_label", aggfunc="size", observed=False)
    xrate = pairs.pivot_table(index="dist", columns="ssim_bin", values="cross_label", aggfunc="mean", observed=False)
    counts.to_csv(reports / "pairs_by_distance_and_ssim.csv")
    (100 * xrate).round(1).to_csv(reports / "cross_label_pct_by_distance_and_ssim.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("pairs:\n", counts)
        print("cross-label %:\n", (100 * xrate).round(1))
    corr_tab = pd.crosstab(pairs.dist, pd.cut(pairs["corr"], [-1, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.001]))
    corr_tab.to_csv(reports / "pairs_by_distance_and_corr.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("pairs by aligned pixel correlation (d<=4):\n", corr_tab)
    print(f"random pairs: cross-label {100 * rnd.cross_label.mean():.1f}%, "
          f"SSIM median {rnd.ssim.median():.3f}, 99th pct {rnd.ssim.quantile(0.99):.3f}, max {rnd.ssim.max():.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].bar(range(65), plain_hist + 0.1, width=1.8)
    ax[0].set_yscale("log")
    ax[0].set(xlabel="pHash Hamming distance (no flips)", ylabel="pairs (log)", title="All 25.9M pairs")
    bins = np.linspace(0, 1, 41)
    for d in sorted(pairs.dist.unique()):
        ax[1].hist(pairs.ssim[pairs.dist == d], bins, histtype="step", density=True, label=f"d={d}")
    ax[1].hist(rnd.ssim, bins, histtype="stepfilled", alpha=0.3, color="grey", density=True, label="random")
    ax[1].set(xlabel="aligned SSIM", ylabel="density", title="SSIM by flip-aware pHash distance")
    ax[1].legend(fontsize=7)
    im = ax[2].imshow(100 * xrate.values, cmap="Reds", vmin=0, vmax=75, aspect="auto")
    ax[2].set_xticks(range(len(xrate.columns)), [f"{c.left:.1f}" for c in xrate.columns])
    ax[2].set_yticks(range(len(xrate.index)), xrate.index)
    for (r, c), v in np.ndenumerate(100 * xrate.values):
        if not np.isnan(v):
            ax[2].text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7)
    ax[2].set(xlabel="aligned SSIM (bin start)", ylabel="pHash distance",
              title=f"% pairs with different labels (random: {100 * rnd.cross_label.mean():.0f}%)")
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout()
    fig.savefig(reports / "duplicate_threshold_evidence.png", dpi=120)


if __name__ == "__main__":
    main()
