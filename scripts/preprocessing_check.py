"""Step 4 of the data stage: validate the crop on real images and build the cache.

* crop grids: 20 random kept images per class, original with the crop box drawn
  next to the final 224x224 model input
* crop audit over every kept image: share of foreground pixels left outside the
  crop box, plus a grid of the worst cases
* image-size distribution per class, raw vs kept, and a size-only classifier to
  measure how much image size alone reveals the class (a shortcut risk)
* augmentation grid
* cache: the deterministic part of the pipeline for every kept image, written to
  data/processed/<path>.png, with a check that cached == uncached eval tensors
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from brain_tumor.classes import CLASS_NAMES
from brain_tumor.transforms import NORM_MEAN, NORM_STD, build_transforms, head_mask, preprocess_for_cache


def crop_box(gray, margin=0.05):
    """Same rule as transforms.crop_to_brain. Returns the box and the plain threshold
    mask (all foreground, specks included), so the audit can see what the box leaves out."""
    mask = gray > 0.5 * threshold_otsu(gray)
    ys, xs = np.nonzero(head_mask(gray))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = round((x1 - x0 + 1) * margin), round((y1 - y0 + 1) * margin)
    h, w = gray.shape
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + 1 + mx), min(h, y1 + 1 + my)), mask


def to_display(x):
    """Undo normalisation of a (3, H, W) tensor for plotting channel 0."""
    return (x[0] * NORM_STD[0] + NORM_MEAN[0]).clamp(0, 1).numpy()


def grid(rows, raw, out, title, cols=4):
    """rows: list of (path, caption). Each tile = original + box | model input."""
    eval_tf = build_transforms(train=False)
    n = len(rows)
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, 2 * cols, figsize=(2.2 * 2 * cols, 2.4 * r))
    axes = np.atleast_2d(axes)
    for a in axes.flat:
        a.axis("off")
    for k, (path, cap) in enumerate(rows):
        img = Image.open(raw / path)
        gray = np.asarray(img.convert("L"))
        (x0, y0, x1, y1), _ = crop_box(gray)
        a = axes[k // cols, 2 * (k % cols)]
        a.imshow(gray, cmap="gray")
        a.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1))
        a.set_title(f"{Path(path).name} {gray.shape[1]}x{gray.shape[0]}", fontsize=6)
        b = axes[k // cols, 2 * (k % cols) + 1]
        b.imshow(to_display(eval_tf(img)), cmap="gray", vmin=0, vmax=1)
        b.set_title(cap, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--data", default="data")
    ap.add_argument("--reports", default="reports/data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raw, data, reports = Path(args.raw), Path(args.data), Path(args.reports)
    man = pd.read_csv(data / "manifest.csv")
    images = pd.read_csv(data / "interim" / "images.csv")
    stats = {}

    # crop grids, 20 per class
    for cls in CLASS_NAMES:
        sample = man[man.label == cls].sample(20, random_state=args.seed)
        grid([(p, "model input 224x224") for p in sample.path], raw,
             reports / f"crop_grid_{cls}.png", f"{cls}: original + crop box (red) | model input")

    # crop audit over all kept images
    audit = []
    for p in tqdm(man.path, desc="crop audit", mininterval=10):
        gray = np.asarray(Image.open(raw / p).convert("L"))
        (x0, y0, x1, y1), mask = crop_box(gray)
        inside = mask[y0:y1, x0:x1].sum()
        audit.append({"path": p, "fg_outside": 1 - inside / max(mask.sum(), 1),
                      "area_kept": (x1 - x0) * (y1 - y0) / gray.size,
                      "aspect": (x1 - x0) / (y1 - y0)})
    audit = pd.DataFrame(audit)
    audit.to_csv(reports / "crop_audit.csv", index=False)
    stats["crop_fg_outside_pct"] = {q: round(100 * audit.fg_outside.quantile(q), 2) for q in (0.5, 0.9, 0.99, 1.0)}
    stats["crop_area_kept_pct"] = {q: round(100 * audit.area_kept.quantile(q), 1) for q in (0.0, 0.01, 0.5, 1.0)}
    stats["crops_losing_over_5pct_fg"] = int((audit.fg_outside > 0.05).sum())
    stats["crops_keeping_under_25pct_area"] = int((audit.area_kept < 0.25).sum())
    worst = pd.concat([audit.nlargest(8, "fg_outside"), audit.nsmallest(8, "area_kept")])
    grid([(r.path, f"fg outside {100 * r.fg_outside:.1f}%, area {100 * r.area_kept:.0f}%")
          for r in worst.itertuples()], raw, reports / "crop_worst_cases.png",
         "Worst crops: most foreground outside the box (top), smallest box (bottom)")

    # size distribution and size-only shortcut test
    images["kept"] = images.path.isin(man.path)
    size_tab = images.groupby(["label", "kept"]).apply(
        lambda g: pd.Series({"n": len(g), "median_w": g.width.median(), "share_512": (g.width == 512).mean().round(3),
                             "share_under_300": (g.width < 300).mean().round(3)}), include_groups=False)
    size_tab.to_csv(reports / "image_sizes.csv")
    print(size_tab)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a, (name, sub) in zip(ax, [("raw (7,200)", images), ("kept after dedupe", images[images.kept])]):
        for cls in CLASS_NAMES:
            a.hist(sub.width[sub.label == cls], bins=np.arange(100, 1450, 25), histtype="step", label=cls)
        a.set(title=f"Image width, {name}", xlabel="width (px)", yscale="log")
        a.legend()
    fig.tight_layout()
    fig.savefig(reports / "image_sizes.png", dpi=110)
    plt.close(fig)

    feats = man.assign(aspect=man.width / man.height)[["width", "height", "aspect"]].values
    tr, va = (man.split == "train").values, (man.split == "val").values
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(feats[tr], man.label[tr])
    stats["size_only_val_macro_recall"] = round(balanced_accuracy_score(man.label[va], rf.predict(feats[va])), 3)

    # augmentation grid
    train_tf = build_transforms(train=True)
    fig, axes = plt.subplots(4, 7, figsize=(14, 8.5))
    for row, cls in enumerate(CLASS_NAMES):
        p = man[(man.label == cls) & (man.split == "train")].sample(1, random_state=args.seed).path.iloc[0]
        img = Image.open(raw / p)
        axes[row, 0].imshow(to_display(build_transforms(train=False)(img)), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{cls} (eval)", fontsize=8)
        for k in range(1, 7):
            torch.manual_seed(100 * row + k)
            axes[row, k].imshow(to_display(train_tf(img)), cmap="gray", vmin=0, vmax=1)
            axes[row, k].set_title("train aug", fontsize=8)
    for a in axes.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(reports / "augmentation_grid.png", dpi=80)
    plt.close(fig)

    # cache + equivalence check
    cache = data / "processed"
    for p in tqdm(man.path, desc="cache", mininterval=10):
        out = cache / (p + ".png")
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            preprocess_for_cache(Image.open(raw / p)).save(out)
    full, cached = build_transforms(train=False), build_transforms(train=False, cached=True)
    check = man.sample(200, random_state=1).path
    diffs = [float((full(Image.open(raw / p)) - cached(Image.open(cache / (p + ".png")))).abs().max()) for p in check]
    stats["cache_images"] = len(man)
    stats["cache_eval_max_abs_diff_200_samples"] = max(diffs)

    (reports / "preprocessing_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
