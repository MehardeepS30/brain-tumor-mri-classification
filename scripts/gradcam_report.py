"""Part 4: Grad-CAM figures and location statistics for the selected checkpoints.

Post-training analysis only: nothing here changes a model, threshold or preprocessing.

For every test image and each model, Grad-CAM is computed at the model's
`gradcam_layer` (features[-1]) for the predicted class and for the true class. Each map
is summarised by where its weight falls on the 224x224 model input:
  * head_mass      share inside the head mask (same mask as the crop)
  * border_mass    share within 11 px (5%) of the image edge
  * offhead_fg     share on bright pixels outside the head (burned-in text, markers)
A blank map (all zero) means the image gave no positive evidence for that class.

Writes reports/gradcam/: figures, cam_stats_<model>.csv and summary.json.

    python scripts/gradcam_report.py outputs/baseline outputs/efficientnet_b0
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader

from brain_tumor.classes import CLASS_NAMES
from brain_tumor.dataset import ManifestDataset
from brain_tumor.evaluate import classification_metrics, load_selected_model, predict
from brain_tumor.gradcam import GradCAM, border_mass, cam_mass_inside, is_blank, overlay
from brain_tumor.transforms import build_transforms, head_mask

NAMES = {"baseline": "Baseline CNN", "efficientnet_b0": "EfficientNet-B0"}
SMALL_PX = 400  # same definition as scripts/evaluate.py
BORDER = 11
NOTUMOR = CLASS_NAMES.index("notumor")


def compute_maps(model, x_all, classes, device, batch=32):
    cam = GradCAM(model, model.gradcam_layer)
    out = []
    for k in range(0, len(x_all), batch):
        h, _ = cam(x_all[k:k + batch].to(device), class_idx=torch.as_tensor(classes[k:k + batch]))
        out.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def stats_for(maps, heads, fgs):
    rows = []
    for h, hm, fg in zip(maps, heads, fgs):
        rows.append({"blank": is_blank(h), "head_mass": cam_mass_inside(h, hm),
                     "border_mass": border_mass(h, BORDER), "offhead_fg": cam_mass_inside(h, fg & ~hm)})
    return pd.DataFrame(rows)


def figure(rows, title, path, ctx, other=None):
    """rows: test-frame indices. Columns: original | model input | CAM (pred) |
    overlay (pred) | overlay (true class) [| overlay other model (its pred)]."""
    cols = 5 + (other is not None)
    fig, axes = plt.subplots(len(rows), cols, figsize=(2.3 * cols, 2.55 * len(rows)), squeeze=False)
    for r, i in enumerate(rows):
        t, p = ctx["y"][i], ctx["pred"][i]
        raw = ctx["raw"][i]
        gray = ctx["gray"][i]
        hp, ht = ctx["map_pred"][i], ctx["map_true"][i]
        size = "small" if ctx["small"][i] else "large"
        panels = [
            (raw, f"original {raw.shape[1]}x{raw.shape[0]} ({size})"),
            (gray, f"true {CLASS_NAMES[t]}\npred {CLASS_NAMES[p]} ({ctx['conf'][i]:.2f})"),
            (hp, "Grad-CAM, predicted class" + (" [blank]" if is_blank(hp) else "")),
            (overlay(gray, hp), f"overlay: {CLASS_NAMES[p]}"),
            (overlay(gray, ht), f"overlay: true {CLASS_NAMES[t]}" + (" [blank]" if is_blank(ht) else "")
             + (" (= pred)" if t == p else f"\np(true)={ctx['probs'][i, t]:.2f}")),
        ]
        if other is not None:
            op = other["pred"][i]
            panels.append((overlay(gray, other["map_pred"][i]),
                           f"{other['label']}: pred {CLASS_NAMES[op]} ({other['conf'][i]:.2f})"))
        for c, (img, cap) in enumerate(panels):
            a = axes[r, c]
            if img.ndim == 2 and c == 2:
                a.imshow(img, cmap="jet", vmin=0, vmax=1)
            elif img.ndim == 2:
                a.imshow(img, cmap="gray", vmin=0, vmax=1 if c else None)
            else:
                a.imshow(img)
            a.set_title(cap, fontsize=6.5)
            a.set_xticks([])
            a.set_yticks([])
        axes[r, 0].set_ylabel(Path(ctx["path"][i]).name, fontsize=6.5)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (2.55 * len(rows))))  # 0.45 in for the title
    fig.savefig(path, dpi=90)
    plt.close(fig)


def pick(idx, n, seed):
    idx = np.asarray(idx)
    if len(idx) <= n:
        return list(idx)
    return list(np.random.default_rng(seed).choice(idx, n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--metrics", default="reports/results/metrics.json")
    ap.add_argument("--out", default="reports/gradcam")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = json.loads(Path(args.metrics).read_text())

    man = pd.read_csv(args.manifest)
    test = man[man.split == "test"].reset_index(drop=True)
    ds = ManifestDataset(man.assign(path=man.path + ".png"), args.cache, "test",
                         build_transforms(train=False, cached=True))
    x_all = torch.stack([ds[i][0] for i in range(len(ds))])
    gray = [np.asarray(Image.open(Path(args.cache) / (p + ".png")), dtype=np.float32) / 255 for p in test.path]
    heads = [head_mask(np.asarray(Image.open(Path(args.cache) / (p + ".png")))) for p in test.path]
    fgs = [g * 255 > 0.5 * threshold_otsu((g * 255).astype(np.uint8)) for g in gray]
    raw = [np.asarray(Image.open(Path(args.raw) / p).convert("L")) for p in test.path]
    small = (test[["width", "height"]].max(axis=1) < SMALL_PX).to_numpy()

    ctxs, summary = {}, {}
    for run in map(Path, args.runs):
        model, ck = load_selected_model(run, device)
        name = ck["config"]["model"]
        probs, y = predict(model, DataLoader(ds, batch_size=64), device)
        pred = probs.argmax(1)
        # the predictions must reproduce the saved evaluation exactly
        m = classification_metrics(y, probs)
        assert m["confusion_matrix"] == saved[name]["test"]["confusion_matrix"], name
        assert ck["epoch"] == saved[name]["best_epoch"], name
        map_pred = compute_maps(model, x_all, pred, device)
        map_true = compute_maps(model, x_all, y, device)
        st = test[["path", "label", "width", "height"]].assign(
            small=small, pred=[CLASS_NAMES[p] for p in pred], conf=probs.max(1),
            p_true=probs[np.arange(len(y)), y], correct=pred == y)
        sp, stt = stats_for(map_pred, heads, fgs), stats_for(map_true, heads, fgs)
        st = st.join(sp.add_suffix("_pred")).join(stt.add_suffix("_true"))
        st.to_csv(out / f"cam_stats_{name}.csv", index=False)
        ctxs[name] = dict(y=y, pred=pred, probs=probs, conf=probs.max(1), map_pred=map_pred,
                          map_true=map_true, raw=raw, gray=gray, small=small, path=list(test.path),
                          label=NAMES[name], st=st)

        tumour = y != NOTUMOR
        groups = {
            "all test": np.ones(len(y), bool),
            "correct": pred == y,
            "misclassified": pred != y,
            "tumour called notumor": tumour & (pred == NOTUMOR),
            "small tumour, correct": tumour & small & (pred == y),
            "large tumour, correct": tumour & ~small & (pred == y),
            "small tumour called notumor": tumour & small & (pred == NOTUMOR),
            "notumor, correct": (y == NOTUMOR) & (pred == y),
        }
        summary[name] = {
            "best_epoch": ck["epoch"],
            "blank_pred_maps": int(sp.blank.sum()), "blank_true_class_maps": int(stt.blank.sum()),
            # reference: what head_mass / border_mass would be for a perfectly flat map
            "head_area_share_median": round(float(np.median([h.mean() for h in heads])), 3),
            "border_area_share": round(1 - (224 - 2 * BORDER) ** 2 / 224 ** 2, 3),
            "groups": {g: {"n": int(msk.sum()),
                           "head_mass_median": round(float(np.nanmedian(st.head_mass_pred[msk])), 3) if msk.any() else None,
                           "border_mass_median": round(float(np.nanmedian(st.border_mass_pred[msk])), 3) if msk.any() else None,
                           "offhead_fg_median": round(float(np.nanmedian(st.offhead_fg_pred[msk])), 4) if msk.any() else None}
                       for g, msk in groups.items()},
        }

        # figures
        s = args.seed
        correct = [j for c in range(4) for j in pick(np.where((y == c) & (pred == y))[0], 2, s + c)]
        figure(correct, f"{NAMES[name]}: correctly classified test images (2 per class)",
               out / f"{name}_correct.png", ctxs[name])
        wrong = [j for c in range(4) for j in pick(np.where((y == c) & (pred != y))[0], 2, s + c)]
        figure(wrong, f"{NAMES[name]}: misclassified test images (up to 2 per true class)",
               out / f"{name}_misclassified.png", ctxs[name])
        missed = np.where(tumour & (pred == NOTUMOR))[0]
        missed = pick(missed[small[missed]], 6, s) + pick(missed[~small[missed]], 4, s)
        figure(missed, f"{NAMES[name]}: tumours predicted as notumor (small first)",
               out / f"{name}_tumour_as_notumor.png", ctxs[name])
        sl = pick(np.where(groups["small tumour, correct"])[0], 4, s) + \
            pick(np.where(groups["large tumour, correct"])[0], 4, s)
        figure(sl, f"{NAMES[name]}: correctly classified tumours, 4 small then 4 large",
               out / f"{name}_small_vs_large.png", ctxs[name])
        ok = ~sp.blank.to_numpy()
        order = np.argsort(st.head_mass_pred.to_numpy())
        order = [j for j in order if ok[j]]
        figure(order[:4] + order[-4:],
               f"{NAMES[name]}: least (top 4) and most (bottom 4) map weight inside the head",
               out / f"{name}_focus_extremes.png", ctxs[name])

    # focus: the small tumours EfficientNet called notumor, with the baseline alongside
    if {"efficientnet_b0", "baseline"} <= set(ctxs):
        e, b = ctxs["efficientnet_b0"], ctxs["baseline"]
        six = np.where((e["y"] != NOTUMOR) & small & (e["pred"] == NOTUMOR))[0]
        figure(list(six), "EfficientNet-B0: small tumour images predicted as notumor (last column: baseline)",
               out / "efficientnet_b0_small_missed_focus.png", e, other=b)
        rows = e["st"].iloc[six][["path", "label", "width", "height", "conf", "p_true", "head_mass_pred",
                                   "border_mass_pred", "offhead_fg_pred", "head_mass_true", "blank_true"]].copy()
        rows["baseline_pred"] = [CLASS_NAMES[b["pred"][i]] for i in six]
        rows.round(3).to_csv(out / "efficientnet_b0_small_missed.csv", index=False)
        summary["efficientnet_b0_small_missed"] = rows.round(3).to_dict(orient="records")
        summary["test_size_counts"] = {"small_tumour": int(((e["y"] != NOTUMOR) & small).sum()),
                                       "large_tumour": int(((e["y"] != NOTUMOR) & ~small).sum())}

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "efficientnet_b0_small_missed"}, indent=1, default=float))
    if "efficientnet_b0_small_missed" in summary:
        print(pd.DataFrame(summary["efficientnet_b0_small_missed"]).to_string())


if __name__ == "__main__":
    main()
