"""Evaluate the selected (best-on-validation) checkpoints on val and the untouched test split.

For each run directory: metrics JSON, confusion matrices, ROC curves, learning curves,
and recall split by original image size (small = longest side < SMALL_PX) to probe the
resolution shortcut. Writes reports/results/.

    python scripts/evaluate.py outputs/baseline outputs/efficientnet_b0
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
from torch.utils.data import DataLoader

from brain_tumor.classes import CLASS_NAMES
from brain_tumor.dataset import ManifestDataset
from brain_tumor.evaluate import bootstrap_ci, classification_metrics, predict, roc_curves
from brain_tumor.models import build_model
from brain_tumor.train import load_checkpoint
from brain_tumor.transforms import build_transforms

SMALL_PX = 400
NAMES = {"baseline": "Baseline CNN", "efficientnet_b0": "EfficientNet-B0"}


def size_breakdown(man, y, pred):
    small = (man[["width", "height"]].max(axis=1) < SMALL_PX).to_numpy()
    rows = []
    for c, cls in enumerate(CLASS_NAMES):
        for name, mask in (("small", small), ("large", ~small)):
            m = mask & (y == c)
            rows.append({"class": cls, "size": name, "n": int(m.sum()),
                         "recall": float((pred[m] == c).mean()) if m.any() else np.nan,
                         "pred_notumor_rate": float((pred[m] == CLASS_NAMES.index("notumor")).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def plot_cm(ax, cm, title):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), CLASS_NAMES, rotation=30)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set(xlabel="predicted", ylabel="true", title=title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--out", default="reports/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = pd.read_csv(args.manifest).assign(path=lambda d: d.path + ".png")
    summary, sizes, preds = {}, [], {}

    for run in map(Path, args.runs):
        ck = load_checkpoint(run / "best.pt", map_location=device)
        name = ck["config"]["model"]
        model = build_model(name, pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        res = {"best_epoch": ck["epoch"], "stage_at_best": ck["stage"]}
        for split in ("val", "test"):
            ds = ManifestDataset(manifest, args.cache, split, build_transforms(train=False, cached=True))
            probs, y = predict(model, DataLoader(ds, batch_size=64), device)
            m = classification_metrics(y, probs)
            m["macro_recall_95ci"] = bootstrap_ci(y, probs.argmax(1), n=1000, seed=0)
            res[split] = m
            preds[(name, split)] = (y, probs)
            sub = manifest[manifest.split == split].reset_index(drop=True)
            sizes.append(size_breakdown(sub, y, probs.argmax(1)).assign(model=name, split=split))
        summary[name] = res

        log = pd.read_csv(run / "log.csv")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(log.epoch, log.train_loss, label="train")
        ax[0].plot(log.epoch, log.val_loss, label="val")
        ax[0].set(xlabel="epoch", ylabel="weighted CE loss", title=f"{NAMES[name]}: loss")
        ax[1].plot(log.epoch, log.val_macro_recall, "k", lw=2, label="macro")
        for cls in CLASS_NAMES:
            ax[1].plot(log.epoch, log[f"val_recall_{cls}"], lw=1, label=cls)
        ax[1].axvline(ck["epoch"], ls="--", c="grey")
        ax[1].set(xlabel="epoch", ylabel="val recall", title="validation recall (dashed: selected)", ylim=(0, 1.02))
        for a in ax:
            a.legend(fontsize=7)
        if "stage" in log and (log.stage == "frozen").any():
            for a in ax:
                a.axvspan(0.5, (log.stage == "frozen").sum() + 0.5, color="grey", alpha=0.1)
        fig.tight_layout()
        fig.savefig(out / f"learning_curves_{name}.png", dpi=110)
        plt.close(fig)

    models = list(summary)
    fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6))
    for a, name in zip(np.atleast_1d(ax), models):
        plot_cm(a, summary[name]["test"]["confusion_matrix"], f"{NAMES[name]} (test)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices_test.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4))
    for a, name in zip(np.atleast_1d(ax), models):
        y, probs = preds[(name, "test")]
        for c, (fpr, tpr, _) in enumerate(roc_curves(y, probs)):
            a.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} AUC={summary[name]['test']['auc'][c]:.3f}")
        a.plot([0, 1], [0, 1], ":", c="grey")
        a.set(xlabel="false positive rate", ylabel="true positive rate", title=f"{NAMES[name]} (test, one-vs-rest)")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_test.png", dpi=110)
    plt.close(fig)

    sizes = pd.concat(sizes)
    sizes.to_csv(out / "recall_by_original_size.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    for name in models:
        for split in ("val", "test"):
            m = summary[name][split]
            print(f"\n{NAMES[name]} {split} (n={m['n']}, best epoch {summary[name]['best_epoch']}): "
                  f"macro recall {m['macro_recall']:.4f} CI {m['macro_recall_95ci'][0]:.3f}-{m['macro_recall_95ci'][1]:.3f}, "
                  f"acc {m['accuracy']:.4f}, macro AUC {m['macro_auc']:.4f}")
            print(pd.DataFrame({"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "auc": m["auc"]},
                               index=CLASS_NAMES).round(4).to_string())
            print("confusion (rows true):", m["confusion_matrix"])
    print("\nrecall by original size (test):")
    print(sizes[sizes.split == "test"].pivot_table(index=["class"], columns=["model", "size"],
                                                   values=["n", "recall", "pred_notumor_rate"]).round(3).to_string())


if __name__ == "__main__":
    main()
