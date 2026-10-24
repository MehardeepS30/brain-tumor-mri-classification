"""Evaluation: predictions, confusion matrix, per-class precision/recall, ROC/AUC."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from brain_tumor.classes import NUM_CLASSES


def load_selected_model(run_dir, device):
    """The checkpoint chosen on validation (best.pt) of a run, in eval mode."""
    from brain_tumor.models import build_model
    from brain_tumor.train import load_checkpoint

    ck = load_checkpoint(Path(run_dir) / "best.pt", map_location=device)
    model = build_model(ck["config"]["model"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def predict(model, loader, device):
    """Softmax probabilities and targets for a whole loader, in eval mode."""
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)).float(), dim=1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def classification_metrics(y, probs, num_classes=NUM_CLASSES):
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(num_classes)))
    tp = np.diag(cm).astype(float)
    recall = tp / np.maximum(cm.sum(1), 1)
    precision = tp / np.maximum(cm.sum(0), 1)
    auc = [float(roc_auc_score(y == c, probs[:, c])) if 0 < (y == c).sum() < len(y) else float("nan")
           for c in range(num_classes)]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_recall": float(recall.mean()),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": (2 * precision * recall / np.maximum(precision + recall, 1e-12)).tolist(),
        "auc": auc,
        "macro_auc": float(np.nanmean(auc)),
        "confusion_matrix": cm.tolist(),
    }


def roc_curves(y, probs, num_classes=NUM_CLASSES):
    return [roc_curve(y == c, probs[:, c]) for c in range(num_classes)]


def bootstrap_ci(y, pred, n=1000, seed=0, alpha=0.05, num_classes=NUM_CLASSES):
    """Percentile bootstrap CI for macro recall, resampling images."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], pred[idx]
        rec = [(pp[yy == c] == c).mean() for c in range(num_classes) if (yy == c).any()]
        vals.append(np.mean(rec))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
