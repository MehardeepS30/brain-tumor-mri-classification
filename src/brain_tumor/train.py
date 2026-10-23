"""Training loop shared by both models (Part 3 plan).

* class-weighted cross-entropy, weights w_c = N / (K n_c) from the train split
* AdamW; per-step linear warm-up then cosine decay, restarted for each stage
* EfficientNet-B0: `frozen_epochs` with only the head trainable (BN frozen), then
  fine-tuning everything with the backbone at `finetune_lr`, head at `head_finetune_lr`
* model selection on validation macro recall (ties: lower val loss); early stopping
  on the same metric. The test split is never loaded here.
* best.pt / last.pt hold model, optimiser, scheduler, scaler and counters; every
  epoch re-seeds from (seed, epoch), so a resumed run repeats an uninterrupted one.
"""

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from brain_tumor.classes import CLASS_NAMES, NUM_CLASSES
from brain_tumor.dataset import ManifestDataset
from brain_tumor.models import build_model, count_params
from brain_tumor.transforms import build_transforms


@dataclass
class TrainConfig:
    model: str = "baseline"
    pretrained: bool = True
    epochs: int = 40  # total, including frozen_epochs
    frozen_epochs: int = 0  # EfficientNet stage A length
    batch_size: int = 32
    lr: float = 1e-3  # baseline, or EfficientNet head during stage A
    finetune_lr: float = 1e-4  # EfficientNet backbone during stage B
    head_finetune_lr: float = 1e-3  # EfficientNet head during stage B
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.01
    patience: int = 8
    min_delta: float = 0.002
    # 0 on purpose: on Windows each worker process takes ~6 s to start, and loaders
    # are rebuilt every epoch (that is what makes resume exact). Reading the cached
    # 224x224 PNGs in-process costs ~0.08 s per batch.
    num_workers: int = 0
    device: str = "cuda"
    amp: bool = True
    seed: int = 42
    out_dir: str = "outputs/run"
    resume: bool = False
    stop_after: int | None = None  # stop after this many epochs in this call
    max_steps_per_epoch: int | None = None  # smoke tests


def macro_recall(y_true, y_pred, num_classes=NUM_CLASSES):
    """Mean over present classes of TP_c / (TP_c + FN_c); per-class NaN if absent."""
    per = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        m = y_true == c
        if m.any():
            per[c] = (y_pred[m] == c).float().mean()
    return float(per[~per.isnan()].mean()), per


def class_weight_tensor(manifest):
    train = manifest.label[manifest.split == "train"]
    counts = train.value_counts().reindex(CLASS_NAMES, fill_value=0).to_numpy(dtype=np.float64)
    w = np.where(counts > 0, len(train) / (NUM_CLASSES * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def save_checkpoint(path, **state):
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _stage_of(cfg, epoch):
    if cfg.model == "efficientnet_b0" and epoch <= cfg.frozen_epochs:
        return "frozen"
    return "finetune" if cfg.model == "efficientnet_b0" else "full"


def _stage_epochs(cfg, stage):
    if stage == "frozen":
        return cfg.frozen_epochs
    if stage == "finetune":
        return cfg.epochs - cfg.frozen_epochs
    return cfg.epochs


def _optimizer(cfg, model, stage):
    if stage == "frozen":
        model.freeze_backbone()
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr}]
    elif stage == "finetune":
        model.unfreeze_backbone()
        groups = [{"params": model.features.parameters(), "lr": cfg.finetune_lr},
                  {"params": model.classifier.parameters(), "lr": cfg.head_finetune_lr}]
    else:
        groups = [{"params": model.parameters(), "lr": cfg.lr}]
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _scheduler(cfg, opt, steps_per_epoch, n_epochs):
    total = max(1, steps_per_epoch * n_epochs)
    warm = min(steps_per_epoch * cfg.warmup_epochs, total - 1)

    def factor(step):
        if step < warm:
            return (step + 1) / (warm + 1)
        t = (step - warm) / max(1, total - warm)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def _loaders(cfg, manifest, cache_root, epoch):
    man = manifest[manifest.split.isin(["train", "val"])].assign(path=lambda d: d.path + ".png")
    train = ManifestDataset(man, cache_root, "train", build_transforms(train=True, cached=True))
    val = ManifestDataset(man, cache_root, "val", build_transforms(train=False, cached=True))
    g = torch.Generator().manual_seed(cfg.seed * 1000 + epoch)
    pin = cfg.device.startswith("cuda")
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=pin,
              persistent_workers=False)
    return (DataLoader(train, shuffle=True, drop_last=False, generator=g, **kw),
            DataLoader(val, shuffle=False, **kw))


def run_epoch(model, loader, loss_fn, device, opt=None, sched=None, scaler=None, amp=False, max_steps=None):
    """One pass. With `opt`, trains; otherwise evaluates. Returns (mean loss, targets,
    predictions, {grad_norm, amp_skipped_steps})."""
    train = opt is not None
    model.train(train)
    total, n, grad_norms, ys, ps = 0.0, 0, [], [], []
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(dev_type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                params = [p for p in model.parameters() if p.grad is not None]
                grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, float("inf"))))
                scaler.step(opt)
                scaler.update()
                sched.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
            ys.append(y.cpu())
            ps.append(logits.argmax(1).cpu())
    # Non-finite norms are fp16 overflow steps, which GradScaler skips and then
    # lowers its scale; report them separately instead of letting them poison the mean.
    finite = [g for g in grad_norms if math.isfinite(g)]
    gn = {"grad_norm": float(np.mean(finite)) if finite else float("nan"),
          "amp_skipped_steps": len(grad_norms) - len(finite)}
    return total / max(n, 1), torch.cat(ys), torch.cat(ps), gn


def fit(cfg, manifest, cache_root):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    device = cfg.device
    _seed(cfg.seed)
    model = build_model(cfg.model, pretrained=cfg.pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor(manifest).to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    steps = _steps_per_epoch(cfg, manifest)

    start, best, best_loss, bad = 1, -1.0, float("inf"), 0
    stage = _stage_of(cfg, 1)
    opt = _optimizer(cfg, model, stage)
    sched = _scheduler(cfg, opt, steps, _stage_epochs(cfg, stage))
    if cfg.resume and (out / "last.pt").exists():
        ck = load_checkpoint(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        start, best, best_loss, bad = ck["epoch"] + 1, ck["best"], ck["best_loss"], ck["bad_epochs"]
        stage = _stage_of(cfg, start)
        opt = _optimizer(cfg, model, stage)
        sched = _scheduler(cfg, opt, steps, _stage_epochs(cfg, stage))
        if ck["stage"] == stage:  # same stage: continue its optimiser and schedule
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
    elif (out / "log.csv").exists():
        (out / "log.csv").unlink()

    done_here = 0
    for epoch in range(start, cfg.epochs + 1):
        new_stage = _stage_of(cfg, epoch)
        if new_stage != stage:
            stage, bad = new_stage, 0
            opt = _optimizer(cfg, model, stage)
            sched = _scheduler(cfg, opt, steps, _stage_epochs(cfg, stage))
        _seed(cfg.seed * 1000 + epoch)
        train_loader, val_loader = _loaders(cfg, manifest, cache_root, epoch)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        tr_loss, _, _, gnorm = run_epoch(model, train_loader, loss_fn, device, opt, sched, scaler,
                                         cfg.amp, cfg.max_steps_per_epoch)
        va_loss, y, p, _ = run_epoch(model, val_loader, loss_fn, device, amp=cfg.amp,
                                     max_steps=cfg.max_steps_per_epoch)
        mr, per = macro_recall(y, p)
        row = {"epoch": epoch, "stage": stage, "lr": opt.param_groups[0]["lr"],
               "train_loss": tr_loss, "val_loss": va_loss, "val_macro_recall": mr,
               **{f"val_recall_{c}": float(per[i]) for i, c in enumerate(CLASS_NAMES)},
               **gnorm, "trainable_params": count_params(model, trainable_only=True),
               "epoch_seconds": round(time.time() - t0, 1),
               "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else 0}

        improved = mr > best + cfg.min_delta or (abs(mr - best) <= cfg.min_delta and va_loss < best_loss)
        if improved:
            best, best_loss = max(best, mr), va_loss
        early_stop_applies = stage != "frozen"
        bad = 0 if improved or not early_stop_applies else bad + 1
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     scaler=scaler.state_dict(), epoch=epoch, stage=stage, best=best, best_loss=best_loss,
                     bad_epochs=bad, config=asdict(cfg), val_macro_recall=mr)
        if improved:
            save_checkpoint(out / "best.pt", **state)
        save_checkpoint(out / "last.pt", **state)
        _append_log(out / "log.csv", row)

        done_here += 1
        if early_stop_applies and bad >= cfg.patience:
            break
        if cfg.stop_after is not None and done_here >= cfg.stop_after:
            break
    return best


def _steps_per_epoch(cfg, manifest):
    n = int((manifest.split == "train").sum())
    steps = math.ceil(n / cfg.batch_size)
    return min(steps, cfg.max_steps_per_epoch) if cfg.max_steps_per_epoch else steps


def _append_log(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
