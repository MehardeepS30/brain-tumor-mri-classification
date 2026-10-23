"""Short real-data training check for both models (not a training run).

For each model: 2 epochs of at most --steps batches each, interrupted after epoch 1
and resumed, plus an uninterrupted twin run for comparison. Checks CUDA, loss,
gradients, which parameters move, checkpoints, resume, GPU memory and speed.
Writes reports/smoke_test.json.
"""

import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from brain_tumor.models import build_model
from brain_tumor.train import TrainConfig, _loaders, _seed, class_weight_tensor, fit, load_checkpoint


def moved(a, b, prefix):
    keys = [k for k in a if k.startswith(prefix) and a[k].is_floating_point() and "running" not in k]
    n = sum(not torch.equal(a[k].cpu(), b[k].cpu()) for k in keys)
    return n, len(keys)


def check(cfg, man, cache, out):
    shutil.rmtree(out, ignore_errors=True)
    run, twin = out / "resumed", out / "straight"
    _seed(cfg.seed)
    model = build_model(cfg.model, pretrained=cfg.pretrained)
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # loss of the untrained model on one real batch; ~ln 4 if it starts unbiased
    x, y = next(iter(_loaders(cfg, man, cache, epoch=1)[0]))
    model = model.to(cfg.device).eval()
    with torch.no_grad():
        logits = model(x.to(cfg.device))
        init_loss = float(nn.functional.cross_entropy(logits, y.to(cfg.device),
                                                      weight=class_weight_tensor(man).to(cfg.device)))
    del model

    fit(replace(cfg, out_dir=str(run), stop_after=1), man, cache)
    after1 = load_checkpoint(run / "last.pt")
    fit(replace(cfg, out_dir=str(run), resume=True), man, cache)
    after2 = load_checkpoint(run / "last.pt")
    fit(replace(cfg, out_dir=str(twin)), man, cache)
    straight = load_checkpoint(twin / "last.pt")

    log = pd.read_csv(run / "log.csv")
    res = {"initial_batch_loss": round(init_loss, 4), "log": log.round(4).to_dict(orient="records")}
    head = "head." if cfg.model == "baseline" else "classifier."
    res["epoch1_head_tensors_moved"] = "%d/%d" % moved(init, after1["model"], head)
    res["epoch1_backbone_tensors_moved"] = "%d/%d" % moved(init, after1["model"], "features.")
    res["epoch2_backbone_tensors_moved_vs_epoch1"] = "%d/%d" % moved(after1["model"], after2["model"], "features.")
    res["loss_finite"] = bool(log[["train_loss", "val_loss"]].map(math.isfinite).all().all())
    res["grad_norm_finite_positive"] = bool(((log.grad_norm > 0) & log.grad_norm.map(math.isfinite)).all())
    res["checkpoint_epochs"] = {"after_interrupt": after1["epoch"], "after_resume": after2["epoch"],
                                "best": load_checkpoint(run / "best.pt")["epoch"]}
    res["scheduler_steps"] = {"after_interrupt": after1["scheduler"]["last_epoch"],
                              "after_resume": after2["scheduler"]["last_epoch"]}
    res["resumed_vs_straight_max_abs_diff"] = max(
        float((after2["model"][k].float() - straight["model"][k].float()).abs().max()) for k in straight["model"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--cache", default="data/processed")
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    man = pd.read_csv(args.manifest)
    report = {"cuda_available": torch.cuda.is_available(),
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "torch": torch.__version__, "train_images": int((man.split == "train").sum())}
    common = dict(epochs=2, batch_size=32, num_workers=0, device="cuda", amp=True, seed=42,
                  max_steps_per_epoch=args.steps, warmup_epochs=1)
    cfgs = {"baseline": TrainConfig(model="baseline", lr=1e-3, patience=8, **common),
            "efficientnet_b0": TrainConfig(model="efficientnet_b0", frozen_epochs=1, lr=1e-3,
                                           finetune_lr=1e-4, head_finetune_lr=1e-3, patience=6, **common)}
    for name, cfg in cfgs.items():
        report[name] = check(cfg, man, args.cache, Path(args.out) / name)
    report["expected_initial_loss_ln4"] = round(math.log(4), 4)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/smoke_test.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
