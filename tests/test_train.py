"""Training-loop tests: metric, loss weights, split hygiene, checkpoint resume."""

import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

from brain_tumor.classes import CLASS_NAMES
from brain_tumor.train import TrainConfig, class_weight_tensor, fit, load_checkpoint, macro_recall


def test_macro_recall_averages_per_class_recall():
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    p = torch.tensor([0, 0, 1, 0, 2, 2, 0, 0])
    overall, per_class = macro_recall(y, p, num_classes=4)
    assert per_class.tolist() == [1.0, 0.5, 1.0, 0.0]
    assert overall == pytest.approx(0.625)


def test_macro_recall_skips_classes_absent_from_targets():
    overall, per_class = macro_recall(torch.tensor([0, 0]), torch.tensor([0, 1]), num_classes=4)
    assert overall == pytest.approx(0.5)
    assert torch.isnan(per_class[1:]).all()


def test_class_weights_come_from_train_split_only():
    man = pd.DataFrame({"label": ["glioma"] * 6 + ["notumor"] * 2 + ["meningioma", "pituitary"] * 10,
                        "split": ["train"] * 8 + ["val"] * 10 + ["test"] * 10})
    w = class_weight_tensor(man)
    # train: 6 glioma, 2 notumor, 0 others -> N/(K*n) with empty classes set to 0
    assert w.tolist() == pytest.approx([8 / (4 * 6), 0.0, 8 / (4 * 2), 0.0])


@pytest.fixture
def tiny_cache(tmp_path):
    """12 cached 224x224 images, 3 per class, split into train/val/test."""
    rows = []
    for c, cls in enumerate(CLASS_NAMES):
        for k, split in enumerate(("train", "train", "val")):
            rel = f"{cls}/{k}.jpg"
            img = Image.new("L", (224, 224), 0)
            ImageDraw.Draw(img).ellipse((40 + 20 * c, 40, 180, 180 - 10 * k), fill=60 + 40 * c)
            (tmp_path / cls).mkdir(exist_ok=True)
            img.save(tmp_path / (rel + ".png"))
            rows.append({"path": rel, "label": cls, "split": split})
    rows.append({"path": "does/not/exist.jpg", "label": "glioma", "split": "test"})
    return tmp_path, pd.DataFrame(rows)


def _cfg(out, epochs, **kw):
    base = dict(model="baseline", epochs=epochs, batch_size=4, lr=1e-3, warmup_epochs=1,
                patience=10, num_workers=0, device="cpu", amp=False, seed=0, out_dir=str(out))
    base.update(kw)
    return TrainConfig(**base)


def test_fit_never_reads_the_test_split(tiny_cache, tmp_path):
    root, man = tiny_cache
    fit(_cfg(tmp_path / "run", epochs=1), man, root)  # the test row points at a missing file


def test_fit_writes_best_and_last_checkpoints_and_log(tiny_cache, tmp_path):
    root, man = tiny_cache
    out = tmp_path / "run"
    fit(_cfg(out, epochs=2), man, root)
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    log = pd.read_csv(out / "log.csv")
    assert list(log.epoch) == [1, 2]
    assert {"train_loss", "val_loss", "val_macro_recall", "lr"} <= set(log.columns)
    ck = load_checkpoint(out / "last.pt")
    assert ck["epoch"] == 2 and ck["config"]["model"] == "baseline"


def test_resume_gives_the_same_weights_as_an_uninterrupted_run(tiny_cache, tmp_path):
    root, man = tiny_cache
    straight = tmp_path / "straight"
    fit(_cfg(straight, epochs=3), man, root)

    resumed = tmp_path / "resumed"
    fit(_cfg(resumed, epochs=3, stop_after=1), man, root)  # simulate an interruption
    assert load_checkpoint(resumed / "last.pt")["epoch"] == 1
    fit(_cfg(resumed, epochs=3, resume=True), man, root)

    a = load_checkpoint(straight / "last.pt")["model"]
    b = load_checkpoint(resumed / "last.pt")["model"]
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_effnet_two_stage_freezes_then_unfreezes(tiny_cache, tmp_path):
    root, man = tiny_cache
    out = tmp_path / "eff"
    fit(_cfg(out, epochs=2, model="efficientnet_b0", pretrained=False, frozen_epochs=1,
             lr=1e-3, finetune_lr=1e-4), man, root)
    log = pd.read_csv(out / "log.csv")
    assert list(log.stage) == ["frozen", "finetune"]
    assert log.trainable_params[0] == 1280 * 4 + 4
    assert log.trainable_params[1] > 4_000_000
