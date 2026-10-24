"""Grad-CAM analysis helpers, and Grad-CAM on the saved (selected) checkpoints."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from brain_tumor.gradcam import GradCAM, border_mass, cam_mass_inside, is_blank, overlay
from brain_tumor.evaluate import load_selected_model
from brain_tumor.transforms import build_transforms

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"baseline": ROOT / "outputs/baseline", "efficientnet_b0": ROOT / "outputs/efficientnet_b0"}


# ---------------------------------------------------------------- helpers


def test_cam_mass_inside_mask():
    heat = np.zeros((10, 10))
    heat[2:4, 2:4] = 1.0
    mask = np.zeros((10, 10), bool)
    mask[:5, :5] = True
    assert cam_mass_inside(heat, mask) == pytest.approx(1.0)
    assert cam_mass_inside(np.ones((10, 10)), mask) == pytest.approx(0.25)


def test_border_mass_counts_outer_band():
    heat = np.zeros((20, 20))
    heat[0, :] = 1.0  # all mass on the top row
    assert border_mass(heat, width=2) == pytest.approx(1.0)
    centre = np.zeros((20, 20))
    centre[8:12, 8:12] = 1.0
    assert border_mass(centre, width=2) == pytest.approx(0.0)


def test_blank_map_detection_and_nan_mass():
    blank = np.zeros((8, 8))
    assert is_blank(blank) and not is_blank(np.eye(8))
    assert np.isnan(cam_mass_inside(blank, np.ones((8, 8), bool)))
    assert np.isnan(border_mass(blank, 2))


def test_overlay_is_rgb_in_unit_range_and_blank_map_leaves_image():
    gray = np.linspace(0, 1, 64).reshape(8, 8)
    out = overlay(gray, np.random.default_rng(0).random((8, 8)))
    assert out.shape == (8, 8, 3) and out.min() >= 0 and out.max() <= 1
    plain = overlay(gray, np.zeros((8, 8)))
    assert np.allclose(plain, np.repeat(gray[..., None], 3, axis=2))


# ---------------------------------------------------------------- saved checkpoints


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("name", list(RUNS))
def test_gradcam_on_saved_checkpoint(name):
    ckpt = RUNS[name] / "best.pt"
    man_path = ROOT / "data/manifest.csv"
    if not ckpt.exists() or not man_path.exists():
        pytest.skip("trained checkpoints / data not present")
    before = _sha(ckpt)
    model, ck = load_selected_model(RUNS[name], "cpu")
    assert ck["config"]["model"] == name
    man = pd.read_csv(man_path)
    rows = man[man.split == "test"].head(3)
    tf = build_transforms(train=False, cached=True)
    x = torch.stack([tf(Image.open(ROOT / "data/processed" / (p + ".png"))) for p in rows.path])
    state = {k: v.clone() for k, v in model.state_dict().items()}

    cam = GradCAM(model, model.gradcam_layer)
    target = torch.tensor([0, 1, 3])
    heat, cls = cam(x, class_idx=target)
    assert torch.equal(cls, target)  # the requested class score was used
    assert cam.gradients is not None and torch.isfinite(cam.gradients).all()
    assert cam.gradients.abs().sum() > 0
    assert heat.shape == (3, 224, 224) and torch.isfinite(heat).all()
    assert heat.min() >= 0 and heat.max() <= 1
    # every map is either blank (no positive evidence) or peaks at exactly 1
    peaks = heat.flatten(1).amax(1)
    assert all(p == 0 or abs(p - 1) < 1e-5 for p in peaks.tolist())
    # model and file untouched
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert _sha(ckpt) == before
