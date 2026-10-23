"""Preprocessing tests: crop, aspect ratio, normalisation, train-only augmentation."""

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights

from brain_tumor.transforms import (
    NORM_MEAN,
    NORM_STD,
    build_transforms,
    crop_to_brain,
    pad_to_square,
)


def _synthetic_scan(w=400, h=300, speck=True):
    """Black canvas, grey ellipse 'brain' off-centre, bright blob 'tumour' at
    the brain's right edge, and an isolated bright speck in a corner."""
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    brain = (150, 60, 330, 250)  # x0, y0, x1, y1
    d.ellipse(brain, fill=110)
    d.ellipse((300, 140, 328, 168), fill=230)  # tumour touching the boundary
    if speck:
        d.rectangle((5, 5, 8, 8), fill=255)
    return img, brain


def _bright_bbox(arr, thr=50):
    ys, xs = np.nonzero(arr > thr)
    return xs.min(), ys.min(), xs.max(), ys.max()


# ---------------------------------------------------------------- normalisation


def test_normalisation_matches_efficientnet_weights():
    ref = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    assert tuple(NORM_MEAN) == pytest.approx(tuple(ref.mean))
    assert tuple(NORM_STD) == pytest.approx(tuple(ref.std))


def test_eval_transform_ends_with_that_normalisation():
    norm = [t for t in build_transforms(train=False).transforms if isinstance(t, T.Normalize)]
    assert len(norm) == 1
    assert tuple(norm[0].mean) == pytest.approx(tuple(NORM_MEAN))
    assert tuple(norm[0].std) == pytest.approx(tuple(NORM_STD))


# ---------------------------------------------------------------- crop


def test_crop_keeps_whole_brain_and_tumour_but_drops_corner_speck():
    img, (x0, y0, x1, y1) = _synthetic_scan()
    out = np.asarray(crop_to_brain(img, margin=0.05))
    bx0, by0, bx1, by1 = _bright_bbox(out, thr=50)
    # the brain ellipse (plus tumour) must be fully inside, not touching an edge
    assert bx0 > 0 and by0 > 0
    assert bx1 < out.shape[1] - 1 and by1 < out.shape[0] - 1
    assert (bx1 - bx0 + 1) == pytest.approx(x1 - x0 + 1, abs=2)
    assert (by1 - by0 + 1) == pytest.approx(y1 - y0 + 1, abs=2)
    assert (out > 200).sum() > 400  # tumour pixels survived
    # speck at (5,5) was far from the brain, so it is gone
    assert out.shape[1] < 400 * 0.7


def test_crop_margin_is_five_percent_of_brain_box():
    img, (x0, y0, x1, y1) = _synthetic_scan(speck=False)
    out = crop_to_brain(img, margin=0.05)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    assert out.width == pytest.approx(bw * 1.1, abs=3)
    assert out.height == pytest.approx(bh * 1.1, abs=3)


def test_crop_of_blank_image_returns_it_unchanged():
    img = Image.new("L", (64, 48), 0)
    out = crop_to_brain(img)
    assert out.size == (64, 48)


def test_crop_accepts_rgb_input():
    img, _ = _synthetic_scan()
    out = crop_to_brain(img.convert("RGB"))
    assert out.mode == "RGB"


# ---------------------------------------------------------------- aspect ratio


def test_pad_to_square_does_not_rescale_content():
    img = Image.new("L", (100, 200), 0)
    ImageDraw.Draw(img).rectangle((10, 10, 89, 189), fill=200)
    out = pad_to_square(img)
    assert out.size == (200, 200)
    bx0, by0, bx1, by1 = _bright_bbox(np.asarray(out))
    assert (bx1 - bx0 + 1, by1 - by0 + 1) == (80, 180)


def test_circle_stays_circular_through_eval_pipeline():
    img = Image.new("L", (400, 200), 0)
    ImageDraw.Draw(img).ellipse((150, 50, 250, 150), fill=200)  # a circle on a wide canvas
    x = build_transforms(train=False)(img)
    # undo normalisation on channel 0 and measure the bright region
    ch = x[0] * NORM_STD[0] + NORM_MEAN[0]
    bx0, by0, bx1, by1 = _bright_bbox(ch.numpy(), thr=0.4)
    assert (bx1 - bx0) == pytest.approx(by1 - by0, abs=2)


# ---------------------------------------------------------------- output format


def test_eval_output_is_float_3x224x224():
    img, _ = _synthetic_scan()
    x = build_transforms(train=False)(img)
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32


def test_train_output_is_float_3x224x224():
    img, _ = _synthetic_scan()
    x = build_transforms(train=True)(img)
    assert x.shape == (3, 224, 224)
    assert x.dtype == torch.float32


# ---------------------------------------------------------------- augmentation only in training

RANDOM_OPS = (T.RandomHorizontalFlip, T.RandomVerticalFlip, T.RandomRotation, T.ColorJitter)


def test_eval_transform_has_no_random_ops_and_is_deterministic():
    tf = build_transforms(train=False)
    assert not any(isinstance(t, RANDOM_OPS) for t in tf.transforms)
    img, _ = _synthetic_scan()
    torch.manual_seed(0)
    a = tf(img)
    torch.manual_seed(123)
    b = tf(img)
    assert torch.equal(a, b)


def test_train_transform_has_the_agreed_augmentations_only():
    tf = build_transforms(train=True)
    kinds = {type(t) for t in tf.transforms if isinstance(t, RANDOM_OPS)}
    assert kinds == {T.RandomHorizontalFlip, T.RandomRotation, T.ColorJitter}
    rot = next(t for t in tf.transforms if isinstance(t, T.RandomRotation))
    assert tuple(rot.degrees) == (-10.0, 10.0)
    cj = next(t for t in tf.transforms if isinstance(t, T.ColorJitter))
    assert tuple(cj.brightness) == pytest.approx((0.9, 1.1))
    assert tuple(cj.contrast) == pytest.approx((0.9, 1.1))
    assert cj.saturation is None and cj.hue is None


def test_train_transform_is_actually_random():
    tf = build_transforms(train=True)
    img, _ = _synthetic_scan()
    outs = []
    for s in range(5):
        torch.manual_seed(s)
        outs.append(tf(img))
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
