"""Preprocessing: crop to the brain, pad to square, resize, normalise; train-only augmentation."""

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from skimage.filters import threshold_otsu
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights

IMAGE_SIZE = 224

# Read from the pretrained weights so the constants can't drift from them.
_ref = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
NORM_MEAN = tuple(_ref.mean)
NORM_STD = tuple(_ref.std)


MIN_PART = 0.02


def head_mask(gray):
    """Foreground pixels that belong to the head: above half the Otsu level (so dim
    tissue counts) and in a connected region at least MIN_PART of the largest one.
    Keeping every sizeable region, not just the largest, matters on real scans where
    the skull ring is broken or the hemispheres are separated by dark gaps; specks
    and burned-in text are far smaller and are dropped."""
    mask = gray > 0.5 * threshold_otsu(gray)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    keep = np.nonzero(sizes >= MIN_PART * sizes.max())[0] + 1
    return np.isin(labels, keep)


def crop_to_brain(img, margin=0.05):
    """Crop to the bounding box of the head (see head_mask), plus `margin` of the box
    size on each side. A blank image is returned unchanged."""
    gray = np.asarray(img.convert("L"))
    if gray.max() == gray.min():
        return img
    ys, xs = np.nonzero(head_mask(gray))
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx = round((x1 - x0 + 1) * margin)
    my = round((y1 - y0 + 1) * margin)
    box = (max(0, x0 - mx), max(0, y0 - my), min(img.width, x1 + 1 + mx), min(img.height, y1 + 1 + my))
    return img.crop(box)


def pad_to_square(img, fill=0):
    """Pad the shorter side with black, centred, so a later resize doesn't distort shapes."""
    side = max(img.size)
    dw, dh = side - img.width, side - img.height
    return ImageOps.expand(img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2), fill=fill)


class CropToBrain:
    def __init__(self, margin=0.05):
        self.margin = margin

    def __call__(self, img):
        return crop_to_brain(img, self.margin)


class PadToSquare:
    def __call__(self, img):
        return pad_to_square(img)


class ToGray:
    """Some files are stored as RGB/RGBA/palette; the content is greyscale MRI."""

    def __call__(self, img):
        return img.convert("L")


class ToRGB:
    """The models expect 3 channels (ImageNet layout): replicate the grey channel."""

    def __call__(self, img):
        return img.convert("RGB")


class RandomResolution:
    """With probability p, downscale by a random factor in `scale` and resize back.

    Training only. Most "no tumour" originals are ~230 px while the other classes are
    mostly 512 px, so after resizing to 224 sharpness alone hints at the class. Randomly
    softening every class's images makes that cue unreliable. Uses torch's RNG so it
    follows torch.manual_seed like the other augmentations."""

    def __init__(self, p=0.5, scale=(0.35, 1.0)):
        self.p = p
        self.scale = tuple(scale)

    def __call__(self, img):
        if float(torch.rand(1)) >= self.p:
            return img
        lo, hi = self.scale
        s = lo + (hi - lo) * float(torch.rand(1))
        w, h = img.size
        small = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    def __repr__(self):
        return f"RandomResolution(p={self.p}, scale={self.scale})"


def _deterministic_steps():
    return [ToGray(), CropToBrain(margin=0.05), PadToSquare(), T.Resize((IMAGE_SIZE, IMAGE_SIZE))]


def preprocess_for_cache(img):
    """The deterministic part of the pipeline: greyscale 224x224, cropped and padded."""
    return T.Compose(_deterministic_steps())(img)


def build_transforms(train, cached=False):
    """Deterministic preprocessing for eval; adds resolution jitter, hflip, +-10 deg
    rotation and +-10% brightness/contrast for training only. With cached=True the input is
    already the output of preprocess_for_cache, so those steps are skipped."""
    steps = [] if cached else _deterministic_steps()
    if train:
        steps.append(RandomResolution(p=0.5, scale=(0.35, 1.0)))
    steps.append(ToRGB())
    if train:
        steps += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10, fill=0),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]
    return T.Compose(steps)
