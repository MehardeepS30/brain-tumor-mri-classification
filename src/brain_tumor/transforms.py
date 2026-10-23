"""Preprocessing: crop to the brain, pad to square, resize, normalise; train-only augmentation."""

import numpy as np
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


def crop_to_brain(img, margin=0.05):
    """Crop to the bounding box of the largest bright connected region, plus `margin`
    of the box size on each side. The foreground threshold is half the Otsu level, so
    dim brain tissue still counts as foreground. A blank image is returned unchanged."""
    gray = np.asarray(img.convert("L"))
    if gray.max() == gray.min():
        return img
    mask = gray > 0.5 * threshold_otsu(gray)
    labels, n = ndimage.label(mask)
    if n == 0:
        return img
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    ys, xs = np.nonzero(labels == int(np.argmax(sizes)) + 1)
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


class ToRGB:
    """MRI slices are greyscale; the models expect 3 channels (ImageNet layout)."""

    def __call__(self, img):
        return img.convert("RGB")


def build_transforms(train):
    """Deterministic preprocessing for eval; adds hflip, +-10 deg rotation and
    +-10% brightness/contrast for training only."""
    steps = [ToRGB(), CropToBrain(margin=0.05), PadToSquare(), T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10, fill=0),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]
    return T.Compose(steps)
