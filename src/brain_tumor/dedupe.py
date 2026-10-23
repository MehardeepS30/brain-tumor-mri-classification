"""Exact (SHA-256) and near (64-bit pHash + Hamming distance) duplicate detection."""

import hashlib
from collections import defaultdict

import imagehash
import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def phash64(img):
    """64-bit perceptual hash: 32x32 greyscale -> DCT -> top-left 8x8 -> threshold at median."""
    return int(str(imagehash.phash(img.convert("L"), hash_size=8)), 16)


_DIHEDRAL = (
    None,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)


def phash_variants(img):
    """pHash of the image under all 8 flips/rotations (identity first). pHash itself is
    not flip-invariant, and the dataset contains flipped copies."""
    return [phash64(img if op is None else img.transpose(op)) for op in _DIHEDRAL]


def near_pairs_invariant(variants, max_dist):
    """(i, j, d) with i < j where some flip/rotation of one image is within max_dist
    of the other; d is the smallest such distance."""
    ident = _as_array([v[0] for v in variants])
    best = {}
    for i, vs in enumerate(variants):
        d = np.min([_popcount_row(np.uint64(h), ident) for h in vs], axis=0)
        d[i] = 65
        for j in np.nonzero(d <= max_dist)[0]:
            key = (min(i, int(j)), max(i, int(j)))
            best[key] = min(best.get(key, 65), int(d[j]))
    return sorted((i, j, dd) for (i, j), dd in best.items())


SSIM_SIZE = 64


def ssim_prep(img):
    """Crop to the brain, pad to square, downsample to 64x64 with a light blur (so
    JPEG noise and small shifts matter less)."""
    from brain_tumor.transforms import crop_to_brain, pad_to_square

    img = pad_to_square(crop_to_brain(img.convert("L")))
    return img.resize((SSIM_SIZE, SSIM_SIZE), Image.BOX).filter(ImageFilter.GaussianBlur(1))


def aligned_ssim_prepped(a, b):
    """Best SSIM over the 8 flips/rotations of `a`, for images from ssim_prep."""
    b = np.asarray(b, dtype=np.float64)
    return max(
        structural_similarity(np.asarray(a if op is None else a.transpose(op), dtype=np.float64), b, data_range=255)
        for op in _DIHEDRAL
    )


def aligned_ssim(img_a, img_b):
    return aligned_ssim_prepped(ssim_prep(img_a), ssim_prep(img_b))


CORR_SIZE = 128


def corr_prep(img):
    """Crop, pad to square, 128x128, as 8 z-scored vectors (one per flip/rotation)."""
    from brain_tumor.transforms import crop_to_brain, pad_to_square

    img = pad_to_square(crop_to_brain(img.convert("L"))).resize((CORR_SIZE, CORR_SIZE), Image.BOX)
    out = []
    for op in _DIHEDRAL:
        v = np.asarray(img if op is None else img.transpose(op), dtype=np.float64).ravel()
        v = v - v.mean()
        out.append((v / (np.linalg.norm(v) + 1e-12)).astype(np.float32))
    return out


def aligned_corr_prepped(a, b):
    """Best Pearson correlation between any flip/rotation of a and b (corr_prep output).
    Unchanged by linear brightness/contrast changes, unlike SSIM."""
    return max(float(v @ b[0]) for v in a)


def aligned_corr(img_a, img_b):
    return aligned_corr_prepped(corr_prep(img_a), corr_prep(img_b))


def representatives(groups, pixels):
    """Boolean mask keeping one image per group: the one with most pixels (first on ties)."""
    best = {}
    for i, (g, px) in enumerate(zip(groups, pixels)):
        if g not in best or px > pixels[best[g]]:
            best[g] = i
    keep = set(best.values())
    return [i in keep for i in range(len(groups))]


def hamming(a, b):
    return (a ^ b).bit_count()


def _popcount_row(h, rest):
    return np.bitwise_count(np.bitwise_xor(rest, h)).astype(np.int64)


def _as_array(hashes):
    return np.array(hashes, dtype=np.uint64)


def near_pairs(hashes, max_dist):
    """All (i, j, d) with i < j and Hamming distance d <= max_dist."""
    arr = _as_array(hashes)
    out = []
    for i in range(len(arr) - 1):
        d = _popcount_row(arr[i], arr[i + 1 :])
        for k in np.nonzero(d <= max_dist)[0]:
            out.append((i, i + 1 + int(k), int(d[k])))
    return out


def distance_histogram(hashes):
    """Counts of pairwise Hamming distances 0..64 over all unordered pairs."""
    arr = _as_array(hashes)
    hist = np.zeros(65, dtype=np.int64)
    for i in range(len(arr) - 1):
        hist += np.bincount(_popcount_row(arr[i], arr[i + 1 :]), minlength=65)
    return hist


def group_duplicates(phashes, shas, threshold, pairs=None):
    """Group id per image. Images are joined when their SHA-256 matches or their pHash
    distance is <= threshold, and joins are transitive (union-find). `pairs` (i, j, d)
    replaces the plain pHash comparison, e.g. with near_pairs_invariant output."""
    parent = list(range(len(phashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    first_with_sha = {}
    for i, s in enumerate(shas):
        union(i, first_with_sha.setdefault(s, i))
    if pairs is None:
        pairs = near_pairs(phashes, threshold)
    for i, j, d in pairs:
        if d <= threshold:
            union(i, j)

    roots, ids = {}, []
    for i in range(len(phashes)):
        ids.append(roots.setdefault(find(i), len(roots)))
    return ids


def label_conflicts(groups, labels):
    """Group ids whose members carry more than one label."""
    seen = defaultdict(set)
    for g, label in zip(groups, labels):
        seen[g].add(label)
    return {g for g, ls in seen.items() if len(ls) > 1}
