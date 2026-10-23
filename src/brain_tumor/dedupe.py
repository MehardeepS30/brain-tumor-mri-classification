"""Exact (SHA-256) and near (64-bit pHash + Hamming distance) duplicate detection."""

import hashlib
from collections import defaultdict

import imagehash
import numpy as np


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def phash64(img):
    """64-bit perceptual hash: 32x32 greyscale -> DCT -> top-left 8x8 -> threshold at median."""
    return int(str(imagehash.phash(img.convert("L"), hash_size=8)), 16)


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


def group_duplicates(phashes, shas, threshold):
    """Group id per image. Images are joined when their SHA-256 matches or their pHash
    distance is <= threshold, and joins are transitive (union-find)."""
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
    for i, j, _ in near_pairs(phashes, threshold):
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
