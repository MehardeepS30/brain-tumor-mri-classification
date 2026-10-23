"""Duplicate detection: exact (SHA-256) and near (pHash + Hamming)."""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from brain_tumor.dedupe import (
    aligned_corr,
    aligned_ssim,
    distance_histogram,
    group_duplicates,
    hamming,
    label_conflicts,
    near_pairs,
    near_pairs_invariant,
    phash64,
    phash_variants,
    representatives,
    sha256_file,
)


def _scan(seed, size=256):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x, y = rng.integers(20, size - 60, 2)
        r = int(rng.integers(15, 50))
        d.ellipse((x, y, x + r, y + r), fill=int(rng.integers(60, 255)))
    return img


def _jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")


# ---------------------------------------------------------------- hashing


def test_sha256_identical_bytes_match_and_differ_otherwise(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    _scan(0).save(a)
    b.write_bytes(a.read_bytes())
    _scan(1).save(c)
    assert sha256_file(a) == sha256_file(b)
    assert sha256_file(a) != sha256_file(c)


def test_phash_is_64_bit_int():
    h = phash64(_scan(0))
    assert isinstance(h, int) and 0 <= h < 2**64


def test_phash_close_for_recompressed_and_resized_copy():
    img = _scan(0)
    copy = _jpeg_roundtrip(img.resize((300, 300)), quality=60)
    assert hamming(phash64(img), phash64(copy)) <= 6


def test_phash_far_for_unrelated_images():
    assert hamming(phash64(_scan(0)), phash64(_scan(1))) > 12


def test_hamming_counts_differing_bits():
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(2**64 - 1, 0) == 64


def test_phash_variants_are_the_eight_flips_and_rotations():
    img = _scan(0)
    v = phash_variants(img)
    assert len(v) == 8 and v[0] == phash64(img)
    assert phash64(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)) in v


@pytest.mark.parametrize(
    "op",
    [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_90],
)
def test_invariant_pairs_catch_flipped_or_rotated_copies(op):
    imgs = [_scan(s) for s in range(6)]
    imgs.append(_jpeg_roundtrip(imgs[2].transpose(op), quality=80))  # index 6
    plain = [phash64(i) for i in imgs]
    assert hamming(plain[2], plain[6]) > 12  # plain pHash misses it
    pairs = near_pairs_invariant([phash_variants(i) for i in imgs], max_dist=6)
    assert {(i, j) for i, j, _ in pairs} == {(2, 6)}


# ---------------------------------------------------------------- pairs / histogram


def test_near_pairs_matches_brute_force_and_threshold_is_inclusive():
    rng = np.random.default_rng(0)
    hashes = [int(v) for v in rng.integers(0, 2**63, 60, dtype=np.int64)]
    hashes[5] = hashes[3] ^ 0b111111  # distance exactly 6
    hashes[7] = hashes[3] ^ 0b1111111  # distance exactly 7
    got = {(i, j) for i, j, _ in near_pairs(hashes, max_dist=6)}
    want = {
        (i, j)
        for i in range(60)
        for j in range(i + 1, 60)
        if hamming(hashes[i], hashes[j]) <= 6
    }
    assert got == want
    assert (3, 5) in got and (3, 7) not in got


def test_distance_histogram_covers_all_pairs():
    rng = np.random.default_rng(1)
    hashes = [int(v) for v in rng.integers(0, 2**63, 50, dtype=np.int64)]
    hist = distance_histogram(hashes)
    assert len(hist) == 65
    assert hist.sum() == 50 * 49 // 2


# ---------------------------------------------------------------- grouping


def test_grouping_is_transitive():
    a = 0
    b = a ^ 0b1111  # d(a,b)=4
    c = b ^ (0b1111 << 8)  # d(b,c)=4, d(a,c)=8
    far = 2**64 - 1
    groups = group_duplicates([a, b, c, far], shas=["1", "2", "3", "4"], threshold=6)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_exact_sha_matches_are_grouped_even_if_phash_differs():
    groups = group_duplicates([0, 2**64 - 1], shas=["same", "same"], threshold=6)
    assert groups[0] == groups[1]


def test_singletons_get_distinct_group_ids():
    groups = group_duplicates([0, 2**32 - 1, 2**64 - 1], shas=["a", "b", "c"], threshold=6)
    assert len(set(groups)) == 3


def test_aligned_ssim_is_high_for_flipped_copy_and_low_for_unrelated():
    a = _scan(0)
    flipped = _jpeg_roundtrip(a.transpose(Image.Transpose.FLIP_LEFT_RIGHT), quality=80)
    assert aligned_ssim(a, flipped) > 0.9
    assert aligned_ssim(a, _scan(1)) < 0.7


def test_aligned_corr_ignores_contrast_change_and_flip_but_not_content():
    a = _scan(0)
    arr = np.asarray(a, dtype=np.float64) * 0.6  # contrast change, background stays black
    b = Image.fromarray(arr.astype(np.uint8)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    assert aligned_corr(a, b) > 0.99
    assert aligned_corr(a, _scan(1)) < 0.9


def test_representatives_keep_one_largest_image_per_group():
    groups = [0, 0, 1, 2, 2, 2]
    pixels = [100, 400, 50, 10, 30, 30]
    keep = representatives(groups, pixels)
    assert keep == [False, True, True, False, True, False]  # ties -> first


def test_label_conflicts_lists_groups_with_more_than_one_label():
    groups = [0, 0, 1, 1, 2]
    labels = ["glioma", "glioma", "glioma", "notumor", "pituitary"]
    assert label_conflicts(groups, labels) == {1}
