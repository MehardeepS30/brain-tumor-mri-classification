# Data validation report

Everything below was measured on the real download. Figures and tables are in `reports/data/`, the smoke-test output is in `reports/smoke_test.json`, and the scripts are in `scripts/`, run in this order: `scan_raw.py` → `duplicate_study.py` → `build_manifest.py` → `preprocessing_check.py` → `smoke_test.py`. The test split was not used for any decision.

## Decisions for you

1. **Accept the new duplicate rule** (section 3). The planned rule, pHash distance ≤ 6, does not work on this data: it misses flipped copies and chains unrelated images together.
2. **Add resolution-jitter augmentation** (section 5). Image size alone predicts the class far above chance, mostly for "no tumour". My recommendation is a random down-then-up resize in training.
3. **Approve the full training run** (section 7).

## 1. Download and raw files

| | |
|---|---|
| Source | Kaggle `masoudnickparvar/brain-tumor-mri-dataset` (your message said `masoudnickvar`, which I took to be a typo), files dated 2026-02-13 |
| Zip SHA-256 | `882817250048c78ef7a759cf23e540d7b581f2327b16663c9d3db12f5d2ffdb4` (164,670,110 bytes) |
| Files | 7,200: Training 4 × 1,400, Testing 4 × 400. 7,196 JPEG, 4 PNG |
| Unreadable / unknown folders | 0 / 0 (every file fully decoded) |
| Class folders | `glioma`, `meningioma`, `notumor`, `pituitary`, matching `CLASS_TO_IDX` = 0, 1, 2, 3 |
| Colour modes | 3,067 L, 4,129 RGB (grey content), 3 RGBA, 1 P. All are converted to greyscale |
| Raw data | `data/raw/`, extracted once and now marked read-only. Nothing in the pipeline writes to it |

This release is perfectly balanced (1,800 per class). The older one had 7,023 images. It reaches 1,800 meningioma images partly by adding 203 files named `Tr-aug-me_*` / `Te-aug-me_*`, which are the dataset author's augmented copies.

**Image sizes:** there are 447 distinct sizes, and 70% of images are 512×512. "No tumour" is the outlier, with median width 230 px and only 1% of its images at 512×512. The other classes have 85–96% of their images at 512×512 (`image_sizes.csv`, `image_sizes.png`).

## 2. Patient IDs

**None are available.** File names are only `Tr-gl_123`-style counters. Just 30 files carry any metadata, and it is editing-software, creator or copyright fields from web-sourced images, with no patient fields. Patient-level splitting is therefore impossible, and the README will say so. As a proxy, the split keeps **related-image groups** (section 3) together, built to over-include rather than under-include.

## 3. Duplicates

**What the data showed:**
- **pHash is not flip-invariant, and the dataset contains flipped and rotated copies.** I hash all 8 flips/rotations of every image and take the minimum distance. Of the 10,039 pairs at distance 0, 8,109 were only found this way.
- **Re-encoded copies are everywhere.** No two files are byte-identical across the Training and Testing folders, so SHA-256 alone finds nothing there. Yet **989 of Kaggle's 1,600 Testing images have a near-exact copy in Training**, and 1,414 (88%) have a copy or a closely related image. Accuracy numbers reported on the original split are therefore inflated.
- **The threshold of 6 fails in two ways.** Grouping transitively at ≤ 6 produces one group of 1,688 images. Also, 12.5% of pairs at distance 6 carry different labels, while true copies almost never do; random pairs differ in label 75% of the time. That cross-label rate measures false matches, so a rate of 12.5% means many pairs at distance 6 are not duplicates.

To set the rules I compared every pair within distance 8 (47.5k pairs) on two extra measures:
- **Aligned SSIM:** computed after the crop, at 64 px, best of 8 orientations.
- **Aligned pixel correlation:** computed at 128 px, so it ignores brightness and contrast changes.

The table gives the percentage of pairs with different labels (from `cross_label_pct_by_distance_and_ssim.csv`):

| pHash dist \ SSIM | <0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | ≥0.9 |
|---|---|---|---|---|---|---|---|
| 0 | 0 | – | 0 | 0 | 0 | 0 | 0.2 |
| 2 | 25 | 62 | 6 | 1.7 | 0 | 0 | 0.2 |
| 4 | 81 | 30 | 22 | 3.5 | 0 | 0 | 0.1 |
| 6 | 62 | 30 | 24 | 7.4 | 0.1 | 0.7 | 0 |
| 8 | 70 | 47 | 29 | 14 | 1.0 | 0.1 | 0 |

**Rules applied** (`build_manifest.py`):
1. **Augmented copies:** drop the 203 `-aug-` files.
2. **Copies** (removed, keeping the largest image): identical SHA-256, or pHash ≤ 2 with pixel correlation ≥ 0.99. When I inspected samples of these pairs, most were the same slice re-saved with different brightness or contrast.
3. **Related groups** (kept, but always in the same split): pHash ≤ 4, or ≤ 6 with SSIM ≥ 0.6, or SSIM ≥ 0.7. This rule is deliberately generous, because an extra link only costs some flexibility in the split, while a missed link leaks data. Only same-label pairs are linked.

**Result:**

| | glioma | meningioma | notumor | pituitary | total |
|---|---|---|---|---|---|
| Raw | 1,800 | 1,800 | 1,800 | 1,800 | 7,200 |
| Kept | 1,510 | 871 | 689 | 1,023 | **4,093** |

- **Removed:** 2,887 copies, 203 augmented files and 17 label conflicts. Every removed file and its reason is in `data/excluded.csv`.
- **Groups:** 1,743 related groups. 3,023 images sit in groups larger than one, 136 groups have more than 5 images, and the largest has 228 (`large_groups.csv`).

**Label conflicts** (`conflicts.csv`):
- **Dropped:** 5 copy clusters (17 images) contain the same image under different labels, for example identical scans filed as both meningioma and pituitary. The true label is unknown, so all 17 were dropped.
- **Kept but not linked:** 104 similar pairs carry different labels. Their median pixel correlation is 0.65, so most are look-alikes rather than copies. Only 3 have correlation ≥ 0.95, and none of those 3 straddles two splits.

## 4. Split

Group-aware stratified split, 70/15/15, seed 42. A built-in assertion confirms that **no group spans two splits**.

| | glioma | meningioma | notumor | pituitary | total |
|---|---|---|---|---|---|
| train | 1,057 | 610 | 482 | 716 | 2,865 |
| val | 227 | 131 | 104 | 154 | 616 |
| test | 226 | 130 | 103 | 153 | 612 |

Class shares agree across splits within 0.2 percentage points. Because the largest groups are placed first and land in train, val and test hold more one-off images: test has 461 groups for 612 images, while train has 819 groups for 2,865 images. This makes the test set the harder one, not the easier one.

**Class weights**, recomputed from train only with $w_c = N/(4\,n_c)$: glioma 0.678, meningioma 1.174, notumor 1.486, pituitary 1.000.

## 5. Preprocessing

**I found and fixed a crop bug.** On real scans the crop kept only the *largest* bright region, which cut off half the head when the skull ring was broken or the hemispheres were separated by dark gaps. In one glioma case it cut through the tumour. The crop now keeps every bright region of at least 2% of the largest one, so specks and burned-in scanner text are still dropped. A new test reproduces the failure, and it failed before the fix.

Crop audit over all 4,093 kept images, where "foreground" means pixels above half the Otsu level:
- Foreground left outside the box has a median of 0%, a 99th percentile of 1.0% and a maximum of 6.0%. The worst 8 cases are all burned-in text in the corners (`crop_worst_cases.png`).
- Only 4 images lose more than 5% of their foreground, and in each case the lost part is text, not head.
- Grids of 20 random images per class show the original with the box next to the model input: `crop_grid_{glioma,meningioma,notumor,pituitary}.png`. Padding to a square before resizing keeps the aspect ratio. The augmented-sample grid is `augmentation_grid.png`.

**A shortcut risk I'd like your decision on:** a random forest that sees only width, height and aspect ratio reaches **0.47 validation macro recall**, against 0.25 for chance. The 224×224 resize hides the size itself, but small source images stay blurrier after upscaling, and most of them are "no tumour". A CNN could learn "blurry means no tumour".
- **Recommended:** add a training-only resolution jitter (with probability 0.5, downscale to 35–100% and resize back) so sharpness stops predicting the class. In evaluation I would also report recall separately for originally small and large images, and look at Grad-CAM on small "no tumour" images.
- **Alternative:** leave preprocessing as it is and only document the risk.

**Cache:** the deterministic steps (greyscale, crop, pad, resize to 224) are cached as PNGs under `data/processed/`. On 200 sampled images, the cached and uncached eval tensors differ by a maximum of 0.0.

## 6. Smoke test (real images, GPU)

RTX 4070 Laptop, torch 2.11.0+cu128, CUDA available. For each model the run was 2 epochs of 20 batches, interrupted after epoch 1 and resumed, plus an uninterrupted twin run.

| Check | Baseline | EfficientNet-B0 |
|---|---|---|
| Loss of the untrained model on one batch (ln 4 = 1.386) | 1.370 | 1.464 |
| Train loss, epoch 1 → 2 | 0.988 → 0.870 | 1.281 (frozen) → 1.100 (fine-tune) |
| Val macro recall after epoch 2 (40 batches seen, not a result) | 0.41 | 0.68 |
| Mean grad norm, finite | 2.35, 1.85 | 3.55, 3.79 |
| Tensors that moved in epoch 1 | head 2/2, convs 12/12 | head 2/2, **backbone 0/211** (frozen) |
| Backbone tensors that moved in epoch 2 | 12/12 | **211/211** (unfrozen) |
| `best.pt` / `last.pt` written, epochs resume 1 → 2 | yes | yes |
| Resumed vs uninterrupted weights, max difference | **0.0** | **0.0** |
| Peak GPU memory | 663 MB | 1,510 MB of 8 GB |
| fp16 overflow steps skipped by GradScaler | 0 | 1 (at the unfreeze, expected) |

Two problems came up and are fixed:
- **Logging:** the gradient-norm mean became NaN because of that one skipped fp16 step. Skipped steps are now counted separately.
- **Speed:** the first smoke run took 30 s per 20-batch epoch. The cause was Windows starting 4 DataLoader worker processes, 12.7 s per loader, twice per epoch. With the cache, loading in-process costs 0.08 s per batch, so `num_workers` is now 0, which keeps resume exact.

From the 20-batch timings I estimate about 15 s per full baseline epoch and about 35 s per EfficientNet epoch. That makes the planned full runs (40 and 30 epochs, at most) roughly 10 and 20 minutes. This is an estimate, not a measurement.

The training loop has its own tests (`tests/test_train.py`). They cover the macro-recall metric, train-only class weights, that the test split is never read, the checkpoint files, bit-exact resume and EfficientNet's freeze-then-unfreeze stages. **The full suite is 92 tests, all passing.**

## 7. Still open

- **Full training:** waiting for your approval. The configs are ready: `configs/baseline.json` and `configs/efficientnet_b0.json`.
- **Resolution jitter:** waiting for your decision (section 5).
- **Patient IDs:** patient-level leakage cannot be ruled out, only reduced. The README will state this.
- **Fewer images than expected:** the kept dataset is 43% smaller than the download, and "no tumour" fell the most (1,800 → 689). The class weights compensate in the loss, and the per-class recall will show whether it is enough.
