# Training and test results

Both models were trained with the approved Part 3 plan plus resolution jitter. In each run, the checkpoint picked on **validation** macro recall was evaluated once on the untouched **test** split. Figures are in `reports/results/`, raw numbers in `reports/results/metrics.json`, and per-epoch logs in `outputs/<model>/log.csv`.

> These are classification metrics on one public dataset of web-collected MRI slices. They say nothing about clinical reliability. The test images come from the same sources as the training images; there is no external, prospective or patient-level validation; and patient identity cannot be controlled (see Limitations).

## 1. Final duplicate check before training

- **Three pairs of the same image filed under two labels.** They were mirrored or re-saved copies at pixel correlation 0.978–0.989, just below the copy cut-off of 0.99. Two of the three pairs straddled splits (train/val and test/train). The true label is unknown, so all 6 images were dropped. The rule now drops any cross-label pair with correlation ≥ 0.95 or SSIM ≥ 0.9.
- **Leakage check.** `build_manifest.py` now asserts that no pair of kept images in different splits is near-identical, or related under the grouping rule. Among the 1,917 candidate pairs that cross splits, the highest similarity is SSIM 0.72 (a glioma/meningioma look-alike) and correlation 0.76. Both are within the range of random image pairs.
- **Final split:** 4,087 images, train 2,861, val 614, test 612.
- **Train-split class weights:** glioma 0.677, meningioma 1.176, notumor 1.484, pituitary 1.000.

## 2. Training

| | Baseline CNN | EfficientNet-B0 |
|---|---|---|
| Epochs run | 40 (no early stop) | 5 frozen + 16 fine-tune (early stop at 21) |
| Selected epoch | 36 | 15 (fine-tune stage) |
| Best val macro recall | 0.835 | 0.915 |
| Time per epoch / peak GPU memory | ~16 s / 0.66 GB | ~19 s / 1.5 GB |

- **Validation preprocessing** is deterministic, and a test checks this. Augmentation (resolution jitter, horizontal flip, ±10° rotation, ±10% brightness/contrast) runs only on train.
- **Baseline validation recall swings widely** during the high-learning-rate epochs (for example, glioma recall drops to 0 at epoch 14), then settles as the cosine schedule lowers the rate. This is a common pattern for a small from-scratch CNN with BatchNorm, and it is why selecting on validation matters (`learning_curves_baseline.png`).
- **EfficientNet's first fine-tuning step** overflowed in fp16 and was skipped by GradScaler, which shifts its learning-rate schedule by 1 of about 2,250 scheduled steps. That is harmless; PyTorch's "scheduler before optimizer" warning comes from this step.

## 3. Results

Macro recall is shown with a 95% bootstrap confidence interval (1,000 resamples).

| | Baseline val | Baseline **test** | EfficientNet val | EfficientNet **test** |
|---|---|---|---|---|
| Macro recall | 0.832 (0.80–0.86) | **0.803 (0.77–0.83)** | 0.916 (0.89–0.94) | **0.916 (0.89–0.94)** |
| Accuracy | 0.819 | 0.799 | 0.919 | 0.922 |
| Macro ROC AUC | 0.941 | 0.929 | 0.988 | 0.990 |

Per-class results on test:

| Class | Baseline precision | Baseline recall | Baseline AUC | EffNet precision | EffNet recall | EffNet AUC |
|---|---|---|---|---|---|---|
| glioma (n=226) | 0.961 | 0.761 | 0.940 | 0.939 | **0.947** | 0.990 |
| meningioma (n=130) | 0.599 | 0.654 | 0.830 | 0.862 | **0.862** | 0.982 |
| notumor (n=103) | 0.682 | 0.854 | 0.956 | 0.905 | **0.922** | 0.993 |
| pituitary (n=153) | 0.889 | 0.941 | 0.991 | 0.960 | 0.935 | 0.994 |

**Missed tumours** (a tumour predicted as "no tumour"): the baseline misses 41 of 509 test tumours (8.1%) and EfficientNet misses 10 (2.0%). The most common remaining error for EfficientNet is confusing glioma with meningioma. Confusion matrices are in `confusion_matrices_test.png` and ROC curves in `roc_test.png`.

The two intervals do not overlap, so on this test set the pretrained model is clearly better than the baseline. Both numbers come from a single training seed.

## 4. Does the resolution shortcut remain?

"Small" means the original image's longest side is under 400 px. In the test split, 73 of the 103 "no tumour" images are small, but only 42 of the 509 tumour images are.

| Test tumours called "no tumour" | Small originals (n=42) | Large originals (n=467) | Fisher exact p |
|---|---|---|---|
| Baseline CNN | **28 (67%)** | 13 (2.8%) | 7e-26 |
| EfficientNet-B0 | **6 (14%)** | 4 (0.9%) | 4e-5 |

Recall by class and original size, on test:

| Class | Baseline small | Baseline large | EffNet small | EffNet large |
|---|---|---|---|---|
| glioma (8 small / 218 large) | 0.00 | 0.79 | 0.50 | 0.96 |
| meningioma (29 / 101) | 0.10 | 0.81 | 0.66 | 0.92 |
| notumor (73 / 30) | 0.84 | 0.90 | 0.90 | 0.97 |
| pituitary (5 / 148) | 1.00 | 0.94 | 1.00 | 0.93 |

**Yes, the shortcut remains, and it matters for exactly the costly error.**
- **Baseline:** it largely learned "small or soft image means no tumour".
- **EfficientNet:** much less affected, but a small tumour image is still about 15× more likely to be called "no tumour" than a large one. 6 of its 10 missed tumours come from the 42 small tumour images.
- **Caveat:** the small-tumour groups are small (8 glioma, 29 meningioma), so the per-class numbers are noisy. The pooled difference is highly significant for both models.
- **Likely cause:** image size stands in for the image's source (most "no tumour" images come from a different collection, with different scanners, contrast and compression). Resolution jitter removes the sharpness cue, but cannot remove the others.
- **Not yet measured:** I don't have a no-jitter run to compare against, so I can't yet say how much the jitter helped.

## 5. Limitations

- **No patient-level separation.** The dataset has no patient IDs, so images of the same patient may still sit in different splits. Grouping near-duplicate and closely related images reduces this risk but cannot guarantee it. Adjacent slices of one patient that look different enough would not be grouped.
- **Source confounding.** Class correlates with image source and size (section 4). Some of each model's accuracy may come from source cues rather than anatomy.
- **Label noise.** I found and removed 23 images filed under two labels. Other mislabelled images may remain undetected.
- **One seed, one split.** Each confidence interval reflects test-set sampling only, not variation between training runs.
- **No clinical claim.** These are 2D slices from a curated public dataset without clinical metadata. None of these numbers supports use in diagnosis.

## 6. Next steps (Part 4)

- **Grad-CAM** for both models on `features[-1]`, checking whether the heatmaps land on the tumour or on borders, text and artifacts. The small "no tumour" images and the 10 missed tumours should be looked at first.
- **The README** comparing the two models.
- **Optional:** a no-jitter ablation, and 3 seeds per model.
