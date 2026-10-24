# Model card: brain tumour MRI classifiers v1.0.0

This card covers two models trained in this repository: a from-scratch **Baseline CNN** and a fine-tuned **EfficientNet-B0**. Methods and evidence are described in the [README](../README.md); this card summarises intended use, performance and known risks.

## Model details

| | Baseline CNN | EfficientNet-B0 |
|---|---|---|
| Architecture | 4 × [Conv-BN-ReLU-MaxPool] (32/64/128/256 channels) → GAP → Dropout 0.3 → Linear | torchvision EfficientNet-B0 → Dropout 0.2 → Linear(1280, 4) |
| Parameters | 389,924 | 4,012,672 |
| Initialisation | random | ImageNet (`IMAGENET1K_V1`) |
| Training | 40 epochs, AdamW lr 1e-3 | 5 epochs head-only, then fine-tuning (backbone 1e-4, head 1e-3), early-stopped at 21 |
| Selected checkpoint | epoch 36 | epoch 15 |
| Input | 224×224, greyscale replicated to 3 channels, ImageNet normalisation, after the crop/pad pipeline in `src/brain_tumor/transforms.py` | same |
| Output | softmax over `glioma`, `meningioma`, `notumor`, `pituitary` | same |

Both models were trained with seed 42 on one split (`data/manifest.csv`, published as a release asset).

## Intended use

- **For:** education, portfolio demonstration, and research on dataset leakage, shortcut learning and explainability in medical image classification.
- **Not for:** clinical diagnosis, triage, screening, treatment decisions, or any use on real patients. The models are **not clinically validated**.
- **Inputs outside the training distribution** are unsupported. That includes 3D volumes, other sequences or modalities, other anatomy, and images that have not gone through the repository's preprocessing.

## Data

- **Source:** the Kaggle Brain Tumor MRI Dataset by Masoud Nickparvar (CC BY 4.0). It combines figshare, SARTAJ and Br35H images, and according to its description the "no tumour" images come from Br35H.
- **Cleaning:** 4,087 of the 7,200 images were kept after removing copies, augmented duplicates and conflicting labels.
- **Split:** train 2,861, validation 614, test 612, with related images kept together. There are no patient identifiers.

## Performance

Measured on the test split (n = 612), using the checkpoints selected on validation macro recall. Numbers come from `reports/results/metrics.json`.

| | Baseline CNN | EfficientNet-B0 |
|---|---|---|
| Macro recall (95% bootstrap CI) | 0.803 (0.769–0.832) | 0.916 (0.893–0.940) |
| Accuracy | 0.799 | 0.922 |
| Macro ROC AUC | 0.929 | 0.990 |
| Recall: glioma / meningioma / notumor / pituitary | 0.761 / 0.654 / 0.854 / 0.941 | 0.947 / 0.862 / 0.922 / 0.935 |
| Tumours predicted as `notumor` | 41 / 509 | 10 / 509 |

**Known failure mode: small originals.** Tumour images whose original longest side is under 400 px are predicted `notumor` far more often:
- **Baseline:** 28 of 42 small images (67%), against 13 of 467 large ones (2.8%).
- **EfficientNet-B0:** 6 of 42 (14%), against 4 of 467 (0.9%).

Small image size is associated with the "no tumour" class through image source.

## Explainability

Grad-CAM maps at `features[-1]` are provided for both models (`reports/gradcam/`).
- **EfficientNet-B0:** for correctly classified gliomas and meningiomas, the maps usually concentrate on the visible lesion.
- **Baseline:** its maps often lie on the skull outline and image edges.

These are qualitative observations with coarse location statistics. No lesion masks were available, and Grad-CAM shows where class evidence is concentrated, not what a model has learned or why it decided.

## Limitations and risks

- **Patient-level separation cannot be guaranteed.** The dataset has no patient IDs; related-image grouping reduces, but cannot rule out, the same patient in different splits.
- **Labels may contain residual noise.** 23 images were found under two labels and removed.
- **Source confounding.** Image source and resolution remain associated with class.
- **One run.** One training seed and one split were used; the confidence intervals reflect test-set sampling only.
- **The test split is no longer independent.** It was examined during the post-hoc error analysis (Grad-CAM, size breakdown). Future model changes should be selected on validation data, and ideally evaluated on a new held-out or external set.
- **No external or clinical validation** of any kind has been done.

## Licensing of the checkpoints

The checkpoints are derived from the CC BY 4.0 dataset above, and the EfficientNet-B0 checkpoint is also derived from torchvision's ImageNet-pretrained weights, which carry their own terms. See [Licensing and attribution](../README.md#licensing-and-attribution) in the README.
