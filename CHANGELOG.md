# Changelog

## [1.0.0] - 2026-10-24

First public release.

### Data pipeline
- Kaggle download with SHA-256 record; the raw data is extracted read-only.
- Flip/rotation-aware perceptual hashing, aligned SSIM and pixel correlation to find copies and related images.
- Copy removal, conflicting-label handling, and related-image groups kept within one split.
- Group-aware stratified 70/15/15 split, with assertions that no group, near-identical pair or related pair crosses splits.
- Head crop (all sizeable bright regions), pad-to-square, 224×224 resize, ImageNet normalisation, and a cached deterministic stage.
- Training-only augmentation: resolution jitter, horizontal flip, ±10° rotation, ±10% brightness/contrast.

### Models and training
- Baseline CNN (389,924 parameters) and EfficientNet-B0 with frozen-head then fine-tuning stages.
- Class-weighted cross-entropy, AdamW, warm-up plus cosine schedule, fp16 autocast.
- Model selection and early stopping on validation macro recall.
- Checkpoints that resume bit-exactly.

### Evaluation and explainability
- Test metrics with bootstrap intervals, confusion matrices, ROC/AUC and learning curves.
- Recall broken down by original image size.
- Grad-CAM figures and map-location statistics for both selected checkpoints.

### Results (test split, n = 612)

| Model | Macro recall | Accuracy |
|---|---|---|
| Baseline CNN | 0.803 | 0.799 |
| EfficientNet-B0 | 0.916 | 0.922 |

### Release assets
- `baseline_best.pt` (epoch 36) and `efficientnet_b0_best.pt` (epoch 15): the selected checkpoints.
- `manifest.csv`: the exact split.
- `SHA256SUMS`.
