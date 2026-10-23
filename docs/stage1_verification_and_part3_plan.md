# Parts 1–2 verification and the proposed Part 3 training plan

## 1. What was run

`pytest` on the project venv (Python 3.13, torch 2.11.0+cu128): **75 passed, 0 failed, no warnings** (commit `b1a5d8e`).
All 7 test files failed first with `ModuleNotFoundError: brain_tumor` before any code existed.

Measured on the RTX 4070 Laptop GPU with a batch of 8 random 224×224 images:

| | Baseline CNN | EfficientNet-B0 |
|---|---|---|
| Parameters (exact) | 389,924 | 4,012,672 |
| Trainable with backbone frozen | n/a | 5,124 (the 1280→4 head only) |
| Trainable after unfreezing | n/a | 4,012,672 |
| Grad-CAM layer | `features[-1]` → (B, 256, 14, 14) | `features[-1]` → (B, 1280, 7, 7) |
| Output | (B, 4) | (B, 4) |

Baseline count by hand (conv layers have no bias because BatchNorm follows; each BN has a scale and a shift):

$$
\sum_{(c_{in},c_{out})} \left(9\,c_{in}c_{out} + 2c_{out}\right) + (256\cdot 4 + 4)
= 928 + 18{,}560 + 73{,}984 + 295{,}424 + 1{,}028 = 389{,}924
$$

Normalisation is read from `EfficientNet_B0_Weights.IMAGENET1K_V1`: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225); a test asserts it. The backbone weights are checked tensor-by-tensor against torchvision's ImageNet weights.

Class mapping (fixed, matches the Kaggle folders): glioma 0, meningioma 1, notumor 2, pituitary 3. The dataset rejects unknown labels and unknown split names, and a loader batch goes straight into the model and gives (B, 4).

## 2. Two tests I changed, and why

**Unfreezing test.** It asserted that every backbone tensor changes after one SGD step. 4 of 211 did not. I looked: they are BatchNorm shifts right before a residual addition, with gradients around $10^{-9}$. The step $\eta g \approx 10^{-10}$ is far below float32 resolution for a value near 1 ($\varepsilon \approx 1.2\times10^{-7}$), so the stored number can't move. The test now checks that every backbone tensor gets a non-zero gradient, that every tensor whose step is representable actually moves, that every conv kernel moves, and that BN running statistics update.

**Grad-CAM "non-empty" test.** It failed for the untrained baseline, and the cause is worth knowing. Both models end in global average pooling and a linear layer at the Grad-CAM layer, so for class $c$:

$$
\frac{\partial y_c}{\partial A^k_{ij}} = \frac{w_{ck}}{HW}
\quad\Rightarrow\quad
\frac{1}{HW}\sum_{i,j}\sum_k \alpha_k^c A^k_{ij} = \frac{y_c - b_c}{HW}
$$

So the map before ReLU averages to the image's evidence for $c$ (the logit minus the head bias). At random init the baseline's evidence was negative for every class, so the ReLU'd map is empty, which is the correct Grad-CAM answer. The tests now check this identity exactly, and check that a briefly fitted model gives a non-empty map for its predicted class. Practical consequence for Part 4: a Grad-CAM for a class the model votes *against* is legitimately blank.

## 3. Still unverified (needs the real data)

- Whether any patient IDs exist in file names or metadata.
- The pHash threshold of 6 on real image pairs (distance histogram, SSIM per distance bucket).
- The crop on 20 real images per class, and the original image-size distribution.
- Real class counts, unreadable files, label-conflict groups.

All of this is the data stage, which needs the download (`masoudnickparvar/brain-tumor-mri-dataset`, about 150 MB). `kaggle.json` is in place and the download code is written and unit-tested, but I have not run it.

## 4. Proposed Part 3: training

Both models get the same data, splits, preprocessing, loss and selection rule, so the comparison is fair. The test split is not touched until both models are final.

**Loss: class-weighted cross-entropy.**

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} w_{y_i}\,\log \frac{e^{z_{i,y_i}}}{\sum_{k=1}^{4} e^{z_{i,k}}},
\qquad w_c = \frac{N}{4\,n_c}
$$

$n_c$ is the class count in the training split. This dataset is close to balanced, so the weights will sit near 1; they are a guard against a class becoming rare after deduplication, not a big lever.

**Optimiser: AdamW**, weight decay $10^{-4}$. Adam adapts the step per parameter, which suits a from-scratch net; AdamW applies decay directly to the weights instead of through the gradient, which regularises more predictably.

**Schedule: cosine decay** after a 1-epoch linear warm-up:
$\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)$.
Large steps early, small steps near the end, with no step-schedule tuning.

| | Baseline | EfficientNet-B0, stage A (frozen) | EfficientNet-B0, stage B (fine-tune) |
|---|---|---|---|
| Trainable | all | head only, BN frozen | all |
| Peak LR | $10^{-3}$ | $10^{-3}$ | $10^{-4}$ (head $10^{-3}$) |
| Max epochs | 40 | 5 | 25 |
| Early-stop patience | 8 | none | 6 |

The low fine-tuning LR keeps the ImageNet features from being overwritten in the first few hundred steps. Stage A first gives the new head sensible weights, so stage B's early gradients aren't random noise flowing back into the backbone.

**Batch size 32** with mixed precision (`torch.autocast`). Both models fit easily in 8 GB at 224×224; 32 gives about 150 steps per epoch on ~4,900 training images, which is enough updates for a short schedule, and BatchNorm statistics are stable at that size.

**Selection metric: macro recall on the validation split** (also called balanced accuracy):

$$
\text{macro recall} = \frac{1}{4}\sum_{c} \frac{TP_c}{TP_c + FN_c}
$$

Recall is what you said matters, because a missed tumour is the costly error, and macro averaging stops a strong class from hiding a weak one. Ties are broken by lower validation loss. I also log per-class recall every epoch, so a model that trades glioma recall for no-tumour recall is visible.

**Early stopping** on the same metric, with a minimum improvement of 0.002 to count as progress.

**Checkpointing:** `best.pt` (best validation macro recall) and `last.pt` every epoch, each with model, optimiser, scheduler, epoch and the config, so an interrupted run resumes instead of restarting. Per-epoch metrics go to a CSV for the learning-curve plots in the README.

**Reproducibility:** fixed seeds for Python, NumPy and torch, the split read from `manifest.csv`, and the config saved next to every checkpoint.

**Speed:** the crop/pad/resize step is deterministic, so I'll cache its output once and apply only the random augmentation per epoch. I expect a baseline epoch in well under a minute on your GPU, but I'll measure it on the first run rather than guess.

**Optional:** train each model with 3 seeds and report mean ± std, so "EfficientNet beats the baseline" is backed by more than one run. It roughly triples training time.
