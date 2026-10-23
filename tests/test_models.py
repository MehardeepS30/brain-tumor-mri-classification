"""Model tests: output shapes, exact parameter counts, freezing semantics.

Two different things are checked for freezing:
  * flags  -- which parameters have requires_grad=True
  * effect -- which parameters/buffers actually change after an optimiser step
They can disagree: a frozen BatchNorm layer still updates its running
mean/var in train mode even though its weights have requires_grad=False.
"""

import copy

import pytest
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from brain_tumor.classes import NUM_CLASSES
from brain_tumor.models import (
    BaselineCNN,
    EfficientNetB0Classifier,
    build_model,
    count_params,
)

B = 2
H = W = 224


def _x(b=B):
    torch.manual_seed(0)
    return torch.randn(b, 3, H, W)


def _snapshot(module):
    """Copy of every parameter and buffer, keyed by name."""
    state = {f"param:{k}": v.detach().clone() for k, v in module.named_parameters()}
    state.update({f"buffer:{k}": v.detach().clone() for k, v in module.named_buffers()})
    return state


def _changed(before, after_module, prefix):
    after = _snapshot(after_module)
    return {k for k in before if k.startswith(prefix) and not torch.equal(before[k], after[k])}


def _one_step(model, lr=0.1):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    logits = model(_x(4))
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    opt.zero_grad()
    loss.backward()
    opt.step()


@pytest.fixture(scope="module")
def effnet():
    return EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=True)


# ---------------------------------------------------------------- baseline


def test_baseline_output_shape():
    model = BaselineCNN(num_classes=NUM_CLASSES).eval()
    assert model(_x()).shape == (B, NUM_CLASSES)


def test_baseline_exact_parameter_count():
    # conv layers have no bias (BatchNorm follows), each BN has weight+bias
    convs = [(3, 32), (32, 64), (64, 128), (128, 256)]
    expected = sum(cin * cout * 3 * 3 + 2 * cout for cin, cout in convs)
    expected += 256 * NUM_CLASSES + NUM_CLASSES  # linear head
    model = BaselineCNN(num_classes=NUM_CLASSES)
    assert count_params(model) == expected == 389_924
    assert count_params(model, trainable_only=True) == expected


def test_baseline_last_conv_block_activation_shape():
    # 224 -> 112 -> 56 -> 28 -> 14 after four 2x2 max-pools
    model = BaselineCNN(num_classes=NUM_CLASSES).eval()
    acts = model.features(_x())
    assert acts.shape == (B, 256, 14, 14)
    assert model.gradcam_layer is model.features[-1]


def test_baseline_head_is_gap_dropout_linear():
    model = BaselineCNN(num_classes=NUM_CLASSES)
    assert isinstance(model.pool, nn.AdaptiveAvgPool2d)
    assert isinstance(model.dropout, nn.Dropout) and model.dropout.p == 0.3
    assert isinstance(model.head, nn.Linear)
    assert (model.head.in_features, model.head.out_features) == (256, NUM_CLASSES)


def test_baseline_training_step_updates_every_parameter():
    model = BaselineCNN(num_classes=NUM_CLASSES)
    before = _snapshot(model)
    _one_step(model)
    changed = _changed(before, model, "param:")
    all_params = {f"param:{k}" for k, _ in model.named_parameters()}
    assert changed == all_params


# ---------------------------------------------------------------- efficientnet


def test_effnet_output_shape(effnet):
    effnet.eval()
    assert effnet(_x()).shape == (B, NUM_CLASSES)


def test_effnet_classifier_replaced_with_four_logits(effnet):
    linear = effnet.classifier[-1]
    assert isinstance(linear, nn.Linear)
    assert (linear.in_features, linear.out_features) == (1280, NUM_CLASSES)
    assert isinstance(effnet.classifier[0], nn.Dropout) and effnet.classifier[0].p == 0.2


def test_effnet_backbone_uses_imagenet_weights(effnet):
    reference = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    ours = effnet.features.state_dict()
    theirs = reference.features.state_dict()
    assert ours.keys() == theirs.keys()
    assert all(torch.equal(ours[k], theirs[k]) for k in ours)


def test_effnet_gradcam_layer_is_last_feature_block_with_4d_output(effnet):
    effnet.eval()
    assert effnet.gradcam_layer is effnet.features[-1]
    acts = effnet.features(_x())
    assert acts.shape == (B, 1280, 7, 7)


def test_build_model_by_name():
    assert isinstance(build_model("baseline"), BaselineCNN)
    assert isinstance(build_model("efficientnet_b0", pretrained=False), EfficientNetB0Classifier)
    with pytest.raises(ValueError):
        build_model("resnet9000")


# ---------------------------------------------------------------- freezing: flags


def test_freeze_leaves_only_head_requires_grad():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)
    model.freeze_backbone()
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"classifier.1.weight", "classifier.1.bias"}
    assert count_params(model, trainable_only=True) == 1280 * NUM_CLASSES + NUM_CLASSES


def test_unfreeze_enables_requires_grad_everywhere():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)
    model.freeze_backbone()
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.parameters())
    assert count_params(model, trainable_only=True) == count_params(model)


# ---------------------------------------------------------------- freezing: effect


def test_frozen_backbone_batchnorm_stays_in_eval_mode_under_train():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)
    model.freeze_backbone()
    model.train()
    bns = [m for m in model.features.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)
    assert model.classifier.training  # dropout in the head still active


def test_frozen_step_changes_head_only_including_bn_running_stats():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=True)
    model.freeze_backbone()
    before = _snapshot(model)
    _one_step(model)
    changed = _changed(before, model, "")
    assert changed == {"param:classifier.1.weight", "param:classifier.1.bias"}


def test_unfrozen_step_changes_backbone_params_and_bn_running_stats():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=True)
    model.freeze_backbone()
    model.unfreeze_backbone()
    before = _snapshot(model)
    _one_step(model)
    changed_params = _changed(before, model, "param:features.")
    changed_stats = {k for k in _changed(before, model, "buffer:features.") if "running_" in k}
    backbone = dict(model.features.named_parameters())
    # every backbone tensor is in the graph and gets a nonzero gradient
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.values())
    # A few BatchNorm biases right before a residual add get gradients ~1e-9,
    # so lr*grad is below float32 resolution and the stored value can't move.
    # Require movement only where the step is representable.
    eps = torch.finfo(torch.float32).eps
    movable = {
        f"param:features.{k}"
        for k, p in backbone.items()
        if (0.1 * p.grad.abs()).max() > 4 * eps * p.detach().abs().max()
    }
    assert movable <= changed_params
    # and every convolution kernel in the backbone moved
    conv_weights = {f"param:features.{k}" for k, p in backbone.items() if p.dim() == 4}
    assert conv_weights <= changed_params
    assert changed_stats, "BatchNorm running stats should update once unfrozen"


def test_freeze_does_not_alter_weights():
    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)
    before = copy.deepcopy(model.state_dict())
    model.freeze_backbone()
    model.unfreeze_backbone()
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
