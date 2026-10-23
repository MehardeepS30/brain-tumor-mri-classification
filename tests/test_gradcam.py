"""Grad-CAM tests, run against both models."""

import copy

import pytest
import torch

from brain_tumor.classes import NUM_CLASSES
from brain_tumor.gradcam import GradCAM
from brain_tumor.models import BaselineCNN, EfficientNetB0Classifier

B = 2
H = W = 224
LAYER_HW = {"baseline": (14, 14), "efficientnet_b0": (7, 7)}


@pytest.fixture(scope="module", params=["baseline", "efficientnet_b0"])
def named_model(request):
    torch.manual_seed(0)
    if request.param == "baseline":
        model = BaselineCNN(num_classes=NUM_CLASSES)
    else:
        model = EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=True)
    return request.param, model


def _x():
    torch.manual_seed(1)
    return torch.randn(B, 3, H, W)


def test_target_layer_activations_are_4d(named_model):
    name, model = named_model
    cam = GradCAM(model, model.gradcam_layer)
    cam(_x())
    assert cam.activations.dim() == 4
    assert cam.activations.shape[0] == B
    assert tuple(cam.activations.shape[2:]) == LAYER_HW[name]


def test_class_score_gradient_wrt_activations(named_model):
    _, model = named_model
    cam = GradCAM(model, model.gradcam_layer)
    cam(_x(), class_idx=torch.tensor([1, 3]))
    g = cam.gradients
    assert g.shape == cam.activations.shape
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0


def _head(model):
    return model.head if isinstance(model, BaselineCNN) else model.classifier[-1]


def _evidence(model, x):
    """Logit minus head bias: the part of each class score that comes from the image."""
    model.eval()
    with torch.no_grad():
        return model(x) - _head(model).bias


def test_heatmap_shape_and_range(named_model):
    _, model = named_model
    heatmap, cls = GradCAM(model, model.gradcam_layer)(_x())
    assert heatmap.shape == (B, H, W)
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    assert cls.shape == (B,)


def test_raw_cam_mean_equals_class_evidence(named_model):
    # Both models end in GAP -> Linear at the target layer, so
    # dy_c/dA^k_ij = w_ck / (HW), and mean_ij sum_k alpha_k A^k_ij = (y_c - b_c) / (HW).
    # Hence the map is non-empty exactly when the image pushes class c up.
    _, model = named_model
    x = _x()
    cam = GradCAM(model, model.gradcam_layer)
    _, cls = cam(x)
    raw = (cam.gradients.mean(dim=(2, 3), keepdim=True) * cam.activations).sum(dim=1)
    hw = raw.shape[1] * raw.shape[2]
    want = _evidence(model, x).gather(1, cls[:, None])[:, 0] / hw
    assert torch.allclose(raw.mean(dim=(1, 2)), want, rtol=1e-3, atol=1e-7)


def test_heatmap_nonempty_for_class_with_positive_evidence(named_model):
    # At random init the baseline has negative evidence for every class (its score
    # is all bias), so its map is legitimately empty. Fit a copy to the two inputs
    # first, as a trained model would be.
    _, model = named_model
    model = copy.deepcopy(model)
    x = _x()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(20):
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([1, 3]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    ev = _evidence(model, x)
    cls = ev.argmax(dim=1)
    assert (ev.gather(1, cls[:, None]) > 0).all()
    heatmap, _ = GradCAM(model, model.gradcam_layer)(x, class_idx=cls)
    assert (heatmap.flatten(1).amax(dim=1) > 0).all()


def test_default_class_is_the_prediction(named_model):
    _, model = named_model
    x = _x()
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    _, cls = GradCAM(model, model.gradcam_layer)(x)
    assert torch.equal(cls, pred)


def test_gradcam_leaves_model_untouched(named_model):
    _, model = named_model
    model.train()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    GradCAM(model, model.gradcam_layer)(_x())
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # params and BN stats
    assert all(p.grad is None for p in model.parameters())
    assert model.training  # mode restored


def test_hooks_are_removed(named_model):
    _, model = named_model
    layer = model.gradcam_layer
    n_fwd = len(layer._forward_hooks)
    GradCAM(model, layer)(_x())
    assert len(layer._forward_hooks) == n_fwd
