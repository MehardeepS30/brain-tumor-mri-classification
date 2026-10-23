"""Grad-CAM (Selvaraju et al., 2017).

For class c with score y^c and target-layer activations A^k (k = channel):
    alpha_k^c = mean_{i,j} dy^c / dA^k_{ij}
    L^c       = ReLU( sum_k alpha_k^c A^k )
L^c is upsampled to the input size and scaled to [0, 1] per image.
"""

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    def __call__(self, x, class_idx=None):
        """Return (heatmap of shape (B, H, W) in [0, 1], class index per image).

        Runs in eval mode and differentiates only w.r.t. the activations, so
        parameters, their .grad and BatchNorm statistics are left untouched.
        """
        was_training = self.model.training
        captured = {}

        def hook(_module, _inp, out):
            captured["a"] = out

        handle = self.target_layer.register_forward_hook(hook)
        try:
            self.model.eval()
            with torch.enable_grad():
                # Gives the graph a path to the activations even when the layer is frozen.
                x = x.detach().clone().requires_grad_(True)
                logits = self.model(x)
                acts = captured["a"]
                if class_idx is None:
                    class_idx = logits.argmax(dim=1)
                class_idx = torch.as_tensor(class_idx, device=logits.device).reshape(-1)
                score = logits.gather(1, class_idx[:, None]).sum()
                (grads,) = torch.autograd.grad(score, acts)
        finally:
            handle.remove()
            self.model.train(was_training)

        self.activations = acts.detach()
        self.gradients = grads.detach()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
        peak = cam.flatten(1).amax(dim=1).clamp_min(1e-12)
        return cam / peak[:, None, None], class_idx.detach()
