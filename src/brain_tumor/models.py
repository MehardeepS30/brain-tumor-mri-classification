"""The two classifiers: a from-scratch baseline CNN and EfficientNet-B0."""

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from brain_tumor.classes import NUM_CLASSES


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def _conv_block(cin, cout):
    # No conv bias: the BatchNorm shift makes it redundant.
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """4 x [Conv3x3-BN-ReLU-MaxPool] (32/64/128/256) -> GAP -> Dropout(0.3) -> Linear."""

    def __init__(self, num_classes=NUM_CLASSES, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, cin = [], 3
        for cout in channels:
            blocks.append(_conv_block(cin, cout))
            cin = cout
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(cin, num_classes)

    @property
    def gradcam_layer(self):
        return self.features[-1]

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.head(self.dropout(x))


class EfficientNetB0Classifier(nn.Module):
    """torchvision EfficientNet-B0 with its 1000-way head replaced by Dropout(0.2) + Linear(1280, 4).

    While the backbone is frozen, its BatchNorm layers are kept in eval mode so
    their running statistics stay at the ImageNet values.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)
        self.features = net.features
        self.avgpool = net.avgpool
        in_features = net.classifier[-1].in_features
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.backbone_frozen = False

    @property
    def gradcam_layer(self):
        return self.features[-1]

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False
        self.backbone_frozen = True
        self.train(self.training)

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True
        self.backbone_frozen = False
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.backbone_frozen:
            for m in self.features.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(1)
        return self.classifier(x)


def build_model(name, num_classes=NUM_CLASSES, pretrained=True):
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if name == "efficientnet_b0":
        return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}; expected 'baseline' or 'efficientnet_b0'")
