"""
model.py
---------
Model factory for tree species classification.

Default: EfficientNet-B3 (timm, ImageNet-pretrained) — best accuracy/compute
trade-off for fine-grained texture classification (leaf venation, bark
ridges) at a size that's cheap to deploy (~12M params, fast ONNX export).

The backbone is fully swappable via config (`model.backbone`) to any timm
model name, e.g. "convnext_tiny", "vit_small_patch16_224", "resnet50" —
useful if a future iteration has enough data to justify a larger model.
"""

import timm
import torch
import torch.nn as nn


class TreeSpeciesClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 drop_rate: float = 0.3, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # timm handles the classifier head resizing internally via num_classes.
        # Not all architectures accept drop_path_rate (e.g. plain resnets), so
        # fall back gracefully if the kwarg is rejected.
        try:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    model_cfg = cfg["model"]
    model = TreeSpeciesClassifier(
        backbone=model_cfg["backbone"],
        num_classes=num_classes,
        pretrained=model_cfg["pretrained"],
        drop_rate=model_cfg.get("drop_rate", 0.3),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.2),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
