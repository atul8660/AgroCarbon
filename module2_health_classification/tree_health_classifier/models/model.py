"""
model.py
---------
Model factory for tree health classification.

Default: EfficientNet-B3 (timm, ImageNet-pretrained) — same backbone choice
as Module 1 (Species Classification), for two reasons:
1. Fine-grained texture sensitivity: disease lesions, chlorosis, necrotic
   spotting are low-level, localized visual cues — exactly what EfficientNet's
   compound-scaled MBConv blocks are strong at, and multiple published
   results on PlantVillage / Plant Pathology confirm EfficientNet variants
   match or beat ResNet/ViT at this data scale.
2. Deployment consistency: Module 1 and Module 2 run back-to-back in the
   same pipeline per tree image — matching backbone families keeps combined
   inference latency and memory predictable, and simplifies future work like
   sharing a common backbone/feature extractor across both heads.

The backbone is fully swappable via config (`model.backbone`) to any timm
model name if a future iteration needs a different accuracy/compute trade-off.
"""

import timm
import torch
import torch.nn as nn


class TreeHealthClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 drop_rate: float = 0.3, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Not all timm architectures accept drop_path_rate (e.g. plain resnets);
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
    model = TreeHealthClassifier(
        backbone=model_cfg["backbone"],
        num_classes=num_classes,
        pretrained=model_cfg["pretrained"],
        drop_rate=model_cfg.get("drop_rate", 0.3),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.2),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
