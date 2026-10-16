import torch
import torch.nn as nn
import timm

class BiomassBackbone(nn.Module):
    """
    ConvNeXt Tiny Backbone for Deep Feature Extraction.
    We drop the final classification head to yield dense structural 
    embeddings capable of regression.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load ConvNeXt Tiny without the classifier
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,       # Removes the final fully connected layer
            global_pool="avg"    # Global Average Pooling produces a 1D vector per image
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, H, W)
        Returns:
            features: Tensor of shape (B, 768) for convnext_tiny
        """
        return self.model(x)
