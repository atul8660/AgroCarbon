import torch
import torch.nn as nn
from .backbone import BiomassBackbone
from .fusion import FeatureFusion

class ResidualRegressionBlock(nn.Module):
    """
    Stabilizes deep regression for continuous numerical targets.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection match
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + res

class BiomassEstimator(nn.Module):
    """
    The complete end-to-end unified network.
    Backbone -> Fusion -> Residual Regression Head -> Biomass (kg)
    """
    def __init__(self, cfg):
        super().__init__()
        self.backbone = BiomassBackbone(pretrained=cfg.model.pretrained)
        
        self.fusion = FeatureFusion(
            deep_dim=cfg.features.deep_feature_dim,
            handcrafted_dim=cfg.features.num_handcrafted,
            num_species=cfg.features.num_species,
            species_emb_dim=cfg.features.species_embedding_dim,
            health_emb_dim=cfg.features.health_embedding_dim,
            fusion_dim=cfg.model.fusion_dim,
            dropout=cfg.model.dropout
        )
        
        # Residual Regression Head
        self.regressor = nn.Sequential(
            ResidualRegressionBlock(cfg.model.fusion_dim, cfg.model.fusion_dim // 2, cfg.model.dropout),
            ResidualRegressionBlock(cfg.model.fusion_dim // 2, cfg.model.fusion_dim // 4, cfg.model.dropout),
            nn.Linear(cfg.model.fusion_dim // 4, 1) # Output 1 continuous value: AGB in kg
        )
        
    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor, species: torch.Tensor, health: torch.Tensor) -> torch.Tensor:
        """
        Outputs Biomass prediction in kilograms.
        Shape: (B, 1)
        """
        deep_feats = self.backbone(image)
        fused = self.fusion(deep_feats, handcrafted, species, health)
        biomass_kg = self.regressor(fused)
        
        # ReLu on the final output ensures we don't predict negative mass
        return torch.relu(biomass_kg)
