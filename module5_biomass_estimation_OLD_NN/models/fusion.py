import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    """
    Fuses deep ConvNeXt features, handcrafted geometric stats, and categorical
    predictions (species, health) into a single dense vector.
    """
    def __init__(self, 
                 deep_dim: int, 
                 handcrafted_dim: int, 
                 num_species: int, 
                 species_emb_dim: int,
                 health_emb_dim: int,
                 fusion_dim: int,
                 dropout: float = 0.3):
        super().__init__()
        
        # Categorical Embeddings
        self.species_embed = nn.Embedding(num_species, species_emb_dim)
        # Assuming 3 health states (0: Unhealthy, 1: Moderate, 2: Healthy) + padding
        self.health_embed = nn.Embedding(4, health_emb_dim)
        
        # Layer Normalization for the raw continuous handcrafted features
        self.handcrafted_norm = nn.LayerNorm(handcrafted_dim)
        
        # Compute total dimension after concatenation
        total_dim = deep_dim + handcrafted_dim + species_emb_dim + health_emb_dim
        
        # Linear projection to unified fusion_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                deep_feats: torch.Tensor, 
                handcrafted: torch.Tensor, 
                species_idx: torch.Tensor, 
                health_idx: torch.Tensor) -> torch.Tensor:
        """
        deep_feats: (B, deep_dim)
        handcrafted: (B, handcrafted_dim)
        species_idx: (B,)
        health_idx: (B,)
        """
        # Embed categorical data
        s_emb = self.species_embed(species_idx)
        h_emb = self.health_embed(health_idx)
        
        # Normalize geometric features to prevent large scalar dominance (e.g. Area in millions)
        handcrafted_norm = self.handcrafted_norm(handcrafted)
        
        # Concatenate everything
        x = torch.cat([deep_feats, handcrafted_norm, s_emb, h_emb], dim=-1)
        
        # Fuse
        fused = self.fusion(x)
        return fused
