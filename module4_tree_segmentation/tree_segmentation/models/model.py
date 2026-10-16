"""
model.py
---------
Model factory for tree segmentation.

Production model: SegFormer-B5 (`nvidia/mit-b5` encoder), fine-tuned as a
3-class semantic segmenter (background/canopy/trunk). See README section 2
and the project design notes for the full SAM2 vs. SegFormer-B5 vs.
Mask2Former comparison and why SegFormer-B5 is what's actually served.

SAM2Wrapper is a thin, optional wrapper used by data/sam2_pseudo_labeling.py
(training-set bootstrapping) and, if enabled in config, by
inference/segment.py as a zero-shot fallback path — never the primary model.

OFFLINE NOTE
------------
TreeSegformer is constructed from a locally-defined SegformerConfig rather
than from_pretrained() so that inference works without any HuggingFace
network access. The B5 architecture hyperparameters are hardcoded here
(depths, hidden_sizes, etc.) — they are fixed by the SegFormer-B5 paper
and will not change.

Checkpoints trained under transformers<=4.x used different internal key
names than transformers>=5.x (e.g. encoder.block → stages, dense1 → fc1).
remap_checkpoint_keys() translates the old layout to the current one so
that any checkpoint produced by either version loads cleanly.
"""

import logging
import re

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

logger = logging.getLogger("model")


# ---------------------------------------------------------------------------
# SegFormer-B5 architecture constants (from the SegFormer paper / HF model card)
# These are FIXED for the B5 variant and must not be changed without retraining.
# ---------------------------------------------------------------------------
_SEGFORMER_B5_CONFIG = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[3, 6, 40, 3],               # B5 transformer depths per stage
    sr_ratios=[8, 4, 2, 1],             # sequence-reduction ratios per stage
    hidden_sizes=[64, 128, 320, 512],   # embedding dims per stage
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    num_attention_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    classifier_dropout_prob=0.1,
    decoder_hidden_size=768,            # MLP decode head width
    image_size=640,
    semantic_loss_ignore_index=255,
)


def remap_checkpoint_keys(state_dict: dict) -> dict:
    """Remap a checkpoint's state_dict from the old HuggingFace transformers
    v4.x SegFormer key layout to the current v5.x layout.

    This handles all known renames between the two versions:

    Old (v4.x, stripped of 'model.' wrapper prefix)     → New (v5.x)
    ─────────────────────────────────────────────────    ─────────────────────────────────────
    segformer.encoder.patch_embeddings.N.*               segformer.stages.N.patch_embeddings.*
    segformer.encoder.layer_norm.N.*                     segformer.stages.N.layer_norm.*
    segformer.encoder.block.N.K.attention.self.query.*   segformer.stages.N.blocks.K.attention.q_proj.*
    segformer.encoder.block.N.K.attention.self.key.*     segformer.stages.N.blocks.K.attention.k_proj.*
    segformer.encoder.block.N.K.attention.self.value.*   segformer.stages.N.blocks.K.attention.v_proj.*
    segformer.encoder.block.N.K.attention.output.dense.* segformer.stages.N.blocks.K.attention.o_proj.*
    segformer.encoder.block.N.K.attention.self.sr.*      segformer.stages.N.blocks.K.attention.sequence_reduction.sequence_reduction.*
    segformer.encoder.block.N.K.attention.self.layer_norm.* segformer.stages.N.blocks.K.attention.sequence_reduction.layer_norm.*
    segformer.encoder.block.N.K.layer_norm_1.*           segformer.stages.N.blocks.K.layernorm_before.*
    segformer.encoder.block.N.K.layer_norm_2.*           segformer.stages.N.blocks.K.layernorm_after.*
    segformer.encoder.block.N.K.mlp.dense1.*             segformer.stages.N.blocks.K.mlp.fc1.*
    segformer.encoder.block.N.K.mlp.dense2.*             segformer.stages.N.blocks.K.mlp.fc2.*
    decode_head.linear_c.N.*                             decode_head.linear_projections.N.*
    """

    def _remap_one(old_key: str) -> str:
        has_model_prefix = old_key.startswith("model.")
        # Strip 'model.' prefix temporarily for remapping
        k = old_key[len("model."):] if has_model_prefix else old_key

        # decode_head.linear_c.N.* → decode_head.linear_projections.N.*
        k = re.sub(
            r"^decode_head\.linear_c\.(\d+)\.",
            lambda m: f"decode_head.linear_projections.{m.group(1)}.",
            k,
        )

        # segformer.encoder.patch_embeddings.N.* → segformer.stages.N.patch_embeddings.*
        k = re.sub(
            r"^segformer\.encoder\.patch_embeddings\.(\d+)\.",
            lambda m: f"segformer.stages.{m.group(1)}.patch_embeddings.",
            k,
        )

        # segformer.encoder.layer_norm.N.* → segformer.stages.N.layer_norm.*
        k = re.sub(
            r"^segformer\.encoder\.layer_norm\.(\d+)\.",
            lambda m: f"segformer.stages.{m.group(1)}.layer_norm.",
            k,
        )

        # segformer.encoder.block.N.K.* → segformer.stages.N.blocks.K.*  (with sub-key renames)
        def _remap_block(m: re.Match) -> str:
            n, blk, rest = m.group(1), m.group(2), m.group(3)
            p = f"segformer.stages.{n}.blocks.{blk}."

            # Attention projection renames
            if rest.startswith("attention.self.query."):
                return p + "attention.q_proj." + rest[len("attention.self.query."):]
            if rest.startswith("attention.self.key."):
                return p + "attention.k_proj." + rest[len("attention.self.key."):]
            if rest.startswith("attention.self.value."):
                return p + "attention.v_proj." + rest[len("attention.self.value."):]
            if rest.startswith("attention.output.dense."):
                return p + "attention.o_proj." + rest[len("attention.output.dense."):]
            # Sequence-reduction renames
            if rest.startswith("attention.self.sr."):
                return (
                    p
                    + "attention.sequence_reduction.sequence_reduction."
                    + rest[len("attention.self.sr."):]
                )
            if rest.startswith("attention.self.layer_norm."):
                return (
                    p
                    + "attention.sequence_reduction.layer_norm."
                    + rest[len("attention.self.layer_norm."):]
                )
            # LayerNorm renames
            if rest.startswith("layer_norm_1."):
                return p + "layernorm_before." + rest[len("layer_norm_1."):]
            if rest.startswith("layer_norm_2."):
                return p + "layernorm_after." + rest[len("layer_norm_2."):]
            # MLP renames: dense1 → fc1, dense2 → fc2
            if rest.startswith("mlp."):
                mlp_rest = rest[len("mlp."):]
                mlp_rest = re.sub(r"^dense1\.", "fc1.", mlp_rest)
                mlp_rest = re.sub(r"^dense2\.", "fc2.", mlp_rest)
                return p + "mlp." + mlp_rest
            # Fallback — pass through unchanged
            return p + rest

        k = re.sub(
            r"^segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)$",
            _remap_block,
            k,
        )
        return "model." + k if has_model_prefix else k

    return {_remap_one(k): v for k, v in state_dict.items()}


class TreeSegformer(nn.Module):
    """Wraps HuggingFace's SegformerForSemanticSegmentation so the rest of
    the codebase depends on a stable `forward(x) -> logits [B,C,H,W]`
    interface rather than the HF ModelOutput object directly.

    The model is built entirely from a local SegformerConfig (no network
    access required); only the number of output classes is taken from the
    caller.
    """

    def __init__(self, pretrained_checkpoint: str, num_classes: int):
        """
        Parameters
        ----------
        pretrained_checkpoint : str
            Kept for API compatibility (logged, not used for construction).
        num_classes : int
            Number of segmentation classes (3 for background / canopy / trunk).
        """
        super().__init__()
        self.num_classes = num_classes

        seg_config = SegformerConfig(
            **_SEGFORMER_B5_CONFIG,
            num_labels=num_classes,
            label2id={str(i): i for i in range(num_classes)},
            id2label={str(i): str(i) for i in range(num_classes)},
        )
        self.model = SegformerForSemanticSegmentation(seg_config)
        logger.info(
            f"TreeSegformer built from local SegformerConfig-B5 "
            f"(num_classes={num_classes}, params={sum(p.numel() for p in self.parameters()):,})"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]
        logits = nn.functional.interpolate(
            logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits


def build_model(cfg) -> nn.Module:
    """Builds the SegFormer-B5 model for training or inference.

    Constructs the model from a locally-defined SegformerConfig; no
    HuggingFace network access is required.  The pretrained_checkpoint
    field in config is retained for documentation/logging purposes only
    and is not used during construction.
    """
    seg_cfg = cfg.model.segformer
    logger.info(
        f"Building SegFormer-B5 from local config "
        f"(num_classes={seg_cfg.num_classes}) — no network call."
    )
    model = TreeSegformer(
        pretrained_checkpoint=seg_cfg.pretrained_checkpoint,  # logged only
        num_classes=seg_cfg.num_classes,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SAM2Wrapper:
    """Optional SAM2 box-prompted segmenter. Lazily imports sam2 so the rest
    of the module works without it installed when SAM2 features are unused
    (config.model.sam2.enabled_for_* both False).
    """

    def __init__(self, checkpoint: str, model_config: str, device: str = "cuda"):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "The 'sam2' package is required for SAM2Wrapper. Install it per "
                "https://github.com/facebookresearch/sam2"
            ) from e

        sam2_model = build_sam2(model_config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        logger.info(f"SAM2Wrapper ready (checkpoint={checkpoint}, device={device})")

    def segment_with_box(self, image_rgb, box) -> "np.ndarray":
        """box: [xmin, ymin, xmax, ymax] in absolute pixel coordinates.
        Returns a binary (H, W) uint8 mask."""
        import numpy as np

        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(box=np.array(box), multimask_output=False)
        return masks[0].astype(np.uint8)
