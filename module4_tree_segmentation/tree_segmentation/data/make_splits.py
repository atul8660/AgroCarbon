#!/usr/bin/env python3
"""
make_splits.py
---------------
Creates train/val/test splits from data/processed/manifest.csv, stratified
by source dataset (so part-supervised samples from Urban Street Tree aren't
concentrated only in train), and reports per-split image statistics.

Usage:
    python data/make_splits.py --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("make_splits")


def load_config(config_path: str):
    return OmegaConf.load(config_path)


def stratified_split_by_source(df: pd.DataFrame, train_frac: float, val_frac: float,
                                test_frac: float, seed: int) -> tuple:
    train_rows, val_rows, test_rows = [], [], []
    for source_name, group in df.groupby("source"):
        if len(group) < 3:
            train_rows.append(group)
            continue
        train_val, test = train_test_split(group, test_size=test_frac, random_state=seed)
        relative_val = val_frac / (train_frac + val_frac)
        train, val = train_test_split(train_val, test_size=relative_val, random_state=seed)
        train_rows.append(train)
        val_rows.append(val)
        test_rows.append(test)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    return train_df, val_df, test_df


def compute_split_stats(df: pd.DataFrame) -> dict:
    coverages, canopy_ratios, trunk_ratios = [], [], []
    for _, row in df.iterrows():
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        total = mask.size
        fg = (mask > 0).sum()
        canopy = (mask == 1).sum()
        trunk = (mask == 2).sum()
        coverages.append(fg / total)
        canopy_ratios.append(canopy / total)
        trunk_ratios.append(trunk / total)

    return {
        "num_samples": len(df),
        "num_part_supervised": int(df["has_part_supervision"].astype(bool).sum()) if len(df) else 0,
        "source_counts": df["source"].value_counts().to_dict() if len(df) else {},
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_canopy_ratio": float(np.mean(canopy_ratios)) if canopy_ratios else 0.0,
        "mean_trunk_ratio": float(np.mean(trunk_ratios)) if trunk_ratios else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg.paths.processed_data_dir)
    splits_dir = Path(cfg.paths.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError(f"{manifest_path} not found. Run prepare_dataset.py first.")

    df = pd.read_csv(manifest_path)
    df["has_part_supervision"] = df["has_part_supervision"].astype(bool)

    train_df, val_df, test_df = stratified_split_by_source(
        df, cfg.data.train_split, cfg.data.val_split, cfg.data.test_split, cfg.project.seed
    )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    logger.info("=" * 60)
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        stats = compute_split_stats(split_df)
        logger.info(
            f"[{split_name}] N={stats['num_samples']} | part-supervised={stats['num_part_supervised']} | "
            f"mean_coverage={stats['mean_coverage']:.3f} | mean_canopy={stats['mean_canopy_ratio']:.3f} | "
            f"mean_trunk={stats['mean_trunk_ratio']:.3f}"
        )
        logger.info(f"[{split_name}] Per-source counts: {stats['source_counts']}")
    logger.info(f"Splits written to: {splits_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
