#!/usr/bin/env python3
"""
make_splits.py
---------------
Creates stratified train/val/test splits from the unified processed dataset
(data/processed/<species>/*.jpg) and writes CSV manifests consumed by
data/dataset.py.

Stratification ensures every species is proportionally represented in all
three splits, which matters here because per-species image counts vary
widely across merged sources (20s to 1000s of images).

Usage:
    python data/make_splits.py --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("make_splits")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_records(processed_dir: Path) -> pd.DataFrame:
    records = []
    for class_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        species = class_dir.name
        for img_path in class_dir.glob("*.jpg"):
            records.append({"filepath": str(img_path.resolve()), "species": species})
    return pd.DataFrame(records)


def stratified_split(df: pd.DataFrame, train_frac: float, val_frac: float,
                      test_frac: float, seed: int) -> tuple:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    # First split off test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_frac, stratify=df["species"], random_state=seed
    )
    # Then split remaining into train/val
    relative_val_frac = val_frac / (train_frac + val_frac)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_frac,
        stratify=train_val_df["species"], random_state=seed
    )
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = collect_records(processed_dir)
    if df.empty:
        raise RuntimeError(
            f"No images found in {processed_dir}. Run prepare_dataset.py first."
        )

    # Drop classes with too few images to stratify meaningfully
    min_per_class = cfg["data"]["min_images_per_class"]
    counts = df["species"].value_counts()
    valid_classes = counts[counts >= max(min_per_class, 3)].index
    dropped = set(counts.index) - set(valid_classes)
    if dropped:
        logger.warning(f"Dropping classes with too few images to split: {sorted(dropped)}")
    df = df[df["species"].isin(valid_classes)].reset_index(drop=True)

    train_df, val_df, test_df = stratified_split(
        df,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
        cfg["data"]["test_split"],
        cfg["project"]["seed"],
    )

    # Build canonical label map (sorted for reproducibility)
    classes = sorted(df["species"].unique())
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    for split_df in (train_df, val_df, test_df):
        split_df["label"] = split_df["species"].map(label_map)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    label_map_df = pd.DataFrame(
        {"label": list(label_map.values()), "species": list(label_map.keys())}
    ).sort_values("label")
    label_map_df.to_csv(splits_dir / "label_map.csv", index=False)

    logger.info("=" * 60)
    logger.info(f"Total images: {len(df)} | Classes: {len(classes)}")
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"Splits written to: {splits_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
