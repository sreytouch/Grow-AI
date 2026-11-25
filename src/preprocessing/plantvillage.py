"""
GrowAI Phase 2 — PlantVillage Image Pre-Processing
====================================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Pipeline:
  1. Class filtering  — retain home-garden-relevant classes only
  2. Deduplication    — perceptual hash (pHash, threshold=8)
  3. Resize           — 224×224 px (MobileNetV3 input)
  4. Normalization    — ImageNet mean/std
  5. Augmentation     — flip, rotation, brightness, Gaussian blur (train set only)
  6. Stratified split — 70/15/15 by class label
  7. Export           — TFRecord or PyTorch ImageFolder structure

Reference:
  Mohanty et al. (2016). Frontiers in Plant Science.
  MDPI Sensors (Jan 2025). Plant Leaf Disease Detection: A Multi-Dataset Approach.
  arXiv:2408.13154 — stratified 70/15/15 split protocol.
"""

import shutil
from collections import Counter
from math import ceil
from pathlib import Path

from PIL import Image
import imagehash
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.utils.config import (
    PLANTVILLAGE_DIR, DATA_PROCESSED, IMAGE_SIZE,
    PHASH_THRESHOLD, HOME_GARDEN_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ── Step 1: Discover and filter classes ───────────────────────────────────────

def get_filtered_classes(root: Path) -> list[Path]:
    """Return class directories matching HOME_GARDEN_CLASSES."""
    all_classes = [d for d in root.iterdir() if d.is_dir()]
    filtered = [d for d in all_classes
                if any(hg in d.name for hg in HOME_GARDEN_CLASSES)]
    print(f"Total classes found : {len(all_classes)}")
    print(f"After home-garden filter: {len(filtered)}")
    return filtered


# ── Step 2: Deduplicate ───────────────────────────────────────────────────────

def deduplicate_images(image_paths: list[Path], threshold: int = PHASH_THRESHOLD) -> list[Path]:
    """
    Remove near-duplicate images using perceptual hash (pHash).

    Parameters
    ----------
    image_paths : list of image file paths
    threshold   : max pHash Hamming distance to consider images as duplicates

    Returns
    -------
    list of unique image paths
    """
    seen_hashes: dict = {}
    unique, duplicates = [], []

    for path in tqdm(image_paths, desc="Deduplicating"):
        try:
            img = Image.open(path).convert("RGB")
            h = imagehash.phash(img)
            if not any(abs(h - seen) <= threshold for seen in seen_hashes):
                seen_hashes[h] = path
                unique.append(path)
            else:
                duplicates.append(path)
        except Exception as e:
            print(f"  [WARNING] Could not process {path.name}: {e}")

    print(f"Removed {len(duplicates):,} duplicates — kept {len(unique):,} unique images")
    return unique


# ── Step 3–4: Transforms ──────────────────────────────────────────────────────

def get_transforms():
    """
    Returns torchvision transforms for train and eval sets.
    Augmentation applied to training set only (prevents data leakage).
    """
    try:
        from torchvision import transforms

        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        return train_transform, eval_transform

    except ImportError:
        print("[WARNING] torchvision not installed. Run: pip install torchvision")
        return None, None


# ── Step 5: Stratified split ──────────────────────────────────────────────────

def stratified_split(image_paths: list[Path], labels: list[str]):
    """
    Split into train/val/test with 70/15/15 ratio, stratified by class.
    Protocol from arXiv:2408.13154.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(1 - TRAIN_RATIO),
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )
    print(f"Split — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def filter_sparse_classes(image_paths: list[Path], labels: list[str]):
    """
    Remove classes that are too small for a 70/15/15 stratified split.

    A class needs at least 7 samples to guarantee that the temporary holdout
    created by the first 70/30 split still contains at least 2 items for the
    second stratified val/test split.
    """
    min_samples = max(2, ceil(2 / (1 - TRAIN_RATIO)))
    label_counts = Counter(labels)
    dropped = {
        label: count for label, count in label_counts.items()
        if count < min_samples
    }

    if dropped:
        print(
            "[WARNING] Dropping sparse classes before split: "
            + ", ".join(f"{label} ({count})" for label, count in sorted(dropped.items()))
        )

    kept_pairs = [
        (path, label)
        for path, label in zip(image_paths, labels)
        if label_counts[label] >= min_samples
    ]
    kept_paths = [path for path, _ in kept_pairs]
    kept_labels = [label for _, label in kept_pairs]
    return kept_paths, kept_labels, dropped


# ── Step 6: Copy into split folders ──────────────────────────────────────────

def export_split(split_data: dict, output_root: Path):
    """
    Copy images into train/val/test subfolders (PyTorch ImageFolder format).

    Parameters
    ----------
    split_data  : {'train': (paths, labels), 'val': ..., 'test': ...}
    output_root : destination root directory
    """
    if output_root.exists():
        shutil.rmtree(output_root)

    for split_name, (paths, labels) in split_data.items():
        for path, label in tqdm(zip(paths, labels),
                                total=len(paths),
                                desc=f"Copying {split_name}"):
            dest = output_root / split_name / label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)
    print(f"Export complete → {output_root}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(plantvillage_dir: Path = PLANTVILLAGE_DIR,
                 output_dir: Path = DATA_PROCESSED / "plantvillage_split",
                 deduplicate: bool = True):
    """
    Run the complete PlantVillage pre-processing pipeline.
    """
    print("\n=== PlantVillage Pre-Processing Pipeline ===\n")

    # 1. Filter classes
    class_dirs = get_filtered_classes(plantvillage_dir)

    # Collect all image paths and labels
    all_paths, all_labels = [], []
    for class_dir in class_dirs:
        imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        all_paths.extend(imgs)
        all_labels.extend([class_dir.name] * len(imgs))

    print(f"Total images before dedup: {len(all_paths):,}")

    # 2. Deduplicate
    if deduplicate:
        all_paths = deduplicate_images(all_paths)
        # Rebuild labels after dedup
        all_labels = [p.parent.name for p in all_paths]

    all_paths, all_labels, dropped = filter_sparse_classes(all_paths, all_labels)
    if dropped:
        print(f"Classes retained after sparse-class filter: {len(set(all_labels))}")
    if not all_paths:
        print("[ERROR] No images left after sparse-class filtering.")
        return {}

    # 3. Stratified split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(all_paths, all_labels)

    # 4. Export
    split_data = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }
    export_split(split_data, output_dir)

    print("\n=== PlantVillage Pipeline Complete ===")
    return split_data


if __name__ == "__main__":
    run_pipeline()
