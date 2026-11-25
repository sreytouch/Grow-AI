"""
GrowAI Phase 2 — Data Quality Report
======================================
Authors: Madike Asia, Sreytouch Lang (Jessica)
Thullen Lab, July 2026

Run after every pre-processing pipeline to validate output before modeling.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def data_quality_report(df: pd.DataFrame, dataset_name: str, save_path: str = None) -> dict:
    """
    Print and optionally save a data quality summary for a cleaned DataFrame.

    Parameters
    ----------
    df           : cleaned pandas DataFrame
    dataset_name : human-readable label (e.g. 'USDA NASS Quick Stats')
    save_path    : optional path to save the report as a JSON file

    Returns
    -------
    dict with all quality metrics
    """
    sep = "=" * 60

    total_missing  = int(df.isnull().sum().sum())
    total_dups     = int(df.duplicated().sum())
    memory_mb      = df.memory_usage(deep=True).sum() / 1024 ** 2
    missing_by_col = (df.isnull().mean() * 100).sort_values(ascending=False)
    top_missing    = missing_by_col[missing_by_col > 0].head(5)
    dtype_counts   = df.dtypes.value_counts().to_dict()

    print(f"\n{sep}")
    print(f"  DATA QUALITY REPORT: {dataset_name}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)
    print(f"  Rows × Columns  : {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Missing values  : {total_missing:,} total cells")
    print(f"  Duplicate rows  : {total_dups:,}")
    print(f"  Memory usage    : {memory_mb:.2f} MB")
    print(f"\n  Column dtypes:")
    for dtype, count in dtype_counts.items():
        print(f"    {str(dtype):<20} {count} columns")

    if len(top_missing) > 0:
        print(f"\n  Columns with missing values (top 5):")
        for col, pct in top_missing.items():
            print(f"    {col:<35} {pct:.1f}%")
    else:
        print(f"\n  No missing values detected ✓")

    print(f"\n  Sample (first 3 rows):")
    print(df.head(3).to_string(max_cols=6))
    print(f"{sep}\n")

    report = {
        "dataset":       dataset_name,
        "generated_at":  datetime.now().isoformat(),
        "shape":         list(df.shape),
        "total_missing": total_missing,
        "duplicate_rows": total_dups,
        "memory_mb":     round(memory_mb, 3),
        "dtype_counts":  {str(k): v for k, v in dtype_counts.items()},
        "missing_pct_top5": top_missing.round(2).to_dict(),
    }

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {path}")

    return report


def image_dataset_report(class_dirs: list, dataset_name: str = "PlantVillage") -> dict:
    """
    Print a quality summary for an image classification dataset.

    Parameters
    ----------
    class_dirs : list of Path objects, one per class folder
    """
    sep = "=" * 60
    class_counts = {d.name: len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
                    for d in class_dirs if d.is_dir()}
    total = sum(class_counts.values())
    min_class = min(class_counts, key=class_counts.get) if class_counts else "N/A"
    max_class = max(class_counts, key=class_counts.get) if class_counts else "N/A"

    print(f"\n{sep}")
    print(f"  IMAGE DATASET REPORT: {dataset_name}")
    print(sep)
    print(f"  Total classes   : {len(class_counts)}")
    print(f"  Total images    : {total:,}")
    print(f"  Smallest class  : {min_class} ({class_counts.get(min_class, 0):,} images)")
    print(f"  Largest class   : {max_class} ({class_counts.get(max_class, 0):,} images)")
    print(f"  Avg per class   : {total // max(len(class_counts), 1):,}")
    print(f"{sep}\n")

    return {
        "dataset":     dataset_name,
        "num_classes": len(class_counts),
        "total_images": total,
        "class_counts": class_counts,
    }


if __name__ == "__main__":
    # Quick self-test with dummy data
    import numpy as np
    df_test = pd.DataFrame({
        "crop":    ["tomato", "lettuce", None, "pepper"],
        "yield":   [4.2, 3.1, 2.8, None],
        "state":   ["CA", "CA", "TX", "TX"],
    })
    data_quality_report(df_test, "Test Dataset")
