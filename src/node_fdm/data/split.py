#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset splitting utilities based on ICAO file counts."""

import os
import random
import pandas as pd
from pathlib import Path


def split_by_icao_counts(
    output_dir_path: Path,
    test_sum_target: int = 100,
    min_test_sum: int = 80,
    min_test_icaos: int = 2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Split ICAO groups into test/train/val based on their file counts.

    Args:
        output_dir_path: Directory containing per-flight files.
        test_sum_target: Target total count of files in the test split.
        min_test_sum: Minimum total files required in the test split.
        min_test_icaos: Minimum number of ICAO groups to include in test split.
        val_ratio: Fraction of remaining files to assign to validation.
        seed: Random seed for shuffling.

    Returns:
        DataFrame with filepath, aircraft type, and split assignments.
    """
    random.seed(seed)
    output_dir_path = Path(output_dir_path)

    files = [f for f in os.listdir(output_dir_path) if "_" in f]
    icaos = [f.split("_")[3] for f in files]
    df = pd.DataFrame({"file": files, "icao": icaos})

    counts = df["icao"].value_counts().sample(frac=1, random_state=seed)

    test_icaos, total = [], 0
    for icao, count in counts.items():
        if (
            total < test_sum_target
            or len(test_icaos) < min_test_icaos
            or total < min_test_sum
        ):
            test_icaos.append(icao)
            total += count
        else:
            break

    remaining = counts.drop(test_icaos)
    total_remaining = remaining.sum()

    val_target = int(total_remaining * val_ratio)
    val_icaos, running_sum = [], 0
    for icao, count in remaining.sample(frac=1, random_state=seed).items():
        if running_sum + count <= val_target:
            val_icaos.append(icao)
            running_sum += count
        else:
            break

    def assign_split(icao: str) -> str:
        """Map an ICAO code to its dataset split."""
        if icao in test_icaos:
            return "test"
        elif icao in val_icaos:
            return "val"
        else:
            return "train"

    df["split"] = df["icao"].apply(assign_split)
    df["aircraft_type"] = output_dir_path.name
    df["filepath"] = df["file"].apply(lambda f: str(output_dir_path / f))

    print(
        f"{output_dir_path.name}: test={len(df[df.split=='test'])}, val={len(df[df.split=='val'])}, train={len(df[df.split=='train'])}"
    )

    return df[["filepath", "aircraft_type", "split"]]


def make_global_split_csv(output_dir_path: Path) -> pd.DataFrame:
    """Apply the ICAO-based split for each aircraft folder and combine results.

    Args:
        output_dir_path: Root directory containing aircraft subfolders.

    Returns:
        Combined split DataFrame saved to `dataset_split.csv`.
    """
    output_dir_path = Path(output_dir_path)
    all_dfs = []

    for acft in sorted(os.listdir(output_dir_path)):
        acft_path = output_dir_path / acft
        if acft_path.is_dir():
            df_split = split_by_icao_counts(acft_path)
            all_dfs.append(df_split)

    final_df = pd.concat(all_dfs, ignore_index=True)

    csv_path = output_dir_path / "dataset_split.csv"
    final_df.to_csv(csv_path, index=False)

    print(f"\n✅ Split summary saved to: {csv_path}")
    print(final_df["split"].value_counts())
    return final_df
