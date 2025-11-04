import os
import random
import pandas as pd
from pathlib import Path

def split_by_icao_counts(output_dir_path, test_sum_target=100, min_test_sum=80, min_test_icaos=2, val_ratio=0.2, seed=42):
    """Split ICAO groups into test / train / val based on their file counts."""
    random.seed(seed)
    output_dir_path = Path(output_dir_path)

    # 1️⃣ Extraire les ICAO depuis les noms de fichiers
    files = [f for f in os.listdir(output_dir_path) if "_" in f]
    icaos = [f.split("_")[3] for f in files]
    df = pd.DataFrame({"file": files, "icao": icaos})

    counts = df["icao"].value_counts().sample(frac=1, random_state=seed)

    # 2️⃣ Construire le set test avec contraintes
    test_icaos, total = [], 0
    for icao, count in counts.items():
        if total < test_sum_target or len(test_icaos) < min_test_icaos or total < min_test_sum:
            test_icaos.append(icao)
            total += count
        else:
            break

    remaining = counts.drop(test_icaos)
    total_remaining = remaining.sum()

    # 3️⃣ Split 80/20 sur le reste
    val_target = int(total_remaining * val_ratio)
    val_icaos, running_sum = [], 0
    for icao, count in remaining.sample(frac=1, random_state=seed).items():
        if running_sum + count <= val_target:
            val_icaos.append(icao)
            running_sum += count
        else:
            break

    # 4️⃣ Assigner split à chaque fichier
    def assign_split(icao):
        if icao in test_icaos:
            return "test"
        elif icao in val_icaos:
            return "val"
        else:
            return "train"

    df["split"] = df["icao"].apply(assign_split)
    df["aircraft_type"] = output_dir_path.name
    df["filepath"] = df["file"].apply(lambda f: str(output_dir_path / f))

    # Logs clairs
    print(f"{output_dir_path.name}: test={len(df[df.split=='test'])}, val={len(df[df.split=='val'])}, train={len(df[df.split=='train'])}")

    return df[["filepath", "aircraft_type", "split"]]


def make_global_split_csv(output_dir_path):
    """
    Apply the ICAO-based split for each aircraft folder and
    save the final combined CSV at the root of output_dir_path.
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

