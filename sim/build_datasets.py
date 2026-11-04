"""
Builds and saves the complete research-grade dataset split for training, validation, and testing.
Supports scaling synthetic flight hours and ingesting manual user flight recordings (CSVs).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sim.generator import MissionGenerator
from sim.attacks import ATTACK_NAMES, SENSOR_KEYS


def ingest_recorded_missions(recorded_dir: Path, min_rows: int = 50):
    """
    Scans a directory for user-recorded flight CSVs and parses valid labeled flights.
    """
    if not recorded_dir.exists():
        return []
    
    csv_files = sorted(list(recorded_dir.glob("*.csv")))
    valid_dfs = []
    
    for f in csv_files:
        if f.name.startswith("unlabeled_"):
            # Skip intentionally unlabeled benchmark test files
            continue
        try:
            df = pd.read_csv(f)
            # Must have at least 50 frames and sensor columns
            if len(df) < min_rows:
                continue
            if "pitot_ias" not in df.columns or "accel_x" not in df.columns:
                continue
            
            # Check for label columns
            if "attack_id" not in df.columns:
                df["attack_id"] = 0
                df["attack_name"] = "Normal"
            if "attack_name" not in df.columns:
                df["attack_name"] = df["attack_id"].map(lambda x: ATTACK_NAMES[int(x)] if int(x) < len(ATTACK_NAMES) else "Unknown")
            
            # Ensure compromised_* columns exist
            for s in SENSOR_KEYS:
                col = f"compromised_{s}"
                if col not in df.columns:
                    df[col] = 0
            
            valid_dfs.append(df)
            print(f"  [+] Ingested custom flight log: {f.name} ({len(df)} frames, {len(df)*0.02:.1f}s)")
        except Exception as e:
            print(f"  [-] Warning: could not parse {f.name}: {e}")
            
    return valid_dfs


def main():
    parser = argparse.ArgumentParser(description="UAV IDS Dataset Builder & Scaler")
    parser.add_argument("--data-dir", type=str, default="data", help="Output directory for datasets")
    parser.add_argument("--train-missions", type=int, default=15, help="Number of synthetic missions per attack class for training (default: 15)")
    parser.add_argument("--val-missions", type=int, default=4, help="Number of synthetic missions per attack class for validation (default: 4)")
    parser.add_argument("--test-missions", type=int, default=4, help="Number of synthetic missions per attack class for testing (default: 4)")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration of each synthetic mission in seconds (default: 30.0)")
    parser.add_argument("--include-recorded", action="store_true", help="Scan and merge manual flight CSVs from data/recorded_missions")
    parser.add_argument("--recorded-dir", type=str, default="data/recorded_missions", help="Directory holding manual flight CSVs")
    args = parser.parse_args()

    data_dir = args.data_dir
    generator = MissionGenerator(config_dir="config")
    
    print(f"============================================================")
    print(f"🚀 BUILDING UAV CYBER-PHYSICAL IDS DATASETS")
    print(f"   Synthetic Config: Train={args.train_missions} missions/class, Val={args.val_missions}, Test={args.test_missions}")
    print(f"   Flight Duration: {args.duration:.1f}s @ 50 Hz (1,500 samples/mission)")
    print(f"============================================================")

    # 1. Generate Synthetic Splits
    train_pq = generator.generate_dataset_split(
        output_dir=data_dir,
        split_name="train",
        missions_per_attack=args.train_missions,
        mission_duration_s=args.duration
    )
    
    val_pq = generator.generate_dataset_split(
        output_dir=data_dir,
        split_name="val",
        missions_per_attack=args.val_missions,
        mission_duration_s=args.duration
    )
    
    test_pq = generator.generate_dataset_split(
        output_dir=data_dir,
        split_name="test",
        missions_per_attack=args.test_missions,
        mission_duration_s=args.duration
    )

    # 2. Ingest Manual Recorded Flights (if requested or available)
    if args.include_recorded:
        recorded_path = Path(args.recorded_dir)
        print(f"\n📂 Scanning for manual flight recordings in: {recorded_path}")
        manual_dfs = ingest_recorded_missions(recorded_path)
        
        if manual_dfs:
            print(f"Successfully loaded {len(manual_dfs)} manual flight recordings.")
            # Load synthetic train and val
            df_train = pd.read_parquet(train_pq)
            df_val = pd.read_parquet(val_pq)
            
            max_id = int(max(df_train["mission_id"].max(), df_val["mission_id"].max())) + 1
            
            # Assign unique mission IDs to manual flights
            train_manuals = []
            val_manuals = []
            for i, m_df in enumerate(manual_dfs):
                m_df = m_df.copy()
                m_df["mission_id"] = max_id + i
                if i % 5 == 0 and len(manual_dfs) > 1:
                    val_manuals.append(m_df)
                else:
                    train_manuals.append(m_df)
            
            if train_manuals:
                df_train = pd.concat([df_train] + train_manuals, ignore_index=True)
                df_train.to_parquet(train_pq, index=False)
                df_train.head(5000).to_csv(Path(data_dir) / "train/train_sample.csv", index=False)
                print(f"  [+] Merged {len(train_manuals)} manual flights into Train Split -> Total samples: {len(df_train):,}")
            
            if val_manuals:
                df_val = pd.concat([df_val] + val_manuals, ignore_index=True)
                df_val.to_parquet(val_pq, index=False)
                df_val.head(5000).to_csv(Path(data_dir) / "val/val_sample.csv", index=False)
                print(f"  [+] Merged {len(val_manuals)} manual flights into Val Split -> Total samples: {len(df_val):,}")
        else:
            print("  [!] No eligible recorded flights found to merge.")

    print(f"\n✓ Dataset preparation complete! Parquet splits saved in '{data_dir}/'")


if __name__ == "__main__":
    main()
