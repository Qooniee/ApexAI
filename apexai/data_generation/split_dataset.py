#!/usr/bin/env python3
r"""Split dataset into train/valid/test sets by vehicle ID.

This script supports two modes:
1. Copy mode: Split already-generated X/y training files
2. Generate mode: Generate X/y files from lap data while splitting

Usage:
    # Copy mode (default): Split existing X_train/y_train files
    python split_dataset.py --input datasets/training_dataset_10Hz \
        --output datasets/drivingdatasets/input

    # Generate mode: Generate X/y from lap files while splitting
    python split_dataset.py --input datasets/preprocessed_10Hz/VIR \
        --output datasets/drivingdatasets/input --generate-from-laps
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_vehicle_files_from_training(
    source_x_dir: Path, source_y_dir: Path
) -> dict[str, list[tuple[Path, Path]]]:
    """Group pre-generated X/y training files by vehicle ID.

    Args:
        source_x_dir: Source directory containing X_train files.
        source_y_dir: Source directory containing y_train files.

    Returns:
        Dictionary mapping vehicle_id to list of (x_file, y_file) tuples.
    """
    vehicle_files = defaultdict(list)

    # Get all X files and find corresponding y files
    for x_file in sorted(source_x_dir.glob("*_X_train.csv")):
        # Extract vehicle ID from filename
        # Format: VIR_R1_vehicle_GR86_002_2_lap_001_X_train.csv
        parts = x_file.stem.split("_")
        if "vehicle" not in parts or "lap" not in parts:
            print(f"Warning: Unexpected filename format: {x_file.name}")
            continue

        vehicle_idx = parts.index("vehicle")
        lap_idx = parts.index("lap")
        vehicle_id = "_".join(parts[vehicle_idx + 1 : lap_idx])

        # Find corresponding y file
        y_filename = x_file.name.replace("_X_train.csv", "_y_train.csv")
        y_file = source_y_dir / y_filename

        if y_file.exists():
            vehicle_files[vehicle_id].append((x_file, y_file))
        else:
            print(f"Warning: Missing y file for {x_file.name}")

    return vehicle_files


def get_vehicle_files_from_laps(lap_dirs: list[Path]) -> dict[str, list[Path]]:
    """Group lap CSV files by vehicle ID.

    Args:
        lap_dirs: List of directories containing lap CSV files (e.g., R1/, R2/).

    Returns:
        Dictionary mapping vehicle_id to list of lap file paths.
    """
    vehicle_files = defaultdict(list)

    for lap_dir in lap_dirs:
        # Get all CSV files (skip metadata.json)
        for lap_file in sorted(lap_dir.glob("*.csv")):
            # Extract vehicle ID from filename
            # Format: VIR_R1_vehicle_GR86_002_2_lap_001.csv
            parts = lap_file.stem.split("_")

            if "vehicle" not in parts or "lap" not in parts:
                print(f"Warning: Skipping file with unexpected format: {lap_file.name}")
                continue

            vehicle_idx = parts.index("vehicle")
            lap_idx = parts.index("lap")
            vehicle_id = "_".join(parts[vehicle_idx + 1 : lap_idx])

            vehicle_files[vehicle_id].append(lap_file)

    return vehicle_files


def split_lap_to_xy(
    lap_file: Path,
    output_x_dir: Path,
    output_y_dir: Path,
    split_name: str,
    target_label: str = "vehicle_label",
) -> tuple[Path, Path]:
    """Split a single lap file into X and y files.

    Args:
        lap_file: Path to lap CSV file.
        output_x_dir: Directory to save X features.
        output_y_dir: Directory to save y labels.
        split_name: Name of split ("train", "valid", "test").
        target_label: Column name for prediction target.

    Returns:
        Tuple of (x_file_path, y_file_path).
    """
    # Load lap data
    df = pd.read_csv(lap_file)

    # Feature columns (exclude metadata and target)
    metadata_cols = ["timestamp", "vehicle_id", "vehicle_label", "lap", "elapsed_time"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Extract features (x) and target (y)
    x = df[feature_cols]
    y = df[[target_label]]

    # Generate output filenames
    base_name = lap_file.stem
    x_filename = f"{base_name}_X_{split_name}.csv"
    y_filename = f"{base_name}_y_{split_name}.csv"

    x_path = output_x_dir / x_filename
    y_path = output_y_dir / y_filename

    # Save files
    x.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    return x_path, y_path


def split_and_copy_files(
    vehicle_files: dict[str, list[tuple[Path, Path]]],
    dest_base: Path,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = False,
    filter_last_lap: bool = False,
    random_seed: int = 42,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Split pre-generated X/y files by vehicle ID and copy to destination.

    Args:
        vehicle_files: Dictionary mapping vehicle_id to (x_file, y_file) tuples.
        dest_base: Base destination directory containing train/valid/test subdirs.
        train_ratio: Ratio of files for training.
        valid_ratio: Ratio of files for validation.
        test_ratio: Ratio of files for testing.
        shuffle: If True, shuffle laps before splitting (recommended for temporal bias).
        filter_last_lap: If True, remove last lap per vehicle (out laps).
        random_seed: Random seed for reproducibility (default: 42).

    Returns:
        Statistics dictionary with file counts per split.
    """
    import numpy as np

    stats = {"train": 0, "valid": 0, "test": 0}
    vehicle_stats = {}

    for vehicle_id, file_pairs in sorted(vehicle_files.items()):
        # Filter last lap if requested
        if filter_last_lap and len(file_pairs) > 1:
            file_pairs = file_pairs[:-1]  # Remove last lap

        n_files = len(file_pairs)

        # Shuffle if requested
        if shuffle:
            rng = np.random.RandomState(random_seed)
            indices = rng.permutation(n_files)
            file_pairs = [file_pairs[i] for i in indices]

        # Calculate split indices
        n_train = int(n_files * train_ratio)
        n_valid = int(n_files * valid_ratio)
        n_test = n_files - n_train - n_valid

        print(f"\nVehicle {vehicle_id}: {n_files} laps")
        print(f"  Train: {n_train}, Valid: {n_valid}, Test: {n_test}")

        # Split files
        train_files = file_pairs[:n_train]
        valid_files = file_pairs[n_train : n_train + n_valid]
        test_files = file_pairs[n_train + n_valid :]

        # Copy files to respective directories
        for x_file, y_file in train_files:
            shutil.copy2(x_file, dest_base / "train" / "x_train" / x_file.name)
            shutil.copy2(y_file, dest_base / "train" / "y_train" / y_file.name)
            stats["train"] += 1

        for x_file, y_file in valid_files:
            shutil.copy2(x_file, dest_base / "valid" / "x_valid" / x_file.name)
            shutil.copy2(y_file, dest_base / "valid" / "y_valid" / y_file.name)
            stats["valid"] += 1

        for x_file, y_file in test_files:
            shutil.copy2(x_file, dest_base / "test" / "x_test" / x_file.name)
            shutil.copy2(y_file, dest_base / "test" / "y_test" / y_file.name)
            stats["test"] += 1

        vehicle_stats[vehicle_id] = {
            "total": n_files,
            "train": n_train,
            "valid": n_valid,
            "test": n_test,
        }

    return stats, vehicle_stats


def split_and_generate_files(
    vehicle_files: dict[str, list[Path]],
    dest_base: Path,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_label: str = "vehicle_label",
    shuffle: bool = False,
    filter_last_lap: bool = False,
    random_seed: int = 42,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Split lap files by vehicle ID and generate X/y files.

    Args:
        vehicle_files: Dictionary mapping vehicle_id to lap file paths.
        dest_base: Base destination directory containing train/valid/test subdirs.
        train_ratio: Ratio of files for training.
        valid_ratio: Ratio of files for validation.
        test_ratio: Ratio of files for testing.
        target_label: Column name for prediction target.
        shuffle: If True, shuffle laps before splitting (recommended for temporal bias).
        filter_last_lap: If True, remove last lap per vehicle (out laps).
        random_seed: Random seed for reproducibility (default: 42).

    Returns:
        Statistics dictionary with file counts per split.
    """
    import numpy as np

    stats = {"train": 0, "valid": 0, "test": 0}
    vehicle_stats = {}

    # Create output directories
    for split in ["train", "valid", "test"]:
        x_dir = dest_base / split / f"x_{split}"
        y_dir = dest_base / split / f"y_{split}"
        x_dir.mkdir(parents=True, exist_ok=True)
        y_dir.mkdir(parents=True, exist_ok=True)

    print("\nProcessing vehicles:")
    for vehicle_id, lap_files in tqdm(sorted(vehicle_files.items())):
        # Filter last lap if requested
        if filter_last_lap and len(lap_files) > 1:
            lap_files = lap_files[:-1]  # Remove last lap

        n_files = len(lap_files)

        # Shuffle if requested
        if shuffle:
            rng = np.random.RandomState(random_seed)
            indices = rng.permutation(n_files)
            lap_files = [lap_files[i] for i in indices]

        # Calculate split indices
        n_train = int(n_files * train_ratio)
        n_valid = int(n_files * valid_ratio)
        n_test = n_files - n_train - n_valid

        # Split files
        train_files = lap_files[:n_train]
        valid_files = lap_files[n_train : n_train + n_valid]
        test_files = lap_files[n_train + n_valid :]

        # Process train files
        for lap_file in train_files:
            split_lap_to_xy(
                lap_file, dest_base / "train" / "x_train", dest_base / "train" / "y_train", "train"
            )
            stats["train"] += 1

        # Process valid files
        for lap_file in valid_files:
            split_lap_to_xy(
                lap_file, dest_base / "valid" / "x_valid", dest_base / "valid" / "y_valid", "valid"
            )
            stats["valid"] += 1

        # Process test files
        for lap_file in test_files:
            split_lap_to_xy(
                lap_file, dest_base / "test" / "x_test", dest_base / "test" / "y_test", "test"
            )
            stats["test"] += 1

        vehicle_stats[vehicle_id] = {
            "total": n_files,
            "train": n_train,
            "valid": n_valid,
            "test": n_test,
        }

    return stats, vehicle_stats


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train/valid/test")
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing dataset files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/drivingdatasets/input",
        help="Output directory for split datasets",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training ratio (default: 0.7)"
    )
    parser.add_argument(
        "--valid-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)"
    )
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument(
        "--generate-from-laps",
        action="store_true",
        help="Generate X/y files from lap data (default: copy existing X/y files)",
    )
    parser.add_argument(
        "--target-label", type=str, default="vehicle_label", help="Target label column name"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle laps before splitting (recommended to avoid temporal bias)",
    )
    parser.add_argument(
        "--filter-last-lap",
        action="store_true",
        help="Filter out last lap per vehicle (removes out laps)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return

    input_dir = Path(args.input)
    dest_base = Path(args.output)

    print("=" * 60)
    ratios_str = f"{args.train_ratio:.0%}/{args.valid_ratio:.0%}/{args.test_ratio:.0%}"
    print(f"Dataset Split: {ratios_str}")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {dest_base}")
    print(f"Mode:   {'Generate from laps' if args.generate_from_laps else 'Copy existing X/y'}")
    print(f"Shuffle: {'Yes' if args.shuffle else 'No'}")
    print(f"Filter last lap: {'Yes' if args.filter_last_lap else 'No'}")
    if args.shuffle or args.filter_last_lap:
        print(f"Random seed: {args.random_seed}")

    if args.generate_from_laps:
        # Generate mode: Process lap files
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return

        # Find race directories (R1, R2, etc.)
        race_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("R")]

        if not race_dirs:
            print(f"Error: No race directories (R1, R2, ...) found in {input_dir}")
            return

        print(f"\nFound race directories: {[d.name for d in race_dirs]}")

        # Group files by vehicle ID
        print("\nGrouping lap files by vehicle ID...")
        vehicle_files = get_vehicle_files_from_laps(race_dirs)

        print(f"\nFound {len(vehicle_files)} unique vehicles")
        total_laps = sum(len(files) for files in vehicle_files.values())
        print(f"Total lap files: {total_laps}")

        # Split and generate files
        print("\nSplitting and generating X/y files...")
        stats, vehicle_stats = split_and_generate_files(
            vehicle_files,
            dest_base,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            shuffle=args.shuffle,
            filter_last_lap=args.filter_last_lap,
            random_seed=args.random_seed,
        )
    else:
        # Copy mode: Use existing X/y files
        source_x_dir = input_dir / "x_train"
        source_y_dir = input_dir / "y_train"

        # Check source directories exist
        if not source_x_dir.exists() or not source_y_dir.exists():
            print("Error: Source directories not found!")
            print(f"  X: {source_x_dir}")
            print(f"  Y: {source_y_dir}")
            return

        # Create destination directories
        for split in ["train", "valid", "test"]:
            x_dir = dest_base / split / f"x_{split}"
            y_dir = dest_base / split / f"y_{split}"
            x_dir.mkdir(parents=True, exist_ok=True)
            y_dir.mkdir(parents=True, exist_ok=True)

        # Group files by vehicle ID
        print("\nGrouping files by vehicle ID...")
        vehicle_files = get_vehicle_files_from_training(source_x_dir, source_y_dir)

        print(f"\nFound {len(vehicle_files)} unique vehicles")
        total_pairs = sum(len(files) for files in vehicle_files.values())
        print(f"Total file pairs: {total_pairs}")

        # Split and copy files
        print("\nSplitting and copying files...")
        stats, vehicle_stats = split_and_copy_files(
            vehicle_files,
            dest_base,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            shuffle=args.shuffle,
            filter_last_lap=args.filter_last_lap,
            random_seed=args.random_seed,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Split Summary:")
    print("=" * 60)
    total = sum(stats.values())
    train_pct = stats["train"] / total * 100
    valid_pct = stats["valid"] / total * 100
    test_pct = stats["test"] / total * 100
    print(f"Train:      {stats['train']:4d} laps ({train_pct:.1f}%)")
    print(f"Validation: {stats['valid']:4d} laps ({valid_pct:.1f}%)")
    print(f"Test:       {stats['test']:4d} laps ({test_pct:.1f}%)")
    print(f"Total:      {total:4d} laps")

    print("\nPer-vehicle breakdown:")
    print("-" * 60)
    for vehicle_id, v_stats in sorted(vehicle_stats.items()):
        train_cnt = v_stats["train"]
        valid_cnt = v_stats["valid"]
        test_cnt = v_stats["test"]
        total_cnt = v_stats["total"]
        print(
            f"{vehicle_id:20s} Total:{total_cnt:3d} -> "
            f"Train:{train_cnt:3d} Valid:{valid_cnt:3d} Test:{test_cnt:3d}"
        )

    # Save statistics
    stats_file = dest_base / "split_stats.json"
    with stats_file.open("w") as f:
        json.dump(
            {
                "total_laps": total,
                "split_counts": stats,
                "vehicle_breakdown": vehicle_stats,
                "ratios": {
                    "train": args.train_ratio,
                    "valid": args.valid_ratio,
                    "test": args.test_ratio,
                },
                "mode": "generate_from_laps" if args.generate_from_laps else "copy_existing",
            },
            f,
            indent=2,
        )

    print("\n✅ Dataset split completed successfully!")
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
