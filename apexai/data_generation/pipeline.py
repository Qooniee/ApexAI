r"""Complete data generation pipeline from raw telemetry to train/valid/test splits.

This script wraps the entire data generation pipeline:
1. Generate 10Hz resampled lap data from raw telemetry
2. Generate X/y training files from lap data
3. Split into train/valid/test datasets

Usage:
    python apexai/data_generation/pipeline.py \
        --raw-data-dir datasets/rawdata/VIR \
        --export-dir datasets/drivingdatasets/input \
        --target-frequency 10.0 \
        --train-ratio 0.7 \
        --valid-ratio 0.15 \
        --test-ratio 0.15

Example:
    # Process VIR race data with 10Hz resampling
    python apexai/data_generation/pipeline.py \
        --raw-data-dir datasets/rawdata/VIR \
        --export-dir datasets/drivingdatasets/input
"""

import json
import shutil
from pathlib import Path
from typing import Any

from apexai.data_generation.generate_training_data import generate_training_files
from apexai.data_generation.preprocess_telemetry import preprocess_telemetry_to_laps


def run_pipeline(
    raw_data_dir: Path | str,
    export_dir: Path | str,
    *,
    target_frequency: float = 10.0,
    original_frequency: float = 23.0,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    filter_last_lap: bool = True,
    shuffle: bool = True,
    random_seed: int = 42,
    intermediate_cleanup: bool = False,
) -> dict[str, Any]:
    """Run complete data generation pipeline.

    Args:
        raw_data_dir: Directory containing raw telemetry CSV files (e.g., datasets/rawdata/VIR)
        export_dir: Final output directory for train/valid/test splits
        target_frequency: Target sampling frequency [Hz] (default: 10.0)
        original_frequency: Original sampling frequency [Hz] (default: 23.0)
        train_ratio: Training set ratio (default: 0.7)
        valid_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        filter_last_lap: If True, remove last lap per vehicle (out laps) (default: True)
        shuffle: If True, shuffle laps before splitting (default: True)
        random_seed: Random seed for reproducibility (default: 42)
        intermediate_cleanup: If True, delete intermediate directories after completion

    Returns:
        Dictionary with pipeline statistics
    """
    raw_data_dir = Path(raw_data_dir)
    export_dir = Path(export_dir)

    # Validate ratios
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        msg = f"Ratios must sum to 1.0, got {train_ratio + valid_ratio + test_ratio}"
        raise ValueError(msg)

    # Define intermediate directories
    track_name = raw_data_dir.name  # e.g., "VIR"
    freq_str = f"preprocessed_{int(target_frequency)}Hz"
    preprocessed_dir = raw_data_dir.parent.parent / freq_str / track_name
    training_dataset_dir = (
        raw_data_dir.parent.parent / f"training_dataset_{int(target_frequency)}Hz"
    )

    print(f"\n{'=' * 80}")
    print("APEXAI DATA GENERATION PIPELINE")
    print(f"{'=' * 80}")
    print(f"Raw data directory:      {raw_data_dir}")
    print(f"Preprocessed directory:  {preprocessed_dir}")
    print(f"Training dataset dir:    {training_dataset_dir}")
    print(f"Final export directory:  {export_dir}")
    print(f"Target frequency:        {target_frequency} Hz")
    print(f"Split ratios:            {train_ratio:.0%} / {valid_ratio:.0%} / {test_ratio:.0%}")
    print(f"Filter last lap:         {'Yes' if filter_last_lap else 'No'}")
    print(f"Shuffle laps:            {'Yes' if shuffle else 'No'}")
    if shuffle:
        print(f"Random seed:             {random_seed}")
    print(f"{'=' * 80}\n")

    pipeline_stats: dict[str, Any] = {
        "raw_data_dir": str(raw_data_dir),
        "export_dir": str(export_dir),
        "target_frequency": target_frequency,
        "original_frequency": original_frequency,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "steps": {},
    }

    # Step 1: Generate 10Hz resampled lap data
    print(f"\n[STEP 1/3] Generating {target_frequency}Hz resampled lap data...")
    print(f"  Input:  {raw_data_dir}")
    print(f"  Output: {preprocessed_dir}\n")

    # Find all race directories (e.g., "Race 1", "Race 2")
    # Handle nested structure: datasets/rawdata/VIR/virginia-international-raceway/VIR/Race X
    search_dirs = [raw_data_dir]

    # Look for Race directories (might be nested)
    for _ in range(3):  # Search up to 3 levels deep
        candidate_dirs = []
        for search_dir in search_dirs:
            if search_dir.is_dir():
                subdirs = list(search_dir.iterdir())
                # Check if this level contains "Race X" directories
                race_dirs_found = [d for d in subdirs if d.is_dir() and d.name.startswith("Race")]
                if race_dirs_found:
                    race_dirs = sorted(race_dirs_found)
                    break
                # Otherwise add subdirectories for next search level
                candidate_dirs.extend(
                    [d for d in subdirs if d.is_dir() and not d.name.startswith("__")]
                )
        else:
            search_dirs = candidate_dirs
            continue
        break
    else:
        msg = f"No race directories found in {raw_data_dir} or its subdirectories"
        raise FileNotFoundError(msg)

    if not race_dirs:
        msg = f"No race directories found in {raw_data_dir}"
        raise FileNotFoundError(msg)

    step1_stats: dict[str, Any] = {"races": {}, "total_laps": 0, "total_files": 0}

    for race_dir in race_dirs:
        race_name = race_dir.name  # e.g., "Race 1"
        print(f"  Processing {race_name}...")

        # Find telemetry CSV file
        telemetry_files = list(race_dir.glob("*_telemetry_data.csv"))
        if not telemetry_files:
            print(f"    Warning: No telemetry file found in {race_dir}")
            continue

        telemetry_path = telemetry_files[0]

        # Determine race number for output directory
        race_num = telemetry_path.stem.split("_")[0]  # e.g., "R1"
        output_dir = preprocessed_dir / race_num

        # Process race with resampling
        race_stats = preprocess_telemetry_to_laps(
            telemetry_path=telemetry_path,
            output_dir=output_dir,
            use_polars=True,
            min_lap_length=50,  # Reduced from 100 since we're resampling
            interpolate_missing=True,
            interpolation_method="linear",
            resample_frequency=target_frequency,
            original_frequency=original_frequency,
        )

        step1_stats["races"][race_name] = race_stats
        step1_stats["total_laps"] += race_stats.get("num_laps", 0)
        step1_stats["total_files"] += len(race_stats.get("output_files", []))

        print(f"    ✓ Generated {race_stats.get('num_laps', 0)} lap files")

    pipeline_stats["steps"]["step1_resampling"] = step1_stats
    print(
        f"\n  ✓ Step 1 complete: {step1_stats['total_laps']} laps, "
        f"{step1_stats['total_files']} files\n"
    )

    # Step 2: Generate X/y training files
    print("\n[STEP 2/3] Generating X/y training files...")
    print(f"  Input:  {preprocessed_dir}")
    print(f"  Output: {training_dataset_dir}\n")

    # Process each race subdirectory
    race_subdirs = sorted([d for d in preprocessed_dir.iterdir() if d.is_dir()])
    step2_stats: dict[str, Any] = {"races": {}, "total_x_files": 0, "total_y_files": 0}

    for race_subdir in race_subdirs:
        race_name = race_subdir.name  # e.g., "R1"
        print(f"  Processing {race_name}...")

        # Generate training files with separate X/y directories
        gen_stats = generate_training_files(
            race_subdir,
            output_dir=training_dataset_dir,
            separate_xy_dirs=True,
        )

        step2_stats["races"][race_name] = gen_stats
        step2_stats["total_x_files"] += gen_stats.get("num_laps", 0)
        step2_stats["total_y_files"] += gen_stats.get("num_laps", 0)

        print(f"    ✓ Generated {gen_stats.get('num_laps', 0)} X/y file pairs")

    pipeline_stats["steps"]["step2_training_files"] = step2_stats
    print(f"\n  ✓ Step 2 complete: {step2_stats['total_x_files']} X/y file pairs\n")

    # Step 3: Split into train/valid/test
    print("\n[STEP 3/3] Splitting into train/valid/test datasets...")
    print(f"  Input:  {training_dataset_dir}")
    print(f"  Output: {export_dir}\n")

    # Import split_dataset functions from apexai.data_generation
    from apexai.data_generation.split_dataset import (
        get_vehicle_files_from_training,
        split_and_copy_files,
    )

    # Find X/y training files
    source_x_dir = training_dataset_dir / "x_train"
    source_y_dir = training_dataset_dir / "y_train"

    if not source_x_dir.exists() or not source_y_dir.exists():
        msg = f"Training files not found in {training_dataset_dir}"
        raise FileNotFoundError(msg)

    # Get vehicle files
    vehicle_files = get_vehicle_files_from_training(source_x_dir, source_y_dir)
    print(f"  Found {len(vehicle_files)} vehicles")

    # Split and copy files
    split_stats, vehicle_stats = split_and_copy_files(
        vehicle_files=vehicle_files,
        dest_base=export_dir,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        shuffle=shuffle,
        filter_last_lap=filter_last_lap,
        random_seed=random_seed,
    )

    # Count vehicles per split
    train_vehicles = sum(1 for v in vehicle_stats.values() if v["train"] > 0)
    valid_vehicles = sum(1 for v in vehicle_stats.values() if v["valid"] > 0)
    test_vehicles = sum(1 for v in vehicle_stats.values() if v["test"] > 0)

    pipeline_stats["steps"]["step3_split"] = {
        "split_stats": split_stats,
        "vehicle_stats": vehicle_stats,
        "train_vehicles": train_vehicles,
        "valid_vehicles": valid_vehicles,
        "test_vehicles": test_vehicles,
    }

    print("\n  ✓ Step 3 complete:")
    print(f"    - Train: {split_stats['train']} files ({train_vehicles} vehicles)")
    print(f"    - Valid: {split_stats['valid']} files ({valid_vehicles} vehicles)")
    print(f"    - Test:  {split_stats['test']} files ({test_vehicles} vehicles)")

    # Save pipeline statistics
    stats_file = export_dir / "pipeline_stats.json"
    with stats_file.open("w") as f:
        json.dump(pipeline_stats, f, indent=2, default=str)

    print(f"\n  Pipeline statistics saved to: {stats_file}")

    # Optional cleanup of intermediate directories
    if intermediate_cleanup:
        print("\n[CLEANUP] Removing intermediate directories...")
        if preprocessed_dir.exists():
            shutil.rmtree(preprocessed_dir)
            print(f"  ✓ Removed {preprocessed_dir}")
        if training_dataset_dir.exists():
            shutil.rmtree(training_dataset_dir)
            print(f"  ✓ Removed {training_dataset_dir}")

    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Final dataset ready at: {export_dir}\n")

    return pipeline_stats


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Complete data generation pipeline from raw telemetry to train/valid/test splits"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--raw-data-dir",
        type=str,
        required=True,
        help="Directory containing raw telemetry CSV files (e.g., datasets/rawdata/VIR)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        required=True,
        help=(
            "Final output directory for train/valid/test splits "
            "(e.g., datasets/drivingdatasets/input)"
        ),
    )
    parser.add_argument(
        "--target-frequency",
        type=float,
        default=10.0,
        help="Target sampling frequency [Hz] (default: 10.0)",
    )
    parser.add_argument(
        "--original-frequency",
        type=float,
        default=23.0,
        help="Original sampling frequency [Hz] (default: 23.0)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--filter-last-lap",
        action="store_true",
        default=True,
        help="Filter out last lap per vehicle (out laps) (default: True)",
    )
    parser.add_argument(
        "--no-filter-last-lap",
        dest="filter_last_lap",
        action="store_false",
        help="Do not filter last lap (keep all laps)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle laps before splitting (default: True)",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Do not shuffle laps (sequential split)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate directories after completion (default: False)",
    )

    args = parser.parse_args()

    # Run pipeline
    try:
        run_pipeline(
            raw_data_dir=args.raw_data_dir,
            export_dir=args.export_dir,
            target_frequency=args.target_frequency,
            original_frequency=args.original_frequency,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            filter_last_lap=args.filter_last_lap,
            shuffle=args.shuffle,
            random_seed=args.random_seed,
            intermediate_cleanup=args.cleanup,
        )

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
