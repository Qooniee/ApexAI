"""Generate X_train.csv and y_train.csv from preprocessed lap files.

This module creates training data files for each lap without overlapping or splitting.
Data splitting and overlapping will be done later during training.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .preprocess_telemetry import get_all_lap_files, load_preprocessed_lap


def generate_training_files(
    processed_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    target_label: str = "vehicle_label",
    feature_columns: list[str] | None = None,
    output_suffix: str = "_train",
    separate_xy_dirs: bool = True,
) -> dict[str, Any]:
    """Generate X_train.csv and y_train.csv for each lap file.

    Creates individual training files for each lap without overlapping.
    Files are named: <original_name>_X_train.csv and <original_name>_y_train.csv

    Args:
        processed_dir: Directory containing preprocessed lap CSV files.
        output_dir: Directory to save training files. If None, save to processed_dir.
        target_label: Column name for prediction target (e.g., 'vehicle_label').
        feature_columns: List of feature columns to include in X. If None, use all.
        output_suffix: Suffix to add to output files (default: '_train').
        separate_xy_dirs: If True, save X files to x_train/ and y files to y_train/ subdirs.

    Returns:
        Dictionary with generation statistics:
            - num_laps: Number of lap files processed
            - total_samples: Total number of timesteps across all laps
            - feature_columns: List of feature columns used
            - target_label: Target label column name
            - output_files: List of generated file paths

    Examples:
        >>> stats = generate_training_files(
        ...     'dataset/processed/VIR_R1',
        ...     output_dir='dataset/VIR_training_dataset',
        ...     target_label='vehicle_label'
        ... )
        >>> print(f"Generated {stats['num_laps']} training file pairs")

    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir) if output_dir else processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create separate subdirectories for X and y files if requested
    if separate_xy_dirs:
        x_output_dir = output_dir / "x_train"
        y_output_dir = output_dir / "y_train"
        x_output_dir.mkdir(parents=True, exist_ok=True)
        y_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        x_output_dir = output_dir
        y_output_dir = output_dir

    # Get all lap files (excluding already generated training files)
    lap_files = [
        f for f in get_all_lap_files(processed_dir) if not f.stem.endswith(("_X_train", "_y_train"))
    ]

    if len(lap_files) == 0:
        msg = f"No lap files found in {processed_dir}"
        raise ValueError(msg)

    # Load first file to determine available features
    sample_lap = load_preprocessed_lap(lap_files[0])

    # Default feature columns (all columns except metadata and target)
    if feature_columns is None:
        exclude_cols = {"vehicle_id", "vehicle_label", "lap"}
        feature_columns = [col for col in sample_lap.columns if col not in exclude_cols]

    # Verify target label exists
    if target_label not in sample_lap.columns:
        msg = f"Target label '{target_label}' not found in lap files"
        raise ValueError(msg)

    # Generate X_train.csv and y_train.csv for each lap
    output_files = []
    total_samples = 0
    driver_lap_counts: dict[str, int] = {}

    for lap_file in tqdm(lap_files, desc="Processing laps"):
        lap_df = load_preprocessed_lap(lap_file)

        # Extract features (X)
        features = lap_df[feature_columns]

        # Extract target (y)
        y = lap_df[[target_label]]

        # Generate output filenames
        stem = lap_file.stem  # e.g., 'VIR_R1_vehicle_GR86_002_2_lap_001'
        features_filename = f"{stem}_X{output_suffix}.csv"
        y_filename = f"{stem}_y{output_suffix}.csv"

        features_path = x_output_dir / features_filename
        y_path = y_output_dir / y_filename

        # Save to CSV
        features.to_csv(features_path, index=False)
        y.to_csv(y_path, index=False)

        output_files.append({"X": str(features_path), "y": str(y_path)})
        total_samples += len(lap_df)

        # Track label counts (works for both string and numeric labels)
        label_value = lap_df[target_label].iloc[0]

        # Skip if label is NaN
        if pd.isna(label_value):
            continue

        # Convert to string key for dictionary (handles both int and str labels)
        label_key = str(label_value)
        driver_lap_counts[label_key] = driver_lap_counts.get(label_key, 0) + 1

    # Save metadata
    metadata = {
        "num_laps": len(lap_files),
        "total_samples": total_samples,
        "feature_columns": feature_columns,
        "target_label": target_label,
        # Generic name for vehicle identification
        "label_counts": driver_lap_counts,
        "output_files": output_files,
    }

    metadata_path = output_dir / "training_files_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    # Display label distribution
    for label_id in sorted(driver_lap_counts.keys()):
        driver_lap_counts[label_id]

    return metadata


def get_training_file_pairs(processed_dir: Path | str) -> list[dict[str, Path]]:
    """Get all X_train/y_train file pairs from directory.

    Args:
        processed_dir: Directory containing training files.

    Returns:
        List of dicts with 'X' and 'y' paths:
            [
                {
                    'X': Path('...lap_001_X_train.csv'),
                    'y': Path('...lap_001_y_train.csv')
                },
                ...
            ]

    Examples:
        >>> file_pairs = get_training_file_pairs('dataset/processed/VIR_R1')
        >>> len(file_pairs)
        417

    """
    processed_dir = Path(processed_dir)

    # Find all X_train files
    features_files = sorted(processed_dir.glob("*_X_train.csv"))

    file_pairs = []
    for features_file in features_files:
        # Derive y_train filename from X_train filename
        y_file = features_file.parent / features_file.name.replace("_X_train.csv", "_y_train.csv")

        if y_file.exists():
            file_pairs.append({"X": features_file, "y": y_file})
        else:
            pass

    return file_pairs


def load_training_file_pair(
    features_path: Path | str, y_path: Path | str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a single X_train/y_train file pair.

    Args:
        features_path: Path to X_train.csv file.
        y_path: Path to y_train.csv file.

    Returns:
        Tuple of (features_df, y_df).

    """
    features = pd.read_csv(features_path)
    y = pd.read_csv(y_path)
    return features, y


def analyze_driver_distribution(processed_dir: Path | str) -> dict[int, dict]:
    """Analyze the distribution of laps per vehicle.

    Args:
        processed_dir: Directory containing training files.

    Returns:
        Dictionary mapping vehicle_id to statistics:
            {
                0: {'num_laps': 20, 'total_samples': 35000},
                1: {'num_laps': 18, 'total_samples': 30000},
                ...
            }

    Examples:
        >>> dist = analyze_driver_distribution('dataset/processed/VIR_R1')
        >>> dist[0]
        {'num_laps': 20, 'total_samples': 35000}

    """
    file_pairs = get_training_file_pairs(processed_dir)

    driver_stats = {}

    for pair in file_pairs:
        _, y = load_training_file_pair(pair["X"], pair["y"])

        driver_id = int(y.iloc[0, 0])  # Assuming first column is vehicle_label

        if driver_id not in driver_stats:
            driver_stats[driver_id] = {"num_laps": 0, "total_samples": 0}

        driver_stats[driver_id]["num_laps"] += 1
        driver_stats[driver_id]["total_samples"] += len(y)

    return driver_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate X_train and y_train files from lap data")
    parser.add_argument("input_dir", type=str, help="Input directory containing lap CSV files")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for X_train/y_train files (default: same as input)",
    )
    parser.add_argument(
        "--target-label", type=str, default="vehicle_label", help="Target label column name"
    )
    parser.add_argument(
        "--no-separate-dirs",
        action="store_true",
        help="Save X and y files in same directory (default: separate into x_train/y_train)",
    )

    args = parser.parse_args()

    stats = generate_training_files(
        args.input_dir,
        output_dir=args.output,
        target_label=args.target_label,
        separate_xy_dirs=not args.no_separate_dirs,
    )
