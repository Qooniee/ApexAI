"""Preprocess raw telemetry data into lap-based CSV files.

This script converts raw telemetry data (long format) into processed
lap-based files (wide format) for efficient training data loading.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from apexai.signal_processing.resample import resample

from .vir_loader import VIRTelemetryLoader


def preprocess_telemetry_to_laps(
    telemetry_path: Path | str,
    output_dir: Path | str,
    *,
    use_polars: bool = True,
    min_lap_length: int = 100,
    interpolate_missing: bool = True,
    interpolation_method: str = "linear",
    resample_frequency: float | None = None,
    original_frequency: float | None = None,
) -> dict[str, Any]:
    """Preprocess raw telemetry data into lap-based CSV files.

    Args:
        telemetry_path: Path to raw telemetry CSV file.
        output_dir: Directory to save processed lap files.
        use_polars: Use Polars for faster processing.
        min_lap_length: Minimum number of samples per lap (filter short laps).
        interpolate_missing: If True, interpolate missing sensor values.
        interpolation_method: Interpolation method ('linear', 'time', 'polynomial').
        resample_frequency: Target sampling frequency [Hz] (e.g., 10.0 for 10Hz).
                           If None, no resampling is performed.
        original_frequency: Original sampling frequency [Hz] (e.g., 23.0 for ~23Hz).
                          Required if resample_frequency is specified.
                          If None, auto-detected from elapsed_time.

    Returns:
        Dictionary with preprocessing statistics:
            - num_vehicles: Number of unique vehicles
            - num_laps: Total number of laps processed
            - num_samples: Total number of samples
            - output_files: List of generated file paths
            - resampled: Whether resampling was applied

    Examples:
        >>> # Without resampling
        >>> stats = preprocess_telemetry_to_laps(
        ...     'dataset/VIR/Race 1/R1_vir_telemetry_data.csv',
        ...     'dataset/processed/VIR_R1'
        ... )
        >>>
        >>> # With resampling to 10Hz
        >>> stats = preprocess_telemetry_to_laps(
        ...     'dataset/VIR/Race 1/R1_vir_telemetry_data.csv',
        ...     'dataset/processed/VIR_R1_10Hz',
        ...     resample_frequency=10.0,
        ...     original_frequency=23.0
        ... )

    """
    telemetry_path = Path(telemetry_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    loader = VIRTelemetryLoader(telemetry_path, use_polars=use_polars)

    # Load full telemetry data
    df = loader.load_telemetry()

    # Convert to wide format
    wide_df = loader.convert_to_wide_format(df)

    # Add vehicle labels (numeric mapping of vehicle_id)
    wide_df["vehicle_label"] = wide_df["vehicle_id"].map(loader.vehicle_to_label)

    # Sort by vehicle, lap, and timestamp
    wide_df = wide_df.sort_values(["vehicle_id", "lap", "timestamp"])

    # Process each lap
    output_files = []
    num_laps = 0
    num_samples = 0
    num_skipped = 0

    grouped = wide_df.groupby(["vehicle_id", "lap"])

    for (vehicle_id, lap_id), lap_df in tqdm(
        grouped,
        desc="Processing laps",
        total=len(grouped),
    ):
        # Skip short laps
        if len(lap_df) < min_lap_length:
            num_skipped += 1
            continue

        # Reset index and add elapsed time
        lap_df = lap_df.reset_index(drop=True).copy()
        lap_df["timestamp_dt"] = pd.to_datetime(lap_df["timestamp"])
        start_time = lap_df["timestamp_dt"].iloc[0]
        lap_df["elapsed_time"] = (lap_df["timestamp_dt"] - start_time).dt.total_seconds()

        # Select relevant columns
        feature_columns = [
            "elapsed_time",
            "speed",
            "ath",  # throttle
            "pbrake_f",
            "pbrake_r",
            "Steering_Angle",
            "accx_can",
            "accy_can",
            "gear",
            "nmot",  # RPM
        ]

        # Keep only available columns
        available_features = [col for col in feature_columns if col in lap_df.columns]

        output_columns = [
            "timestamp",
            *available_features,
            "vehicle_id",
            "vehicle_label",
            "lap",
        ]

        lap_output = lap_df[output_columns].copy()

        # Interpolate missing values if requested
        if interpolate_missing:
            # Get numeric feature columns (exclude metadata)
            numeric_features = [
                col
                for col in available_features
                if col not in ["timestamp", "vehicle_id", "vehicle_label", "lap"]
            ]

            # Count missing values before interpolation
            missing_before = lap_output[numeric_features].isnull().sum().sum()

            if missing_before > 0:
                # Interpolate using specified method
                lap_output[numeric_features] = lap_output[numeric_features].interpolate(
                    method=interpolation_method,
                    limit_direction="both",  # Interpolate in both directions
                    axis=0,  # Along rows (time axis)
                )

                # For remaining NaN (e.g., at boundaries), use forward/backward fill
                lap_output[numeric_features] = lap_output[numeric_features].ffill().bfill()

                missing_after = lap_output[numeric_features].isnull().sum().sum()

                # Log if there are still missing values after interpolation
                if missing_after > 0:
                    pass

        # Apply resampling if requested
        if resample_frequency is not None:
            # Auto-detect original frequency if not provided
            if original_frequency is None:
                # Calculate average sampling rate from elapsed_time
                if "elapsed_time" in lap_output.columns and len(lap_output) > 1:
                    time_diff = lap_output["elapsed_time"].diff().dropna()
                    avg_dt = time_diff.mean()
                    original_frequency = 1.0 / avg_dt if avg_dt > 0 else 23.0
                else:
                    # Default to ~23Hz for VIR data
                    original_frequency = 23.0

            # Get numeric feature columns for resampling
            numeric_features = [
                col
                for col in available_features
                if col not in ["timestamp", "vehicle_id", "vehicle_label", "lap", "elapsed_time"]
            ]

            # Apply resampling with anti-aliasing
            try:
                lap_output = resample(
                    df=lap_output,
                    time_label="elapsed_time",
                    signal_labels=numeric_features,
                    original_fs=original_frequency,
                    target_fs=resample_frequency,
                    anti_aliasing=True,
                    interpolation_kind="linear",
                )

                # Update metadata columns (use nearest neighbor for discrete values)
                # These will be constant within a lap anyway
                lap_output["vehicle_id"] = vehicle_id
                lap_output["vehicle_label"] = lap_df["vehicle_label"].iloc[0]
                lap_output["lap"] = lap_id

                # Update timestamp to match new elapsed_time
                if "timestamp" in output_columns:
                    base_timestamp = pd.to_datetime(lap_df["timestamp"].iloc[0])
                    lap_output["timestamp"] = base_timestamp + pd.to_timedelta(
                        lap_output["elapsed_time"], unit="s"
                    )

            except Exception as e:
                # Log error but continue with original data
                print(f"Warning: Resampling failed for {vehicle_id} lap {lap_id}: {e}")
                print("Continuing with original sampling rate...")

        # Generate filename
        # Format: VIR_R1_vehicle_GR86-002-2_lap_001.csv
        track_name = telemetry_path.parent.parent.name  # VIR
        race_num = telemetry_path.stem.split("_")[0]  # R1
        vehicle_safe = vehicle_id.replace("-", "_")
        lap_safe = f"{int(lap_id):03d}"

        filename = f"{track_name}_{race_num}_vehicle_{vehicle_safe}_lap_{lap_safe}.csv"
        output_path = output_dir / filename

        # Save to CSV
        lap_output.to_csv(output_path, index=False)

        output_files.append(output_path)
        num_laps += 1
        num_samples += len(lap_df)

    # Save metadata
    import json

    metadata = {
        "num_vehicles": loader.num_classes,
        "num_laps": num_laps,
        "num_samples": num_samples,
        "num_skipped": num_skipped,
        "vehicle_to_label": loader.vehicle_to_label,
        "label_to_vehicle": loader.label_to_vehicle,
        "resampled": resample_frequency is not None,
        "resample_frequency": resample_frequency,
        "original_frequency": original_frequency if resample_frequency is not None else None,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return {
        **metadata,
        "output_files": output_files,
    }


def load_preprocessed_lap(lap_csv_path: Path | str) -> pd.DataFrame:
    """Load a single preprocessed lap CSV.

    Args:
        lap_csv_path: Path to lap CSV file.

    Returns:
        DataFrame with lap data.

    Examples:
        >>> lap_df = load_preprocessed_lap(
        ...     'processed/VIR_R1_vehicle_GR86_002_2_lap_001.csv'
        ... )
        >>> lap_df.shape
        (1500, 14)

    """
    return pd.read_csv(lap_csv_path)


def get_all_lap_files(processed_dir: Path | str, pattern: str = "*.csv") -> list[Path]:
    """Get all preprocessed lap CSV files from directory.

    Args:
        processed_dir: Directory containing preprocessed lap files.
        pattern: Glob pattern for matching files.

    Returns:
        List of lap file paths, sorted by filename.

    Examples:
        >>> files = get_all_lap_files('dataset/processed/VIR_R1')
        >>> len(files)
        420  # 21 vehicles × ~20 laps

    """
    processed_dir = Path(processed_dir)
    return sorted(processed_dir.glob(pattern))


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    stats = preprocess_telemetry_to_laps(input_path, output_path)

    # Save metadata
    import json

    metadata_path = Path(output_path) / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(
            {
                "num_vehicles": stats["num_vehicles"],
                "num_laps": stats["num_laps"],
                "num_samples": stats["num_samples"],
                "num_skipped": stats["num_skipped"],
                "vehicle_to_label": stats["vehicle_to_label"],
                "label_to_vehicle": stats["label_to_vehicle"],
            },
            f,
            indent=2,
        )
