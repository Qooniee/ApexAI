"""VIR telemetry data loader for driver identification.

This module provides functionality to load and preprocess telemetry data
from Virginia International Raceway (VIR) for driver identification tasks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def create_vehicle_label_mapping(vehicle_ids: list[str]) -> dict[str, int]:
    """Create mapping from vehicle IDs to integer labels.

    Args:
        vehicle_ids: List of unique vehicle IDs (e.g., ['GR86-002-2', 'GR86-006-7']).

    Returns:
        Dictionary mapping vehicle IDs to integer labels (0-indexed).

    Examples:
        >>> ids = ['GR86-002-2', 'GR86-006-7', 'GR86-013-80']
        >>> mapping = create_vehicle_label_mapping(ids)
        >>> mapping['GR86-002-2']
        0
        >>> mapping['GR86-006-7']
        1

    """
    sorted_ids = sorted(vehicle_ids)
    return {vehicle_id: idx for idx, vehicle_id in enumerate(sorted_ids)}


def load_vir_telemetry(
    telemetry_path: Path | str,
    *,
    use_polars: bool = True,
    nrows: int | None = None,
) -> pd.DataFrame | pl.DataFrame:
    """Load VIR telemetry data from CSV file.

    Loads telemetry data in long format where each row represents a single
    sensor reading at a specific timestamp.

    Args:
        telemetry_path: Path to the telemetry CSV file.
        use_polars: If True, use Polars for faster loading. If False, use Pandas.
        nrows: Number of rows to read. If None, read all rows.

    Returns:
        DataFrame with columns:
            - timestamp: Recording timestamp
            - vehicle_id: Vehicle identifier
            - lap: Lap number
            - telemetry_name: Sensor name (e.g., 'speed', 'ath', 'pbrake_f')
            - telemetry_value: Sensor reading value

    Raises:
        FileNotFoundError: If telemetry file does not exist.
        ValueError: If required columns are missing.

    Examples:
        >>> df = load_vir_telemetry('dataset/VIR/Race 1/R1_vir_telemetry_data.csv')
        >>> df['telemetry_name'].unique()
        ['accx_can', 'accy_can', 'ath', 'pbrake_r', ...]

    """
    telemetry_path = Path(telemetry_path)

    if not telemetry_path.exists():
        msg = f"Telemetry file not found: {telemetry_path}"
        raise FileNotFoundError(msg)

    required_columns = {
        "timestamp",
        "vehicle_id",
        "lap",
        "telemetry_name",
        "telemetry_value",
    }

    if use_polars:
        # Polars is faster for large CSV files
        df = pl.read_csv(telemetry_path, n_rows=nrows)
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)
        return df
    # Pandas fallback
    df = pd.read_csv(telemetry_path, nrows=nrows)
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)
    return df


class VIRTelemetryLoader:
    """Load and preprocess VIR telemetry data for driver identification.

    This class handles:
        1. Loading telemetry data (long format)
        2. Converting to wide format (timestamp × features)
        3. Splitting data by vehicle/driver
        4. Creating label mappings

    Attributes:
        telemetry_path: Path to telemetry CSV file.
        vehicle_to_label: Mapping from vehicle IDs to integer labels.
        num_classes: Number of unique drivers/vehicles.

    Examples:
        >>> loader = VIRTelemetryLoader('dataset/VIR/Race 1/R1_vir_telemetry_data.csv')
        >>> loader.num_classes
        21
        >>> data = loader.load_telemetry(nrows=10000)
        >>> wide_data = loader.convert_to_wide_format(data)

    """

    def __init__(
        self,
        telemetry_path: Path | str,
        *,
        use_polars: bool = True,
    ) -> None:
        """Initialize VIR telemetry loader.

        Args:
            telemetry_path: Path to the telemetry CSV file.
            use_polars: If True, use Polars for data processing (faster).

        """
        self.telemetry_path = Path(telemetry_path)
        self.use_polars = use_polars
        self.vehicle_to_label: dict[str, int] = {}
        self.label_to_vehicle: dict[int, str] = {}
        self.num_classes: int = 0

        # Initialize mappings
        self._initialize_vehicle_mappings()

    def _initialize_vehicle_mappings(self) -> None:
        """Initialize vehicle ID to label mappings by scanning unique vehicles."""
        # Load ALL data to get unique vehicle IDs (not just first 100k rows)
        df = load_vir_telemetry(
            self.telemetry_path,
            use_polars=self.use_polars,
            nrows=None,  # Read all rows to ensure we get all vehicles
        )

        if self.use_polars:
            unique_vehicles = df.select("vehicle_id").unique().to_series().to_list()
        else:
            unique_vehicles = df["vehicle_id"].unique().tolist()

        self.vehicle_to_label = create_vehicle_label_mapping(unique_vehicles)
        self.label_to_vehicle = {v: k for k, v in self.vehicle_to_label.items()}
        self.num_classes = len(unique_vehicles)

    def load_telemetry(
        self,
        *,
        nrows: int | None = None,
    ) -> pd.DataFrame | pl.DataFrame:
        """Load telemetry data from CSV.

        Args:
            nrows: Number of rows to read. If None, read all rows.

        Returns:
            Telemetry data in long format.

        """
        return load_vir_telemetry(
            self.telemetry_path,
            use_polars=self.use_polars,
            nrows=nrows,
        )

    def convert_to_wide_format(
        self,
        data: pd.DataFrame | pl.DataFrame,
    ) -> pd.DataFrame:
        """Convert telemetry data from long format to wide format.

        Transforms data where each sensor reading is a separate row into
        a format where each timestamp has all sensor readings in columns.

        Args:
            data: Telemetry data in long format.

        Returns:
            Wide format DataFrame with columns:
                - timestamp: Recording timestamp
                - vehicle_id: Vehicle identifier
                - lap: Lap number
                - speed: Speed sensor
                - ath: Throttle position
                - pbrake_f: Front brake pressure
                - ... (other sensors)

        Examples:
            >>> loader = VIRTelemetryLoader('path/to/telemetry.csv')
            >>> long_data = loader.load_telemetry(nrows=10000)
            >>> wide_data = loader.convert_to_wide_format(long_data)
            >>> wide_data.shape
            (1000, 12)  # timestamp, vehicle_id, lap + 9 sensors

        """
        if self.use_polars and isinstance(data, pl.DataFrame):
            # Polars pivot
            wide = data.pivot(
                index=["timestamp", "vehicle_id", "lap"],
                columns="telemetry_name",
                values="telemetry_value",
                aggregate_function="first",
            )
            return wide.to_pandas()

        # Pandas pivot
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()

        wide = data.pivot_table(
            index=["timestamp", "vehicle_id", "lap"],
            columns="telemetry_name",
            values="telemetry_value",
            aggfunc="first",
        ).reset_index()

        # Flatten column names
        wide.columns.name = None

        return wide

    def get_vehicle_data(
        self,
        data: pd.DataFrame,
        vehicle_id: str,
    ) -> pd.DataFrame:
        """Extract data for a specific vehicle.

        Args:
            data: Wide format telemetry data.
            vehicle_id: Vehicle ID to extract.

        Returns:
            Telemetry data for the specified vehicle only.

        Raises:
            ValueError: If vehicle_id is not found in the data.

        """
        if vehicle_id not in self.vehicle_to_label:
            msg = f"Unknown vehicle_id: {vehicle_id}"
            raise ValueError(msg)

        vehicle_data = data[data["vehicle_id"] == vehicle_id].copy()

        if vehicle_data.empty:
            msg = f"No data found for vehicle: {vehicle_id}"
            raise ValueError(msg)

        return vehicle_data

    def create_labels_for_data(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """Create label array from vehicle IDs in the data.

        Args:
            data: DataFrame with 'vehicle_id' column.

        Returns:
            Array of integer labels corresponding to each row.

        Examples:
            >>> loader = VIRTelemetryLoader('path/to/telemetry.csv')
            >>> wide_data = loader.convert_to_wide_format(loader.load_telemetry())
            >>> labels = loader.create_labels_for_data(wide_data)
            >>> labels.shape
            (1000,)

        """
        return data["vehicle_id"].map(self.vehicle_to_label).to_numpy()
