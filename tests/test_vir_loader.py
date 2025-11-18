"""Tests for VIR telemetry data loader."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apexai.data_generation.vir_loader import (
    VIRTelemetryLoader,
    create_vehicle_label_mapping,
    load_vir_telemetry,
)


class TestVehicleLabelMapping:
    """Tests for vehicle ID to label mapping."""

    def test_create_vehicle_label_mapping_basic(self) -> None:
        """Test basic vehicle label mapping creation."""
        vehicle_ids = ["GR86-002-2", "GR86-006-7", "GR86-013-80"]
        mapping = create_vehicle_label_mapping(vehicle_ids)

        assert len(mapping) == 3
        assert all(isinstance(v, int) for v in mapping.values())
        assert set(mapping.values()) == {0, 1, 2}

    def test_create_vehicle_label_mapping_sorted(self) -> None:
        """Test that mapping is sorted alphabetically."""
        vehicle_ids = ["GR86-013-80", "GR86-002-2", "GR86-006-7"]
        mapping = create_vehicle_label_mapping(vehicle_ids)

        # Should be sorted: GR86-002-2 (0), GR86-006-7 (1), GR86-013-80 (2)
        assert mapping["GR86-002-2"] == 0
        assert mapping["GR86-006-7"] == 1
        assert mapping["GR86-013-80"] == 2

    def test_create_vehicle_label_mapping_empty(self) -> None:
        """Test mapping with empty list."""
        mapping = create_vehicle_label_mapping([])
        assert mapping == {}


class TestLoadVIRTelemetry:
    """Tests for loading VIR telemetry data."""

    @pytest.fixture
    def sample_telemetry_csv(self) -> Path:
        """Create a sample telemetry CSV file."""
        csv_content = """expire_at,lap,meta_event,meta_session,meta_source,meta_time,original_vehicle_id,outing,telemetry_name,telemetry_value,timestamp,vehicle_id,vehicle_number
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,accx_can,0.217,2025-07-17T19:16:54.077Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,accy_can,-0.19,2025-07-17T19:16:54.077Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,ath,100.02,2025-07-17T19:16:54.077Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,speed,55.25,2025-07-17T19:16:54.077Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-006-7,0,accx_can,0.3,2025-07-17T19:16:54.100Z,GR86-006-7,7
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-006-7,0,speed,60.0,2025-07-17T19:16:54.100Z,GR86-006-7,7
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
        ) as f:
            f.write(csv_content)
            return Path(f.name)

    def test_load_vir_telemetry_pandas(self, sample_telemetry_csv: Path) -> None:
        """Test loading telemetry data with Pandas."""
        df = load_vir_telemetry(sample_telemetry_csv, use_polars=False)

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "vehicle_id" in df.columns
        assert "telemetry_name" in df.columns
        assert "telemetry_value" in df.columns
        assert len(df) == 6

    def test_load_vir_telemetry_polars(self, sample_telemetry_csv: Path) -> None:
        """Test loading telemetry data with Polars."""
        try:
            import polars as pl

            df = load_vir_telemetry(sample_telemetry_csv, use_polars=True)
            assert isinstance(df, pl.DataFrame)
            assert "timestamp" in df.columns
            assert "vehicle_id" in df.columns
        except ImportError:
            pytest.skip("Polars not installed")

    def test_load_vir_telemetry_nrows(self, sample_telemetry_csv: Path) -> None:
        """Test loading limited number of rows."""
        df = load_vir_telemetry(sample_telemetry_csv, use_polars=False, nrows=3)
        assert len(df) == 3

    def test_load_vir_telemetry_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Telemetry file not found"):
            load_vir_telemetry("nonexistent_file.csv")


class TestVIRTelemetryLoader:
    """Tests for VIRTelemetryLoader class."""

    @pytest.fixture
    def sample_telemetry_csv(self) -> Path:
        """Create a sample telemetry CSV file."""
        csv_content = """expire_at,lap,meta_event,meta_session,meta_source,meta_time,original_vehicle_id,outing,telemetry_name,telemetry_value,timestamp,vehicle_id,vehicle_number
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,accx_can,0.217,2025-07-17T19:16:54.077Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,accy_can,-0.19,2025-07-17T19:16:54.077Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,ath,100.02,2025-07-17T19:16:54.077Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-002-2,0,speed,55.25,2025-07-17T19:16:54.077Z,GR86-002-2,2
,2,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.180Z,GR86-002-2,0,accx_can,0.3,2025-07-17T19:16:54.100Z,GR86-002-2,2
,2,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.180Z,GR86-002-2,0,speed,60.0,2025-07-17T19:16:54.100Z,GR86-002-2,2
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-006-7,0,accx_can,0.4,2025-07-17T19:16:54.077Z,GR86-006-7,7
,1,I_R04_2025-07-20,R1,kafka:gr-raw,2025-07-19T18:06:40.175Z,GR86-006-7,0,speed,65.0,2025-07-17T19:16:54.077Z,GR86-006-7,7
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
        ) as f:
            f.write(csv_content)
            return Path(f.name)

    def test_loader_initialization(self, sample_telemetry_csv: Path) -> None:
        """Test loader initialization and vehicle mapping."""
        loader = VIRTelemetryLoader(sample_telemetry_csv, use_polars=False)

        assert loader.num_classes == 2
        assert "GR86-002-2" in loader.vehicle_to_label
        assert "GR86-006-7" in loader.vehicle_to_label
        assert len(loader.vehicle_to_label) == 2

    def test_convert_to_wide_format(self, sample_telemetry_csv: Path) -> None:
        """Test converting long format to wide format."""
        loader = VIRTelemetryLoader(sample_telemetry_csv, use_polars=False)
        long_data = loader.load_telemetry()
        wide_data = loader.convert_to_wide_format(long_data)

        assert isinstance(wide_data, pd.DataFrame)
        assert "timestamp" in wide_data.columns
        assert "vehicle_id" in wide_data.columns
        assert "accx_can" in wide_data.columns
        assert "speed" in wide_data.columns

        # Should have 3 unique timestamps
        assert len(wide_data) == 3

    def test_get_vehicle_data(self, sample_telemetry_csv: Path) -> None:
        """Test extracting data for a specific vehicle."""
        loader = VIRTelemetryLoader(sample_telemetry_csv, use_polars=False)
        long_data = loader.load_telemetry()
        wide_data = loader.convert_to_wide_format(long_data)

        vehicle_data = loader.get_vehicle_data(wide_data, "GR86-002-2")

        assert len(vehicle_data) == 2  # 2 timestamps for this vehicle
        assert (vehicle_data["vehicle_id"] == "GR86-002-2").all()

    def test_get_vehicle_data_invalid(self, sample_telemetry_csv: Path) -> None:
        """Test error when requesting invalid vehicle."""
        loader = VIRTelemetryLoader(sample_telemetry_csv, use_polars=False)
        long_data = loader.load_telemetry()
        wide_data = loader.convert_to_wide_format(long_data)

        with pytest.raises(ValueError, match="Unknown vehicle_id"):
            loader.get_vehicle_data(wide_data, "INVALID-ID")

    def test_create_labels_for_data(self, sample_telemetry_csv: Path) -> None:
        """Test creating label array from vehicle IDs."""
        loader = VIRTelemetryLoader(sample_telemetry_csv, use_polars=False)
        long_data = loader.load_telemetry()
        wide_data = loader.convert_to_wide_format(long_data)

        labels = loader.create_labels_for_data(wide_data)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(wide_data)
        assert labels.dtype in (np.int64, np.int32)
        assert set(labels) == {0, 1}  # Two vehicles -> labels 0 and 1
