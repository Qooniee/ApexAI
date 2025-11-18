"""Unit tests for signal resampling module."""

import numpy as np
import pandas as pd
import pytest

from apexai.signal_processing.resample import (
    resample,
    resample_dataframe,
    validate_resampling_params,
)


class TestResampleDownsampling:
    """Test cases for downsampling with anti-aliasing."""

    @pytest.fixture
    def high_freq_signal(self) -> pd.DataFrame:
        """Create test signal with high and low frequency components."""
        fs = 100.0  # 100Hz sampling
        duration = 2.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        # Signal: low freq (2Hz) + high freq noise (40Hz)
        signal_low = np.sin(2 * np.pi * 2 * t)
        signal_high = 0.5 * np.sin(2 * np.pi * 40 * t)

        return pd.DataFrame({"time": t, "signal": signal_low + signal_high, "clean": signal_low})

    def test_basic_downsampling(self, high_freq_signal):
        """Test basic downsampling with anti-aliasing."""
        result = resample(
            df=high_freq_signal,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=10.0,
        )

        # Check output shape
        expected_samples = int(np.round(2.0 * 10.0))  # ~20 samples
        assert len(result) == expected_samples

        # Check time column
        assert "time" in result.columns
        assert result["time"].min() == pytest.approx(0.0)
        assert result["time"].max() == pytest.approx(2.0, rel=0.05)

        # Check signal column exists
        assert "signal" in result.columns

    def test_anti_aliasing_effect(self, high_freq_signal):
        """Test that anti-aliasing filter removes high frequencies."""
        # Downsample with anti-aliasing
        with_aa = resample(
            df=high_freq_signal,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=10.0,
            anti_aliasing=True,
        )

        # Downsample without anti-aliasing
        without_aa = resample(
            df=high_freq_signal,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=10.0,
            anti_aliasing=False,
        )

        # With anti-aliasing should have lower variance (high freq removed)
        assert with_aa["signal"].std() < without_aa["signal"].std()

    def test_custom_filter_params(self, high_freq_signal):
        """Test custom anti-aliasing filter parameters."""
        custom_params = {"fpass": 3.0, "fstop": 4.0, "gpass": 1.0, "gstop": 40.0}

        result = resample(
            df=high_freq_signal,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=10.0,
            filter_params=custom_params,
        )

        # Should complete without error
        assert len(result) > 0

    def test_multiple_signals(self, high_freq_signal):
        """Test downsampling multiple signals simultaneously."""
        result = resample(
            df=high_freq_signal,
            time_label="time",
            signal_labels=["signal", "clean"],
            original_fs=100.0,
            target_fs=10.0,
        )

        # Both signals should be present
        assert "signal" in result.columns
        assert "clean" in result.columns

        # Same length
        assert len(result["signal"]) == len(result["clean"])


class TestResampleUpsampling:
    """Test cases for upsampling."""

    @pytest.fixture
    def low_freq_signal(self) -> pd.DataFrame:
        """Create low frequency test signal."""
        fs = 10.0  # 10Hz sampling
        duration = 2.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        signal = np.sin(2 * np.pi * 1 * t)  # 1Hz sine wave

        return pd.DataFrame({"time": t, "signal": signal})

    def test_basic_upsampling(self, low_freq_signal):
        """Test basic upsampling."""
        result = resample(
            df=low_freq_signal,
            time_label="time",
            signal_labels=["signal"],
            original_fs=10.0,
            target_fs=100.0,
        )

        # Check output shape
        expected_samples = int(np.round(2.0 * 100.0))  # ~200 samples
        assert len(result) <= expected_samples + 1  # Allow small rounding differences

        # Check interpolation preserves signal characteristics
        # Mean should be approximately preserved
        assert result["signal"].mean() == pytest.approx(low_freq_signal["signal"].mean(), abs=0.1)

    def test_interpolation_methods(self, low_freq_signal):
        """Test different interpolation methods."""
        methods = ["linear", "cubic", "nearest"]

        for method in methods:
            result = resample(
                df=low_freq_signal,
                time_label="time",
                signal_labels=["signal"],
                original_fs=10.0,
                target_fs=50.0,
                interpolation_kind=method,
            )

            assert len(result) > len(low_freq_signal)
            assert "signal" in result.columns

    def test_no_anti_aliasing_for_upsampling(self, low_freq_signal):
        """Test that anti-aliasing is not applied for upsampling."""
        # Both should give same result for upsampling
        with_aa = resample(
            df=low_freq_signal,
            time_label="time",
            signal_labels=["signal"],
            original_fs=10.0,
            target_fs=50.0,
            anti_aliasing=True,
        )

        without_aa = resample(
            df=low_freq_signal,
            time_label="time",
            signal_labels=["signal"],
            original_fs=10.0,
            target_fs=50.0,
            anti_aliasing=False,
        )

        # Should be identical (no filtering for upsampling)
        np.testing.assert_array_almost_equal(with_aa["signal"].values, without_aa["signal"].values)


class TestResampleValidation:
    """Test cases for input validation."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create simple test data."""
        t = np.linspace(0, 1, 100)
        return pd.DataFrame({"time": t, "signal": np.sin(2 * np.pi * t)})

    def test_missing_time_column(self, sample_data):
        """Test error when time column doesn't exist."""
        with pytest.raises(ValueError, match="Time column.*not found"):
            resample(
                df=sample_data,
                time_label="nonexistent",
                signal_labels=["signal"],
                original_fs=100.0,
                target_fs=10.0,
            )

    def test_missing_signal_column(self, sample_data):
        """Test error when signal column doesn't exist."""
        with pytest.raises(ValueError, match="Signal column.*not found"):
            resample(
                df=sample_data,
                time_label="time",
                signal_labels=["nonexistent"],
                original_fs=100.0,
                target_fs=10.0,
            )

    def test_invalid_frequencies(self, sample_data):
        """Test error with invalid frequencies."""
        with pytest.raises(ValueError, match="must be positive"):
            resample(
                df=sample_data,
                time_label="time",
                signal_labels=["signal"],
                original_fs=-10.0,
                target_fs=10.0,
            )

        with pytest.raises(ValueError, match="must be positive"):
            resample(
                df=sample_data,
                time_label="time",
                signal_labels=["signal"],
                original_fs=100.0,
                target_fs=0.0,
            )

    def test_same_frequency_no_resampling(self, sample_data):
        """Test that no resampling occurs when frequencies are equal."""
        result = resample(
            df=sample_data,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=100.0,
        )

        # Should return copy of original
        pd.testing.assert_frame_equal(result, sample_data)


class TestResampleEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_short_signal(self):
        """Test with very short signal."""
        df = pd.DataFrame({"time": [0, 0.1, 0.2], "signal": [0, 1, 0]})

        result = resample(
            df=df,
            time_label="time",
            signal_labels=["signal"],
            original_fs=10.0,
            target_fs=5.0,
        )

        # Should handle short signals
        assert len(result) > 0

    def test_constant_signal(self):
        """Test with constant signal."""
        t = np.linspace(0, 1, 100)
        df = pd.DataFrame({"time": t, "signal": np.ones(100) * 5.0})

        result = resample(
            df=df,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=10.0,
        )

        # Constant signal should remain constant
        assert np.allclose(result["signal"].values, 5.0, rtol=1e-5)

    def test_large_downsampling_ratio(self):
        """Test with large downsampling ratio."""
        fs = 1000.0
        t = np.linspace(0, 1, int(fs), endpoint=False)
        df = pd.DataFrame({"time": t, "signal": np.sin(2 * np.pi * 5 * t)})

        # Downsample from 1000Hz to 50Hz (20x)
        result = resample(
            df=df,
            time_label="time",
            signal_labels=["signal"],
            original_fs=1000.0,
            target_fs=50.0,
        )

        # Should complete successfully
        assert len(result) == pytest.approx(50, rel=0.1)


class TestResampleIntegration:
    """Integration tests for realistic scenarios."""

    def test_telemetry_downsampling_23hz_to_10hz(self):
        """Test realistic telemetry downsampling (23Hz -> 10Hz)."""
        # Simulate telemetry at 23Hz for 10 seconds
        fs_orig = 23.0
        fs_target = 10.0
        duration = 10.0

        t = np.linspace(0, duration, int(fs_orig * duration), endpoint=False)

        # Realistic telemetry signals
        speed = 50 + 10 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(len(t)) * 2
        throttle = 30 + 20 * np.sin(2 * np.pi * 0.3 * t)

        df = pd.DataFrame({"time": t, "speed": speed, "throttle": throttle})

        result = resample(
            df=df,
            time_label="time",
            signal_labels=["speed", "throttle"],
            original_fs=fs_orig,
            target_fs=fs_target,
        )

        # Check output
        expected_samples = int(np.round(duration * fs_target))
        assert len(result) == expected_samples

        # Signals should preserve mean
        assert result["speed"].mean() == pytest.approx(df["speed"].mean(), abs=2.0)
        assert result["throttle"].mean() == pytest.approx(df["throttle"].mean(), abs=2.0)


class TestValidateResamplingParams:
    """Test parameter validation function."""

    def test_valid_downsampling(self):
        """Test validation for valid downsampling."""
        result = validate_resampling_params(original_fs=100.0, target_fs=10.0, signal_length=1000)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert "recommended_filter" in result
        # Filter params are now more conservative
        assert result["recommended_filter"]["fpass"] < 5.0
        assert result["recommended_filter"]["fpass"] > 0.0

    def test_large_downsampling_warning(self):
        """Test warning for large downsampling ratio."""
        result = validate_resampling_params(original_fs=1000.0, target_fs=10.0, signal_length=1000)

        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "Large downsampling ratio" in result["warnings"][0]

    def test_short_signal_error(self):
        """Test error for too short signal."""
        result = validate_resampling_params(original_fs=100.0, target_fs=10.0, signal_length=5)

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "too short" in result["errors"][0]


class TestResampleDataframeWrapper:
    """Test convenience wrapper function."""

    def test_wrapper_function(self):
        """Test that wrapper function works correctly."""
        t = np.linspace(0, 1, 100)
        df = pd.DataFrame({"time": t, "signal": np.sin(2 * np.pi * t)})

        result = resample_dataframe(
            df=df,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=10.0,
        )

        # Should produce same result as resample()
        result2 = resample(
            df=df,
            time_label="time",
            signal_labels=["signal"],
            original_fs=100.0,
            target_fs=10.0,
        )

        pd.testing.assert_frame_equal(result, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
