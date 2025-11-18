"""Unit tests for signal processing filter module."""

import numpy as np
import pandas as pd
import pytest

from apexai.signal_processing.filter import butterlowpass, filtering


class TestButterlowpass:
    """Test cases for Butterworth lowpass filter function."""

    def test_basic_filtering(self):
        """Test that filter removes high-frequency components."""
        # Create test signal: low freq (1Hz) + high freq (50Hz)
        fs = 200.0  # Sampling frequency
        t = np.linspace(0, 1, int(fs), endpoint=False)
        signal_clean = np.sin(2 * np.pi * 1 * t)  # 1Hz
        signal_noise = 0.5 * np.sin(2 * np.pi * 50 * t)  # 50Hz noise
        x = signal_clean + signal_noise

        # Apply low-pass filter (cutoff around 5Hz)
        y = butterlowpass(
            x=x,
            fpass=5.0,
            fstop=10.0,
            gpass=1.0,
            gstop=40.0,
            fs=fs,
            labelname="test_signal",
        )

        # Check that output has same length
        assert len(y) == len(x)

        # Check that high frequency is attenuated
        # Compare frequency domain
        fft_input = np.fft.fft(x)
        fft_output = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(x), 1 / fs)

        # Find magnitude at 50Hz
        idx_50hz = np.argmin(np.abs(freqs - 50))
        magnitude_before = np.abs(fft_input[idx_50hz])
        magnitude_after = np.abs(fft_output[idx_50hz])

        # High frequency should be attenuated significantly
        assert magnitude_after < magnitude_before * 0.1

    def test_output_dtype(self):
        """Test that output has correct dtype."""
        x = np.random.randn(100)
        y = butterlowpass(x=x, fpass=1.0, fstop=5.0, gpass=1.0, gstop=40.0, fs=50.0)

        assert isinstance(y, np.ndarray)
        assert y.dtype in [np.float64, np.float32]

    def test_no_phase_shift(self):
        """Test that filtfilt prevents phase shift."""
        # Create a step function
        fs = 100.0
        x = np.zeros(200)
        x[100:] = 1.0  # Step at middle

        y = butterlowpass(x=x, fpass=1.0, fstop=5.0, gpass=1.0, gstop=40.0, fs=fs)

        # Find the midpoint of the filtered signal
        midpoint = np.where(y > 0.5)[0][0]

        # Should be close to original step location (100)
        assert abs(midpoint - 100) < 10

    def test_invalid_sampling_frequency(self):
        """Test behavior with invalid sampling frequency.

        Note: Original implementation may not validate, so this tests actual behavior.
        """
        x = np.random.randn(100)

        # These may raise scipy errors or mathematical errors
        # Testing that it doesn't silently succeed
        with pytest.raises((ValueError, ZeroDivisionError, RuntimeError)):
            butterlowpass(x=x, fpass=1.0, fstop=5.0, gpass=1.0, gstop=40.0, fs=0.0)

    def test_invalid_frequencies(self):
        """Test behavior with invalid frequency parameters.

        Note: Original implementation may not validate, testing actual behavior.
        """
        x = np.random.randn(100)
        fs = 100.0

        # fpass >= fstop may cause scipy.signal errors
        with pytest.raises((ValueError, RuntimeError)):
            butterlowpass(x=x, fpass=10.0, fstop=5.0, gpass=1.0, gstop=40.0, fs=fs)


class TestFiltering:
    """Test cases for DataFrame filtering function."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create sample DataFrame with multiple signal columns."""
        fs = 100.0
        t = np.linspace(0, 1, int(fs), endpoint=False)

        # Create signals with different frequencies
        signal1 = np.sin(2 * np.pi * 1 * t) + 0.3 * np.sin(2 * np.pi * 30 * t)
        signal2 = np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(2 * np.pi * 40 * t)
        signal3 = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 50 * t)

        return pd.DataFrame(
            {
                "speed": signal1,
                "throttle": signal2,
                "brake": signal3,
                "timestamp": t,
            }
        )

    def test_basic_dataframe_filtering(self, sample_dataframe):
        """Test that filtering works on DataFrame columns."""
        result = filtering(
            df=sample_dataframe,
            fpass=5.0,
            fstop=10.0,
            gpass=1.0,
            gstop=40.0,
            fs=100.0,
            labelname_list=["speed", "throttle", "brake"],
        )

        # Check shape is preserved
        assert result.shape == sample_dataframe.shape

        # Check columns are preserved
        assert list(result.columns) == list(sample_dataframe.columns)

        # Check that filtered columns are different from original
        assert not np.allclose(result["speed"].values, sample_dataframe["speed"].values)
        assert not np.allclose(result["throttle"].values, sample_dataframe["throttle"].values)

        # Check that unfiltered column remains unchanged
        assert np.allclose(result["timestamp"].values, sample_dataframe["timestamp"].values)

    def test_partial_column_filtering(self, sample_dataframe):
        """Test filtering only specified columns."""
        result = filtering(
            df=sample_dataframe,
            fpass=5.0,
            fstop=10.0,
            gpass=1.0,
            gstop=40.0,
            fs=100.0,
            labelname_list=["speed"],  # Only speed
        )

        # Speed should be filtered
        assert not np.allclose(result["speed"].values, sample_dataframe["speed"].values)

        # Other columns should remain unchanged
        assert np.allclose(result["throttle"].values, sample_dataframe["throttle"].values)
        assert np.allclose(result["brake"].values, sample_dataframe["brake"].values)

    def test_nonexistent_column_handling(self, sample_dataframe):
        """Test that nonexistent columns are skipped gracefully."""
        result = filtering(
            df=sample_dataframe,
            fpass=5.0,
            fstop=10.0,
            gpass=1.0,
            gstop=40.0,
            fs=100.0,
            labelname_list=["speed", "nonexistent_column", "brake"],
        )

        # Should not raise error, just skip nonexistent column
        assert result.shape == sample_dataframe.shape

        # Speed and brake should be filtered
        assert not np.allclose(result["speed"].values, sample_dataframe["speed"].values)
        assert not np.allclose(result["brake"].values, sample_dataframe["brake"].values)

    def test_empty_labelname_list(self, sample_dataframe):
        """Test with empty label list."""
        result = filtering(
            df=sample_dataframe,
            fpass=5.0,
            fstop=10.0,
            gpass=1.0,
            gstop=40.0,
            fs=100.0,
            labelname_list=[],
        )

        # All columns should remain unchanged
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_original_dataframe_unchanged(self, sample_dataframe):
        """Test that original DataFrame is not modified."""
        original_values = sample_dataframe.copy()

        filtering(
            df=sample_dataframe,
            fpass=5.0,
            fstop=10.0,
            gpass=1.0,
            gstop=40.0,
            fs=100.0,
            labelname_list=["speed", "throttle"],
        )

        # Original should be unchanged
        pd.testing.assert_frame_equal(sample_dataframe, original_values)

    def test_invalid_parameters_handling(self, sample_dataframe):
        """Test that invalid filter parameters are handled gracefully.

        The current implementation logs errors but doesn't raise exceptions,
        keeping original data when filtering fails.
        """
        result = filtering(
            df=sample_dataframe,
            fpass=10.0,
            fstop=5.0,  # Invalid: fpass >= fstop
            gpass=1.0,
            gstop=40.0,
            fs=100.0,
            labelname_list=["speed"],
        )

        # Should return DataFrame with original data when filtering fails
        assert result.shape == sample_dataframe.shape
        # Speed column should remain unchanged due to filter error
        assert np.allclose(result["speed"].values, sample_dataframe["speed"].values)

    def test_all_columns_filtering(self, sample_dataframe):
        """Test filtering all numeric columns."""
        numeric_columns = ["speed", "throttle", "brake", "timestamp"]

        result = filtering(
            df=sample_dataframe,
            fpass=5.0,
            fstop=10.0,
            gpass=1.0,
            gstop=40.0,
            fs=100.0,
            labelname_list=numeric_columns,
        )

        # All columns should be filtered
        for col in numeric_columns:
            assert not np.allclose(result[col].values, sample_dataframe[col].values)


class TestFilterIntegration:
    """Integration tests for filter module."""

    def test_realistic_telemetry_filtering(self):
        """Test with realistic telemetry-like data."""
        # Simulate telemetry data at 23Hz for 10 seconds
        fs = 23.0
        duration = 10.0
        n_samples = int(fs * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Create realistic signals with noise
        speed = 50 + 10 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n_samples) * 2
        throttle = 30 + 20 * np.sin(2 * np.pi * 0.3 * t) + np.random.randn(n_samples) * 5
        brake = np.maximum(0, 10 * np.sin(2 * np.pi * 0.2 * t) + np.random.randn(n_samples) * 3)

        df = pd.DataFrame({"speed": speed, "throttle": throttle, "brake": brake, "time": t})

        # Apply realistic filter parameters
        result = filtering(
            df=df,
            fpass=1.0,  # Pass up to 1Hz
            fstop=5.0,  # Stop from 5Hz
            gpass=1.0,
            gstop=40.0,
            fs=fs,
            labelname_list=["speed", "throttle", "brake"],
        )

        # Check that noise is reduced
        for col in ["speed", "throttle", "brake"]:
            # Filtered signal should have lower variance
            assert result[col].std() < df[col].std()

            # Mean should be approximately preserved
            assert abs(result[col].mean() - df[col].mean()) < 1.0

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very short signal
        short_signal = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        result = filtering(
            df=short_signal,
            fpass=0.1,
            fstop=0.5,
            gpass=1.0,
            gstop=40.0,
            fs=10.0,
            labelname_list=["col1"],
        )
        assert result.shape == short_signal.shape

        # Constant signal
        const_signal = pd.DataFrame({"col1": np.ones(100)})
        result = filtering(
            df=const_signal,
            fpass=1.0,
            fstop=5.0,
            gpass=1.0,
            gstop=40.0,
            fs=50.0,
            labelname_list=["col1"],
        )
        # Constant signal should remain constant
        assert np.allclose(result["col1"].values, 1.0, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
