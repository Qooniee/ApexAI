"""Signal resampling with anti-aliasing filter for telemetry data.

This module provides functions for upsampling and downsampling signals
with proper anti-aliasing filtering to prevent aliasing artifacts.
"""

import logging

import numpy as np
import pandas as pd
from scipy import interpolate

from apexai.signal_processing.filter import butterlowpass

log = logging.getLogger(__name__)


def resample(
    df: pd.DataFrame,
    time_label: str,
    signal_labels: list[str],
    original_fs: float,
    target_fs: float,
    interpolation_kind: str = "linear",
    fill_value: str | float = "extrapolate",
    anti_aliasing: bool = True,
    filter_params: dict | None = None,
) -> pd.DataFrame:
    """Resample signals with anti-aliasing filter.

    Automatically detects upsampling or downsampling and applies appropriate
    processing. For downsampling, applies anti-aliasing filter before resampling
    to prevent aliasing artifacts.

    Args:
        df: Input DataFrame containing time series data.
        time_label: Name of the time column.
        signal_labels: List of signal column names to resample.
        original_fs: Original sampling frequency [Hz].
        target_fs: Target sampling frequency [Hz].
        interpolation_kind: Interpolation method ('linear', 'cubic', 'nearest').
        fill_value: How to handle extrapolation ('extrapolate' or a numeric value).
        anti_aliasing: Whether to apply anti-aliasing filter for downsampling.
        filter_params: Optional custom filter parameters. If None, auto-calculated.

    Returns:
        Resampled DataFrame with new sampling rate.

    Raises:
        ValueError: If parameters are invalid or required columns are missing.

    Examples:
        >>> # Downsample from 100Hz to 10Hz
        >>> resampled = resample(df, 'time', ['speed', 'throttle'], 100.0, 10.0)
        >>>
        >>> # Upsample from 10Hz to 100Hz
        >>> resampled = resample(df, 'time', ['speed', 'throttle'], 10.0, 100.0)

    """
    # Validation
    if time_label not in df.columns:
        raise ValueError(f"Time column '{time_label}' not found in DataFrame")

    for label in signal_labels:
        if label not in df.columns:
            raise ValueError(f"Signal column '{label}' not found in DataFrame")

    if original_fs <= 0 or target_fs <= 0:
        raise ValueError(
            f"Sampling frequencies must be positive: "
            f"original_fs={original_fs}, target_fs={target_fs}"
        )

    # Determine resampling mode
    if target_fs > original_fs:
        mode = "upsample"
    elif target_fs < original_fs:
        mode = "downsample"
    else:
        log.info("Target frequency equals original frequency. No resampling needed.")
        return df.copy()

    log.info(f"Resampling from {original_fs}Hz to {target_fs}Hz ({mode}), signals: {signal_labels}")

    # Calculate new time array
    time_max = df[time_label].max()
    n_samples_new = int(np.round(time_max * target_fs))

    log.info(
        f"Original samples: {len(df)}, Target samples: {n_samples_new}, Duration: {time_max:.2f}s"
    )

    if mode == "downsample":
        return _downsample(
            df=df,
            time_label=time_label,
            signal_labels=signal_labels,
            original_fs=original_fs,
            target_fs=target_fs,
            n_samples_new=n_samples_new,
            interpolation_kind=interpolation_kind,
            fill_value=fill_value,
            anti_aliasing=anti_aliasing,
            filter_params=filter_params,
        )
    else:  # upsample
        return _upsample(
            df=df,
            time_label=time_label,
            signal_labels=signal_labels,
            target_fs=target_fs,
            n_samples_new=n_samples_new,
            interpolation_kind=interpolation_kind,
            fill_value=fill_value,
        )


def _downsample(
    df: pd.DataFrame,
    time_label: str,
    signal_labels: list[str],
    original_fs: float,
    target_fs: float,
    n_samples_new: int,
    interpolation_kind: str,
    fill_value: str | float,
    anti_aliasing: bool,
    filter_params: dict | None,
) -> pd.DataFrame:
    """Downsample with anti-aliasing filter.

    Process:
    1. Apply anti-aliasing low-pass filter (cutoff at target Nyquist frequency)
    2. Interpolate to new sampling rate

    """
    log.info("Downsampling: Applying anti-aliasing filter before resampling")

    # Calculate anti-aliasing filter parameters
    target_nyquist = target_fs / 2.0

    if filter_params is None:
        # Auto-calculate conservative filter parameters
        # Use safer margins to avoid filter instability
        fpass = min(target_nyquist * 0.7, original_fs * 0.4)  # Conservative passband
        fstop = min(target_nyquist * 0.9, original_fs * 0.45)  # Safe stopband
        gpass = 3.0  # Allow more ripple for stability
        gstop = 15.0  # Lower attenuation for stability
    else:
        fpass = filter_params.get("fpass", target_nyquist * 0.7)
        fstop = filter_params.get("fstop", target_nyquist * 0.9)
        gpass = filter_params.get("gpass", 3.0)
        gstop = filter_params.get("gstop", 15.0)

    log.info(
        f"Anti-aliasing filter: fpass={fpass:.2f}Hz, fstop={fstop:.2f}Hz, "
        f"target_nyquist={target_nyquist:.2f}Hz"
    )

    # Apply anti-aliasing filter if enabled
    if anti_aliasing:
        filtered_df = df.copy()
        for label in signal_labels:
            try:
                filtered_df[label] = butterlowpass(
                    x=df[label].values,
                    fpass=fpass,
                    fstop=fstop,
                    gpass=gpass,
                    gstop=gstop,
                    fs=original_fs,
                    labelname=label,
                )
            except Exception as e:
                log.error(f"Anti-aliasing filter failed for '{label}': {e}")
                log.warning(f"Proceeding without filtering for '{label}'")
                filtered_df[label] = df[label].values
    else:
        log.warning("Anti-aliasing disabled - risk of aliasing artifacts!")
        filtered_df = df.copy()

    # Apply physical constraints after filtering
    # Signals that must be non-negative due to physical constraints
    non_negative_signals = ["pbrake_f", "pbrake_r", "speed", "nmot"]
    for label in signal_labels:
        if label in non_negative_signals:
            # Clip negative values to zero (floating-point precision errors)
            negative_count = (filtered_df[label] < 0).sum()
            if negative_count > 0:
                min_value = filtered_df[label].min()
                filtered_df[label] = filtered_df[label].clip(lower=0.0)
                log.info(
                    f"Clipped {negative_count} negative values for '{label}' "
                    f"(min was {min_value:.6e}, likely floating-point precision error)"
                )

    # Create new time array
    time_max = df[time_label].max()
    new_time = np.linspace(0, time_max, n_samples_new)

    # Interpolate to new sampling rate
    resampled_df = pd.DataFrame({time_label: new_time})

    for label in signal_labels:
        log.info(f"Interpolating '{label}' to {target_fs}Hz...")
        try:
            interpolator = interpolate.interp1d(
                filtered_df[time_label].values,
                filtered_df[label].values,
                kind=interpolation_kind,
                fill_value=fill_value,
                bounds_error=False,
            )
            resampled_df[label] = interpolator(new_time)
        except Exception as e:
            log.error(f"Interpolation failed for '{label}': {e}")
            raise

    return resampled_df


def _upsample(
    df: pd.DataFrame,
    time_label: str,
    signal_labels: list[str],
    target_fs: float,
    n_samples_new: int,
    interpolation_kind: str,
    fill_value: str | float,
) -> pd.DataFrame:
    """Upsample using interpolation.

    Process:
    1. Interpolate to new higher sampling rate
    No anti-aliasing filter needed for upsampling.

    """
    log.info("Upsampling: Interpolating to higher sampling rate")

    # Create new time array
    time_max = df[time_label].max()
    new_time = np.linspace(0, time_max, n_samples_new)

    # Interpolate to new sampling rate
    resampled_df = pd.DataFrame({time_label: new_time})

    for label in signal_labels:
        log.info(f"Interpolating '{label}' to {target_fs}Hz...")
        try:
            interpolator = interpolate.interp1d(
                df[time_label].values,
                df[label].values,
                kind=interpolation_kind,
                fill_value=fill_value,
                bounds_error=False,
            )
            resampled_df[label] = interpolator(new_time)
        except Exception as e:
            log.error(f"Interpolation failed for '{label}': {e}")
            raise

    return resampled_df


def resample_dataframe(
    df: pd.DataFrame,
    time_label: str,
    signal_labels: list[str],
    original_fs: float,
    target_fs: float,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for resample function.

    See resample() for detailed documentation.
    """
    return resample(
        df=df,
        time_label=time_label,
        signal_labels=signal_labels,
        original_fs=original_fs,
        target_fs=target_fs,
        **kwargs,
    )


def validate_resampling_params(
    original_fs: float, target_fs: float, signal_length: int
) -> dict[str, bool | list[str] | dict[str, float]]:
    """Validate resampling parameters and return recommended filter settings.

    Args:
        original_fs: Original sampling frequency [Hz].
        target_fs: Target sampling frequency [Hz].
        signal_length: Length of the signal.

    Returns:
        Dictionary with validation results and recommended settings.

    """
    results: dict[str, bool | list[str] | dict[str, float]] = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommended_filter": {},
    }

    warnings: list[str] = []
    errors: list[str] = []

    # Check Nyquist criterion
    if target_fs < original_fs:
        target_nyquist = target_fs / 2.0
        _original_nyquist = original_fs / 2.0

        results["recommended_filter"] = {
            "fpass": target_nyquist * 0.8,
            "fstop": target_nyquist * 0.95,
            "gpass": 1.0,
            "gstop": 40.0,
        }

        if target_fs < original_fs / 10:
            warnings.append(
                f"Large downsampling ratio ({original_fs / target_fs:.1f}x). "
                "Consider multiple stages."
            )

    # Check signal length
    min_samples = 10
    if signal_length < min_samples:
        errors.append(
            f"Signal too short ({signal_length} samples). Need at least {min_samples} samples."
        )
        results["valid"] = False

    results["warnings"] = warnings
    results["errors"] = errors

    return results
