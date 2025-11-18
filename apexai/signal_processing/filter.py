"""This module provides functions for applying Butterworth low-pass filters to signals."""

import logging

import numpy as np
import pandas as pd
from scipy import signal

log = logging.getLogger(__name__)


def butterlowpass(
    x: np.ndarray,
    fpass: float,
    fstop: float,
    gpass: float,
    gstop: float,
    fs: float,
    labelname: str = "Signal[-]",
) -> np.ndarray:
    """Apply a low-pass Butterworth filter to the input signal.

    The filtfilt function is used to prevent phase shift by applying
    the filter in both forward and reverse directions.

    Args:
        x: Input signal array.
        fpass: Passband edge frequency [Hz].
        fstop: Stopband edge frequency [Hz].
        gpass: Maximum loss in passband [dB].
        gstop: Minimum attenuation in stopband [dB].
        fs: Sampling frequency [Hz].
        labelname: Name of the signal for logging purposes.

    Returns:
        Filtered signal array with same shape as input.

    Raises:
        ValueError: If filter parameters are invalid.

    """
    # Parameter validation
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    if fpass <= 0 or fstop <= 0:
        raise ValueError(f"Frequencies must be positive: fpass={fpass}, fstop={fstop}")
    if fpass >= fstop:
        raise ValueError(f"Passband freq ({fpass}) must be < stopband freq ({fstop})")
    if fpass >= fs / 2:
        raise ValueError(f"Passband freq ({fpass}) must be < Nyquist freq ({fs / 2})")

    log.info(f"Applying Butterworth filter to: {labelname}")

    # Calculate normalized frequencies
    dt = 1 / fs
    fn = 1 / (2 * dt)  # Nyquist frequency
    wp = fpass / fn
    ws = fstop / fn

    # Design filter
    n, wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(n, wn, "low")

    # Apply zero-phase filtering
    y = signal.filtfilt(b, a, x)

    log.info(
        f"Filtered {labelname}: order={n}, shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}"
    )

    return y


def filtering(
    df: pd.DataFrame,
    fpass: float,
    fstop: float,
    gpass: float,
    gstop: float,
    fs: float,
    labelname_list: list[str],
) -> pd.DataFrame:
    """Apply Butterworth low-pass filter to multiple columns in DataFrame.

    Args:
        df: Input DataFrame containing signal data.
        fpass: Passband edge frequency [Hz].
        fstop: Stopband edge frequency [Hz].
        gpass: Maximum loss in passband [dB].
        gstop: Minimum attenuation in stopband [dB].
        fs: Sampling frequency [Hz].
        labelname_list: List of column names to apply filter to.

    Returns:
        Filtered DataFrame with same columns and index as input.
        Columns not in labelname_list remain unchanged.

    Raises:
        ValueError: If filter parameters are invalid.

    """
    filtered_df = df.copy()

    for labelname in labelname_list:
        if labelname not in df.columns:
            log.warning(f"Column '{labelname}' not found in DataFrame. Skipping.")
            continue

        try:
            filtered_df[labelname] = butterlowpass(
                x=df[labelname].values,
                fpass=fpass,
                fstop=fstop,
                gpass=gpass,
                gstop=gstop,
                fs=fs,
                labelname=labelname,
            )
        except Exception as e:
            log.error(f"Failed to filter column '{labelname}': {e}")
            # Keep original data if filtering fails
            log.warning(f"Keeping original data for '{labelname}'")

    return filtered_df
