"""Data preprocessing utilities for time series data.

This module provides functions for:
- Overlapping window extraction from time series data
- Batch generation from multiple dataset files
- Batch normalization
- Numerical gradient and jerk calculation
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

seed = 32
np.random.seed(seed)


def apply_minmax_normalization(
    data: pd.DataFrame,
    min_values: list[float],
    max_values: list[float],
) -> pd.DataFrame:
    """Apply min-max normalization with clipping to [0, 1] range.

    Normalizes each feature to [0, 1] range using the formula:
        normalized = (x - min) / (max - min)

    Values outside the [min, max] range are clipped to [0, 1].

    Args:
        data: Input DataFrame with features to normalize.
        min_values: Minimum value for each feature.
        max_values: Maximum value for each feature.

    Returns:
        Normalized DataFrame with values in [0, 1] range.

    Examples:
        >>> data = pd.DataFrame({'feat1': [0, 50, 100, 150], 'feat2': [-1, 0, 1, 2]})
        >>> normalized = apply_minmax_normalization(data, [0.0, -1.0], [100.0, 1.0])
        >>> normalized['feat1'].max() <= 1.0
        True

    """
    normalized_data = data.copy()

    for i, col in enumerate(data.columns):
        min_val = min_values[i]
        max_val = max_values[i]

        # Min-max normalization
        normalized_data[col] = (data[col] - min_val) / (max_val - min_val)

        # Clip to [0, 1] range
        normalized_data[col] = np.clip(normalized_data[col], 0.0, 1.0)

    return normalized_data


def apply_zscore_normalization(
    data: pd.DataFrame,
    mean_values: list[float],
    std_values: list[float],
    clip_sigma: float = 3.0,
) -> pd.DataFrame:
    """Apply Z-score (standard) normalization with optional clipping.

    Normalizes each feature to mean=0, std=1 using the formula:
        normalized = (x - mean) / std

    Optionally clips values to ±N standard deviations to handle outliers.

    Args:
        data: Input DataFrame with features to normalize.
        mean_values: Mean value for each feature.
        std_values: Standard deviation for each feature.
        clip_sigma: Number of standard deviations for clipping (e.g., 3.0 for ±3σ).
                   If None, no clipping is applied.

    Returns:
        Normalized DataFrame with mean≈0, std≈1.

    Examples:
        >>> data = pd.DataFrame({'feat1': [0, 50, 100], 'feat2': [-1, 0, 1]})
        >>> normalized = apply_zscore_normalization(data, [50.0, 0.0], [25.0, 0.5])
        >>> abs(normalized['feat1'].mean()) < 0.1  # Close to 0
        True

    """
    normalized_data = data.copy()

    for i, col in enumerate(data.columns):
        mean_val = mean_values[i]
        std_val = std_values[i]

        # Avoid division by zero
        if std_val == 0:
            logger.warning(f"Standard deviation is 0 for column '{col}', setting to 0")
            normalized_data[col] = 0.0
            continue

        # Z-score normalization
        normalized_data[col] = (data[col] - mean_val) / std_val

        # Optional clipping to ±N sigma
        if clip_sigma is not None:
            normalized_data[col] = np.clip(normalized_data[col], -clip_sigma, clip_sigma)

    return normalized_data


def calculate_global_statistics(
    x_dataset_list: list[str],
    features: list[str],
    config: "DictConfig",
) -> tuple[list[float], list[float]]:
    """Calculate global mean and std across all dataset files.

    Args:
        x_dataset_list: List of paths to feature CSV files.
        features: List of feature column names.
        config: Configuration object with data loading parameters.

    Returns:
        Tuple of (mean_values, std_values) for each feature.

    """
    # Use Welford's online algorithm for numerical stability
    n = 0
    mean = np.zeros(len(features))
    M2 = np.zeros(len(features))

    for x_data_path in x_dataset_list:
        x_chunks = pd.read_csv(
            x_data_path,
            chunksize=10000,
            encoding=config.data.loading.encoding,
            sep=config.data.loading.delimiter,
            header=config.data.loading.header_pos.common,
        )
        x_data = pd.concat((data for data in x_chunks), ignore_index=True)
        x_data = x_data[features]

        # Welford's online algorithm
        for _, row in x_data.iterrows():
            n += 1
            delta = row.values - mean
            mean += delta / n
            delta2 = row.values - mean
            M2 += delta * delta2

    # Calculate variance and std
    if n < 2:
        variance = np.zeros(len(features))
    else:
        variance = M2 / (n - 1)

    std = np.sqrt(variance)

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    return mean.tolist(), std.tolist()


def overlapping(
    data: pd.DataFrame,
    samplerate: int,
    fs: int,
    overlap_rate: int,
) -> tuple[np.ndarray, int, float]:
    """Extract overlapping windows from input data.

    Divides time series data into frames of specified size with overlap.
    Returns the extracted frames and their temporal information for
    spectrogram visualization.

    Args:
        data: Input time series DataFrame.
        samplerate: Sampling rate in Hz.
        fs: Frame size (number of samples per frame).
        overlap_rate: Overlap percentage (0-100).

    Returns:
        Tuple containing:
            - array: Overlapping frames of shape (n_frames, frame_size, n_features)
            - n_ave: Number of extracted frames
            - final_time: Time of the last extracted frame

    Examples:
        >>> data = pd.DataFrame(np.random.randn(1000, 3))
        >>> frames, n_frames, end_time = overlapping(data, 100, 100, 50)
        >>> frames.shape
        (19, 100, 3)

    """
    # Total data points
    ts = len(data) / samplerate
    # Frame period
    fc = fs / samplerate
    # Frame shift width with overlap
    x_ol = fs * (1 - (overlap_rate / 100))
    # Number of frames to extract (number of data for averaging)
    n_ave = int((ts - (fc * (overlap_rate / 100))) / (fc * (1 - (overlap_rate / 100))))

    # Array to store extracted data
    array = []
    # Loop to extract data
    for i in range(n_ave):
        # Update extraction position for each loop
        ps = int(x_ol * i)
        # Extract frame_size samples from position ps and add to array
        array.append(data.values[ps : ps + fs : 1])
    final_time = (ps + fs) / samplerate
    array_np = np.array(array)
    # Return extracted overlapping data array and number of data points
    return array_np, n_ave, final_time


def generate_batch(
    x_dataset_list: list[str],
    y_dataset_list: list[str],
    config: "DictConfig",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate batches from dataset file lists with overlapping windows.

    Reads multiple CSV files, applies overlapping window extraction,
    and combines them into batch arrays for training.

    IMPORTANT: Normalization is applied ONCE to all concatenated data to ensure
    reproducibility and consistent floating-point operations.

    Args:
        x_dataset_list: List of paths to feature CSV files.
        y_dataset_list: List of paths to label CSV files.
        config: Configuration object with preprocessing parameters.

    Returns:
        Tuple containing:
            - res_x_batch: Feature batch of shape (batch, seq_len, features)
            - res_y_batch: Label batch of shape (batch,)

    Examples:
        >>> x_files = ['data/x_train_001.csv', 'data/x_train_002.csv']
        >>> y_files = ['data/y_train_001.csv', 'data/y_train_002.csv']
        >>> x_batch, y_batch = generate_batch(x_files, y_files, config)

    """
    overlap_rate = int(config.data.signal_processing.fft.overlap_ratio * 100)
    delta_f = config.data.signal_processing.fft.delta_f
    fs_trans = config.data.signal_processing.resample.target_frequency
    frame_size = int(fs_trans / delta_f)

    # Calculate global statistics for z-score normalization if needed
    # IMPORTANT: Statistics should be calculated from TRAINING data only!
    # For validation/test data, use pre-calculated training statistics.
    global_mean = None
    global_std = None
    if (
        config.data.preprocessing.normalization.enabled
        and config.data.preprocessing.normalization.method == "zscore"
        and config.data.preprocessing.normalization.zscore.mean is None
    ):
        logger.warning(
            "⚠️  Z-score statistics not provided in config. Calculating from current dataset..."
        )
        logger.warning(
            "⚠️  IMPORTANT: If this is validation/test data, "
            "you MUST use training data statistics instead!"
        )
        logger.warning(
            "⚠️  Please set config.data.preprocessing.normalization.zscore.mean "
            "and .std from training data."
        )
        global_mean, global_std = calculate_global_statistics(
            x_dataset_list, config.data.preprocessing.features, config
        )
        logger.info(f"Calculated mean: {global_mean}")
        logger.info(f"Calculated std: {global_std}")
        logger.info(
            "💡 Save these values to config for validation/test splits:\n"
            "   zscore:\n"
            f"     mean: {global_mean}\n"
            f"     std: {global_std}\n"
            f"     clip_sigma: 3.0"
        )

    # STEP 1: Load all CSV files and store as list of (x_data, y_data) tuples
    # This ensures we process all data consistently
    data_list = []
    for x_data_path, y_data_path in zip(x_dataset_list, y_dataset_list, strict=False):
        # Load x_data
        x_chunks = pd.read_csv(
            x_data_path,
            chunksize=10000,
            encoding=config.data.loading.encoding,
            sep=config.data.loading.delimiter,
            header=config.data.loading.header_pos.common,
        )
        x_data = pd.concat((data for data in x_chunks), ignore_index=True)
        x_data = x_data[config.data.preprocessing.features]

        # Load y_data
        y_chunks = pd.read_csv(
            y_data_path,
            chunksize=10000,
            encoding=config.data.loading.encoding,
            sep=config.data.loading.delimiter,
            header=config.data.loading.header_pos.common,
        )
        y_data = pd.concat((data for data in y_chunks), ignore_index=True)

        # Convert string vehicle IDs to numeric labels
        if "vehicle_id" in y_data.columns and hasattr(config.data.labels, "vehicle_id_mapping"):
            vehicle_id_mapping = dict(config.data.labels.vehicle_id_mapping)
            y_data["vehicle_id"] = y_data["vehicle_id"].map(vehicle_id_mapping)

        data_list.append((x_data, y_data))

    # STEP 2: Concatenate all x_data, apply normalization ONCE, then split back
    # This ensures reproducible floating-point operations regardless of file order
    if config.data.preprocessing.normalization.enabled:
        # Concatenate all x_data
        all_x_data = pd.concat([x_data for x_data, _ in data_list], ignore_index=True)

        method = config.data.preprocessing.normalization.method
        if method == "minmax":
            all_x_data_normalized = apply_minmax_normalization(
                all_x_data,
                config.data.preprocessing.normalization.min_max.min,
                config.data.preprocessing.normalization.min_max.max,
            )
        elif method == "zscore":
            # Use global statistics (calculated once for all data)
            if global_mean is not None:
                mean_values = global_mean
                std_values = global_std
            else:
                # Use pre-calculated statistics from config
                mean_values = config.data.preprocessing.normalization.zscore.mean
                std_values = config.data.preprocessing.normalization.zscore.std

            clip_sigma = config.data.preprocessing.normalization.zscore.clip_sigma
            if std_values is None:
                raise ValueError("std_values cannot be None for z-score normalization")
            all_x_data_normalized = apply_zscore_normalization(
                all_x_data,
                mean_values,
                std_values,
                clip_sigma=clip_sigma,
            )
        else:
            logger.warning(f"Unknown normalization method: {method}. Skipping normalization.")
            all_x_data_normalized = all_x_data

        # Split back into individual DataFrames matching original structure
        start_idx = 0
        normalized_data_list = []
        for x_data, y_data in data_list:
            end_idx = start_idx + len(x_data)
            x_data_normalized = all_x_data_normalized.iloc[start_idx:end_idx].reset_index(drop=True)
            normalized_data_list.append((x_data_normalized, y_data))
            start_idx = end_idx
    else:
        # No normalization, use original data
        normalized_data_list = data_list

    # STEP 3: Apply overlapping window extraction to each normalized file
    res_x_batch: np.ndarray = np.array([])
    res_y_batch: np.ndarray = np.array([])
    flag = False

    for x_data, y_data in normalized_data_list:
        # batch => batch_size, sequence_length, features
        x_batch, _n_x_batch, _final_time_x_batch = overlapping(
            x_data,
            fs_trans,
            frame_size,
            overlap_rate,
        )
        # batch => batch_size, sequence_length, label (1D)
        y_batch, _n_y_batch, _final_time_y_batch = overlapping(
            y_data,
            fs_trans,
            frame_size,
            overlap_rate,
        )

        if not flag:
            # Convert to: sequence_length, features, batch_size
            res_x_batch = x_batch.transpose(1, 2, 0)
            flag = True
        else:
            res_x_batch = np.block([res_x_batch, x_batch.transpose(1, 2, 0)])

        # Use median of each window as label
        res_y_batch = np.append(
            res_y_batch,
            np.array([np.median(y_batch[i].flatten()) for i in range(len(y_batch))]),
        )

    # Convert back to: batch_size, sequence_length, features
    res_x_batch = res_x_batch.transpose(2, 0, 1)
    return res_x_batch, res_y_batch


class BatchNorm:
    """Batch normalization for time series data.

    Computes mean and standard deviation statistics from training data
    and applies normalization to training/validation/test batches.
    """

    def fit_transform(
        self,
        x_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit normalization parameters and transform batch.

        Args:
            x_batch: Input batch of shape (batch, seq_len, features).

        Returns:
            Tuple containing:
                - fit_x: Normalized batch
                - mu_scale: Mean for each feature
                - std_scale: Std deviation for each feature

        """
        mu_scale, std_scale = self.fit(x_batch)
        fit_x = self.transform(x_batch, mu_scale, std_scale)
        return fit_x, mu_scale, std_scale

    def fit(
        self,
        x_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate mean and standard deviation for each feature.

        Args:
            x_batch: Input batch of shape (batch, seq_len, features).

        Returns:
            Tuple of (mu_scale, std_scale) for each feature.

        """
        # Calculate mean for entire batch×seq_len matrix per feature
        mu_scale = np.array([x_batch[:, :, i].mean() for i in range(x_batch.shape[2])])
        # Calculate std for entire batch×seq_len matrix per feature
        std_scale = np.array([x_batch[:, :, i].std() for i in range(x_batch.shape[2])])
        return mu_scale, std_scale

    def transform(
        self,
        x_batch: np.ndarray,
        mu_scale: np.ndarray,
        std_scale: np.ndarray,
    ) -> np.ndarray:
        """Apply normalization using precomputed statistics.

        Args:
            x_batch: Input batch to normalize.
            mu_scale: Mean for each feature.
            std_scale: Std deviation for each feature.

        Returns:
            Normalized batch.

        """
        return (x_batch - mu_scale) * (1 / std_scale)


def numerical_gradient(
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Calculate numerical gradient using backward difference.

    Args:
        x: Data array.
        dt: Sampling time in seconds.

    Returns:
        Gradient array.

    Examples:
        >>> x = np.array([0, 1, 4, 9, 16])
        >>> grad = numerical_gradient(x, 0.1)
        >>> len(grad)
        5

    """
    # Backward Difference: f(x + h) - f(x)
    diff_array = np.float64(x[1:]) - np.float64(x[0:-1])
    diff = np.insert(diff_array, 0, diff_array[0])
    return diff / (dt + 1e-8)


def calc_jerk(
    gx: np.ndarray,
    gy: np.ndarray,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate jerk using Gx and Gy acceleration data.

    Args:
        gx: X-axis acceleration in g units.
        gy: Y-axis acceleration in g units.
        time: Time array.

    Returns:
        Tuple containing:
            - j: Composite jerk magnitude
            - jerk_x: X-axis jerk
            - jerk_y: Y-axis jerk

    Examples:
        >>> gx = np.array([0.0, 0.1, 0.2, 0.3])
        >>> gy = np.array([0.0, 0.05, 0.1, 0.15])
        >>> time = np.array([0.0, 0.01, 0.02, 0.03])
        >>> j, jx, jy = calc_jerk(gx, gy, time)

    """
    dt = time[1] - time[0]
    jerk_x = numerical_gradient(gx, dt)
    jerk_y = numerical_gradient(gy, dt)
    j = np.sqrt(np.array(jerk_x) ** 2 + np.array(jerk_y) ** 2)
    return j, jerk_x, jerk_y
