"""Utility tools for confusion matrix generation and MLflow logging.

This module provides helper functions for:
- Computing confusion matrices
- Logging confusion matrices to MLflow with visualization
- Calculating and logging per-class metrics
- Managing dataset file lengths
"""

import collections
import logging
import os
import random
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_confusion_matrix(
    truelabel: np.ndarray,
    predlabel: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Generate confusion matrix from true and predicted labels.

    Args:
        truelabel: Array of ground truth labels.
        predlabel: Array of predicted labels.
        num_classes: Number of classes in the classification problem.

    Returns:
        Confusion matrix as a 2D numpy array of shape (num_classes, num_classes).

    Examples:
        >>> true = np.array([0, 1, 2, 0, 1])
        >>> pred = np.array([0, 2, 2, 0, 1])
        >>> cm = get_confusion_matrix(true, pred, 3)
        >>> cm.shape
        (3, 3)

    """
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    # Build confusion matrix for all iterations in one epoch
    # Number of data points equals x_train.__len()__
    for t, pred in zip(truelabel, predlabel, strict=False):
        confusion_matrix[int(t), int(pred)] += 1

    return confusion_matrix


def log_confusion_matrix_to_mlflow(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    prefix: str = "",
    normalize: bool = False,
    config: "DictConfig | None" = None,
) -> None:
    """Log confusion matrix as image to MLflow for display in MLflow UI.

    Args:
        cm: Confusion matrix as numpy array.
        class_names: List of class names for axis labels.
        prefix: Prefix for filename (e.g., 'train_', 'val_', 'test_').
        normalize: Whether to normalize the confusion matrix.
        config: Configuration object for retrieving class names.

    Examples:
        >>> cm = np.array([[10, 2], [1, 15]])
        >>> log_confusion_matrix_to_mlflow(
        ...     cm, class_names=['Class A', 'Class B'], prefix='test_'
        ... )

    """
    try:
        # Get class names from config file
        if class_names is None:
            if (
                config
                and hasattr(config, "data")
                and hasattr(config.data, "labels")
                and hasattr(config.data.labels, "class_names")
            ):
                # Get from config file (dict format {0: "Walk", 1: "Vehicle", ...})
                class_names_dict = config.data.labels.class_names
                class_names = [class_names_dict[i] for i in sorted(class_names_dict.keys())]
            else:
                # Fallback (legacy hardcoded names)
                class_names = ["Walk", "Vehicle", "Run", "Stair"]

        # Normalization option
        if normalize:
            # Avoid division by zero
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Replace 0 with 1
            cm_normalized = cm.astype("float") / row_sums
            cm_to_plot = cm_normalized
            fmt = ".2f"
            title_suffix = " (Normalized)"
        else:
            # Ensure integer type for non-normalized matrix
            cm_to_plot = cm.astype(int)
            fmt = "d"
            title_suffix = ""

        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_to_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5,
        )

        plt.title(f"{prefix}Confusion Matrix{title_suffix}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # Save as image to MLflow (displayed on MLflow UI)
        filename = f"{prefix}confusion_matrix{'_normalized' if normalize else ''}.png"
        mlflow.log_figure(plt.gcf(), filename)

        # Also save numeric data as JSON
        matrix_data = {
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "normalized": normalize,
        }
        json_filename = f"{prefix}confusion_matrix{'_normalized' if normalize else ''}.json"
        mlflow.log_dict(matrix_data, json_filename)

        # Calculate and log various metrics from confusion matrix
        # Only log metrics for non-normalized matrix to avoid duplicates
        # Use step parameter: 0 for train, 1 for validation to avoid collisions
        if not normalize:
            step_value = 0 if "train" in prefix else 1
            log_confusion_matrix_metrics(cm, class_names, prefix, step=step_value)

        plt.close()

    except Exception:
        logger.exception(
            "Failed to log confusion matrix to MLflow with prefix '%s'",
            prefix,
        )


def log_confusion_matrix_metrics(
    cm: np.ndarray,
    class_names: list[str],
    prefix: str = "",
    step: int | None = None,
) -> None:
    """Calculate detailed metrics from confusion matrix and log to MLflow.

    Computes and logs per-class metrics including precision, recall,
    F1-score, and specificity for each class.

    Args:
        cm: Confusion matrix as numpy array.
        class_names: List of class names.
        prefix: Prefix for metric names (e.g., 'train_', 'val_').
        step: Step number for MLflow logging (use different values to avoid collisions).

    """
    try:
        for i, class_name in enumerate(class_names):
            # True Positive, False Positive, False Negative
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            # Calculate metrics for each class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Log per-class metrics to MLflow with step parameter
            mlflow.log_metric(f"{prefix}precision_{class_name}", precision, step=step)
            mlflow.log_metric(f"{prefix}recall_{class_name}", recall, step=step)
            mlflow.log_metric(f"{prefix}f1_{class_name}", f1, step=step)
            mlflow.log_metric(f"{prefix}specificity_{class_name}", specificity, step=step)

        # Overall metrics
        total_samples = cm.sum()
        accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0.0
        mlflow.log_metric(f"{prefix}confusion_matrix_accuracy", accuracy, step=step)

    except Exception:
        logger.exception(
            "Failed to log confusion matrix metrics with prefix '%s'",
            prefix,
        )


def get_all_file_length(
    train_dataset_list: list[str],
    valid_dataset_list: list[str],
    test_dataset_list: list[str],
    settings: "DictConfig",
) -> dict[str, list[int]]:
    """Get the length of all dataset files.

    Reads CSV files in chunks and calculates total number of samples
    in each file for train, validation, and test datasets.

    Args:
        train_dataset_list: List of paths to training dataset CSV files.
        valid_dataset_list: List of paths to validation dataset CSV files.
        test_dataset_list: List of paths to test dataset CSV files.
        settings: Configuration object with file reading parameters.

    Returns:
        Dictionary with keys 'train', 'valid', 'test', each containing
        a list of dataset lengths.

    Examples:
        >>> lengths = get_all_file_length(
        ...     train_files, valid_files, test_files, config
        ... )
        >>> lengths['train']
        [10000, 15000, 12000]

    """
    dataset_len_list: dict[str, list[int]] = collections.OrderedDict(
        {"train": [], "valid": [], "test": []}
    )

    # Load training data
    for train in train_dataset_list:
        train_chunks = pd.read_csv(
            train,
            chunksize=10000,
            encoding=settings.config["System"]["Encoding"],
            sep=settings.config["System"]["Deliminator"]["CSV"],
            header=settings.config["System"]["Header_pos"]["RT"],
        )
        train_dataset = pd.concat((data for data in train_chunks), ignore_index=True)
        dataset_len_list["train"].append(len(train_dataset))

    # Load validation data
    for valid in valid_dataset_list:
        valid_chunks = pd.read_csv(
            valid,
            chunksize=10000,
            encoding=settings.config["System"]["Encoding"],
            sep=settings.config["System"]["Deliminator"]["CSV"],
            header=settings.config["System"]["Header_pos"]["RT"],
        )
        valid_dataset = pd.concat((data for data in valid_chunks), ignore_index=True)
        dataset_len_list["valid"].append(len(valid_dataset))

    # Load test data
    for test in test_dataset_list:
        test_chunks = pd.read_csv(
            test,
            chunksize=10000,
            encoding=settings.config["System"]["Encoding"],
            sep=settings.config["System"]["Deliminator"]["CSV"],
            header=settings.config["System"]["Header_pos"]["RT"],
        )
        test_dataset = pd.concat((data for data in test_chunks), ignore_index=True)
        dataset_len_list["test"].append(len(test_dataset))

    return dataset_len_list


def seed_everything(seed):
    """Set seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic CuBLAS operations (GRU, LSTM, etc.) on CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Enable full determinism
    g = torch.Generator()
    g.manual_seed(seed)

    def _seed_worker(worker_id: int):
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return g, _seed_worker


def validate_data_shape(
    actual_seq_len: int,
    actual_feature_size: int,
    expected_seq_len: int,
    expected_feature_size: int,
    logger: logging.Logger,
):
    """Validate data shape."""
    logger.info("--- Data Shape Validation ---")
    logger.info(f"Actual data shape: seq_len={actual_seq_len}, features={actual_feature_size}")

    if expected_seq_len is not None or expected_feature_size is not None:
        logger.info(f"Expected shape: seq_len={expected_seq_len}, features={expected_feature_size}")

        if expected_seq_len is not None and actual_seq_len != expected_seq_len:
            logger.warning(
                f"⚠️  Sequence length mismatch: Expected {expected_seq_len}, Got {actual_seq_len}"
            )
            logger.warning("This might indicate:")
            logger.warning("  - Different windowing/segmentation configuration")
            logger.warning("  - Truncated or corrupted data files")
            logger.warning("  - Changed preprocessing pipeline")

        if expected_feature_size is not None and actual_feature_size != expected_feature_size:
            logger.warning(
                f"⚠️  Feature dimension mismatch: "
                f"Expected {expected_feature_size}, Got {actual_feature_size}"
            )
            logger.warning("This might indicate:")
            logger.warning("  - New sensor features added/removed")
            logger.warning("  - Different preprocessing configuration")
            logger.warning("  - Changed feature selection settings")

        if (expected_seq_len is None or actual_seq_len == expected_seq_len) and (
            expected_feature_size is None or actual_feature_size == expected_feature_size
        ):
            logger.info("✅ Data shape matches expectations")
    else:
        logger.info("No expected shape configured - using actual data dimensions")

    logger.info(
        f"✅ Using actual data shape: seq_len={actual_seq_len}, features={actual_feature_size}"
    )
    logger.info("-----------------------------")
