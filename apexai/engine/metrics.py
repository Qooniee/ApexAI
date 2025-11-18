"""Type definitions for training and validation metrics."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt


class TrainingMetrics(TypedDict):
    """Structured metrics returned from training.

    This structure provides a clean interface for metrics collection,
    making it easy to:
    - Log metrics to MLflow in a loop
    - Return metrics for Optuna optimization
    - Reuse trainer in different contexts

    Structure:
        train: Training metrics (tracked every epoch)
            - loss: List of training losses per epoch
        validation: Validation metrics (tracked every epoch)
            - loss: List of validation losses per epoch
            - accuracy: List of validation accuracies per epoch
            - f1_score: List of F1 scores per epoch
            - precision: List of precision scores per epoch
            - recall: List of recall scores per epoch
        confusion_matrix: Final confusion matrices
            - train: Training confusion matrix (numpy array)
            - validation: Validation confusion matrix (numpy array)

    """

    train: dict[str, list[float]]
    validation: dict[str, list[float]]
    confusion_matrix: dict[str, npt.NDArray[np.int_]]
