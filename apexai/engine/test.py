"""Test module for model evaluation on test dataset.

This module provides testing functionality to evaluate how robust
the developed model is on held-out test data.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from util.tools import get_confusion_matrix, log_confusion_matrix_to_mlflow

from .make_graph import testgraph

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def testing(
    config: "DictConfig",
    model: "nn.Module",
    test_loader: "DataLoader",
    loss_fn: "nn.Module",
    logger_vis: list,
) -> tuple[float, float]:
    """Evaluate model performance on test dataset.

    Args:
        config: Configuration object containing model and data settings.
        model: Neural network model to test.
        test_loader: DataLoader for test data.
        loss_fn: Loss function for computing test loss.
        logger_vis: Visualization logger (unused, kept for compatibility).

    Returns:
        Tuple containing:
            - avg_test_loss: Average test loss
            - acc: Test accuracy

    Examples:
        >>> test_loss, test_acc = testing(
        ...     config, model, test_loader, loss_fn, []
        ... )
        >>> logger.info(f"Test accuracy: {test_acc:.4f}")

    """
    # Visdom removed - using MLflow for visualization instead
    model.eval()

    test_loss = 0
    num_classes = config.model.features.num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    confusion = np.zeros((num_classes, num_classes))
    # Store inference results for all test data
    pred_label_list: list[int] = []
    # Store ground truth labels for all test data
    true_label_list: list[int] = []
    # iteration is used later for computing average loss (line 91)
    for iteration, (x, t) in enumerate(test_loader):  # noqa: B007
        x, t = x.to(device), t.to(device)
        y = model(x)
        # Accumulate inference loss for each iteration on test data
        test_loss += loss_fn(y, t)

        # Ground truth labels for test data in current iteration
        rows = t.cpu().numpy()
        # Inference results for test data in current iteration
        cols = y.argmax(1).cpu().numpy()

        # Store predicted labels
        pred_label_list = np.hstack((pred_label_list, cols))
        # Store ground truth labels
        true_label_list = np.hstack((true_label_list, rows))

    # Calculate confusion matrix for test data
    confusion = get_confusion_matrix(
        true_label_list, pred_label_list, config.model.features.num_classes
    )

    # Divide accumulated test loss by iteration count to get average
    avg_test_loss = test_loss / (iteration + 1)

    # Inference accuracy during test phase
    acc = accuracy_score(true_label_list, pred_label_list)
    recall_score(true_label_list, pred_label_list, average="macro")
    precision_score(true_label_list, pred_label_list, average="macro")
    f1 = f1_score(true_label_list, pred_label_list, average="macro")

    testgraph(avg_test_loss, f1, confusion, config)

    # Log test confusion matrix to MLflow (displayed on MLflow UI)
    try:
        # Save test confusion matrix to MLflow
        log_confusion_matrix_to_mlflow(cm=confusion, prefix="test_", normalize=False, config=config)

        # Also save normalized version
        log_confusion_matrix_to_mlflow(cm=confusion, prefix="test_", normalize=True, config=config)

    except Exception:
        logger.exception("Failed to log test confusion matrix to MLflow")

    return avg_test_loss, acc
