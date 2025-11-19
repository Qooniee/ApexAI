"""Validation inference module for model evaluation.

This module provides validation functionality for deep learning models,
including loss computation, metric calculation, and confusion matrix generation.
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

from apexai.util.tools import get_confusion_matrix

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def validation(
    config: "DictConfig",
    model: "nn.Module",
    val_loader: "DataLoader",
    loss_fn: "nn.Module",
    cur_epoch: int,
    max_epoch: int,
    logger_vis=None,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """Perform validation on the validation dataset.

    Args:
        config: Configuration object containing model and data settings.
        model: Neural network model to validate.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function for computing validation loss.
        cur_epoch: Current epoch number.
        max_epoch: Maximum number of epochs (for final validation logging).
        logger_vis: Optional logger for visualization (unused, kept for compatibility).

    Returns:
        Tuple containing:
            - avg_valid_loss: Average validation loss
            - acc: Validation accuracy
            - precision: Macro-averaged precision
            - recall: Macro-averaged recall
            - f1: Macro-averaged F1 score
            - confusion: Confusion matrix

    Examples:
        >>> val_loss, acc, prec, rec, f1, cm = validation(
        ...     config, model, val_loader, loss_fn, 10, 100, []
        ... )
        >>> logger.info(f"Validation accuracy: {acc:.4f}")

    """
    valid_loss = 0
    num_classes = config.model.architecture.num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    confusion = np.zeros((num_classes, num_classes))
    # Store prediction results for all validation data
    pred_label_list: np.ndarray = np.array([])
    # Store ground truth labels for all validation data
    true_label_list: np.ndarray = np.array([])

    with torch.no_grad():
        model.eval()
        # iteration is used later for computing average loss (line 110)
        for iteration, (x, t) in enumerate(val_loader):  # noqa: B007
            x, t = x.to(device), t.to(device)
            # Model outputs logits
            logits = model(x)
            # Accumulate validation loss for each iteration
            valid_loss += loss_fn(logits, t)
            # Ground truth labels for current iteration
            rows = t.cpu().numpy()
            # During inference, apply softmax to logits to get probabilities
            # then argmax
            probs = torch.softmax(logits, dim=1)
            # Predicted labels for current iteration
            cols = probs.argmax(1).cpu().numpy()
            # Store ground truth labels from current iteration
            true_label_list = np.hstack((true_label_list, rows))
            # Store predicted labels from current iteration
            pred_label_list = np.hstack((pred_label_list, cols))

    # Calculate confusion matrix for validation data
    confusion = get_confusion_matrix(
        true_label_list, pred_label_list, config.model.architecture.num_classes
    )
    # Divide accumulated validation loss by iteration count to get average
    if isinstance(valid_loss, torch.Tensor):
        avg_valid_loss = float(valid_loss.cpu().numpy()) / (iteration + 1)
    else:
        avg_valid_loss = float(valid_loss) / (iteration + 1)

    acc = accuracy_score(true_label_list, pred_label_list)
    recall = recall_score(true_label_list, pred_label_list, average="macro")
    precision = precision_score(true_label_list, pred_label_list, average="macro")
    f1 = f1_score(true_label_list, pred_label_list, average="macro")

    # Confusion matrix will be returned and logged by caller
    return avg_valid_loss, acc, precision, recall, f1, confusion
