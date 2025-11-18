"""Training loop for time-series classification models."""

import logging
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from util.tools import get_confusion_matrix, seed_everything

from .inference import validation
from .metrics import TrainingMetrics

# Set random seed for reproducibility
# SEED = 32
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# os.environ["PYTHONHASHSEED"] = str(SEED)

# Configure logging
logger = logging.getLogger(__name__)


def train_model(
    config: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    num_epoch: int,
    scheduler: LRScheduler | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mlflow_logger: Any | None = None,
) -> tuple[nn.Module, TrainingMetrics]:
    """Train a model with the given configuration.

    This function implements the main training loop, including:
    - Forward and backward propagation
    - Gradient clipping
    - Learning rate scheduling
    - Validation after each epoch
    - Metric tracking (loss, accuracy, precision, recall, F1)
    - Confusion matrix generation

    Args:
        config: Hydra configuration object containing all hyperparameters.
        model: PyTorch model to train.
        optimizer: Optimizer for updating model weights.
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        num_epoch: Number of training epochs.
        scheduler: Learning rate scheduler (optional).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        mlflow_logger: MLflow logger instance for nested runs (e.g., Optuna trials).
                       Not used in current implementation.

    Returns:
        A tuple containing:
            - Trained model (nn.Module)
            - Training metrics (TrainingMetrics): Structured dictionary with:
                - train: dict with "loss" key
                - validation: dict with "loss", "accuracy", "f1_score", "precision", "recall" keys
                - confusion_matrix: dict with "train" and "validation" keys

    """
    g, seed_worker = seed_everything(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize metric tracking lists
    res_train_loss: list[float] = []  # Average training loss per epoch
    res_val_loss: list[float] = []  # Average validation loss per epoch
    res_train_acc: list[float] = []  # Training accuracy per epoch
    res_val_acc: list[float] = []  # Validation accuracy per epoch

    res_train_recall: list[float] = []  # Training recall per epoch
    res_train_precision: list[float] = []  # Training precision per epoch
    res_train_f1: list[float] = []  # Training F1 score per epoch

    res_val_precision: list[float] = []  # Validation precision per epoch
    res_val_recall: list[float] = []  # Validation recall per epoch
    res_val_f1: list[float] = []  # Validation F1 score per epoch

    y_train_pred_list: list[int] = []  # Predicted labels per epoch (last batch)
    t_train_list: list[int] = []  # True labels per epoch (last batch)

    lr_list: list[float] = []  # Learning rate per iteration

    num_classes = config.model.architecture.num_classes

    val_acc_best = 0.0

    for epoch in range(1, num_epoch + 1):
        iter_train_list: list[float] = []  # Loss per iteration in this epoch
        train_epoch_loss = 0.0  # Cumulative training loss for this epoch
        train_confusion = np.zeros((num_classes, num_classes))  # Confusion matrix for training

        pred_label_list: np.ndarray = np.array([])  # Predicted labels for all iterations
        true_label_list: np.ndarray = np.array([])  # True labels for all iterations

        model.train()
        model.to(device)
        # iteration is used later for computing average loss (line 171)
        for iteration, (x, t) in enumerate(train_loader):  # noqa: B007
            x, t = x.to(device), t.to(device)

            # DEBUG: Log first 16 batches data for reproducibility verification
            if epoch == 1 and iteration < 16:
                batch_sum = x.sum().item()
                label_sum = t.sum().item()
                logger.info(f"[DEBUG] Epoch 1, Batch {iteration} - Data sum: {batch_sum:.6f}")
                logger.info(f"[DEBUG] Epoch 1, Batch {iteration} - Label sum: {label_sum:.6f}")

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y = model(x)
            train_loss = loss_fn(y, t)

            # Backward pass
            train_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.training.max_norm)

            # Update weights
            optimizer.step()

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                lr_list.append(scheduler.get_last_lr()[0])
            else:
                lr_list.append(config.training.optimizer.learning_rate)

            # Track loss
            iter_train_list.append(train_loss.item())
            train_epoch_loss += train_loss.item()

            # Get predictions
            pred = y.argmax(1)  # Get class with highest probability

            rows_t = t.cpu().numpy()  # True labels
            cols_pred = pred.cpu().numpy()  # Predicted labels

            # Accumulate predictions and true labels for entire epoch
            pred_label_list = np.hstack((pred_label_list, cols_pred))
            true_label_list = np.hstack((true_label_list, rows_t))

            del train_loss

        # Compute confusion matrix for this epoch
        train_confusion = get_confusion_matrix(
            truelabel=true_label_list,
            predlabel=pred_label_list,
            num_classes=config.model.architecture.num_classes,
        )

        # Compute average training loss for this epoch
        avg_train_loss = train_epoch_loss / (iteration + 1)
        res_train_loss.append(avg_train_loss)

        # Compute training metrics (macro-averaged)
        # Note: Macro average treats all classes equally, which is appropriate
        # when class distribution is balanced
        res_train_acc.append(accuracy_score(true_label_list, pred_label_list))
        res_train_recall.append(recall_score(true_label_list, pred_label_list, average="macro"))
        res_train_precision.append(
            precision_score(true_label_list, pred_label_list, average="macro")
        )
        res_train_f1.append(f1_score(true_label_list, pred_label_list, average="macro"))

        # Store final batch predictions for this epoch (rarely used)
        y_train_pred_list.append(int(cols_pred[-1]))
        t_train_list.append(int(rows_t[-1]))

        # Validation after each epoch
        val_loss, val_acc, val_precision, val_recall, val_f1, _val_confusion = validation(
            config, model, val_loader, loss_fn, epoch, num_epoch, mlflow_logger
        )
        res_val_loss.append(val_loss)
        res_val_acc.append(val_acc)
        res_val_precision.append(val_precision)
        res_val_recall.append(val_recall)
        res_val_f1.append(val_f1)

        # Track best validation accuracy
        val_acc_best = max(val_acc_best, val_acc)

        # Display epoch metrics to console
        logger.info(
            f"Epoch [{epoch}/{num_epoch}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {res_train_acc[-1]:.4f} | "
            f"Train F1: {res_train_f1[-1]:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Back to training mode
        model.train()

    # Build structured metrics dictionary
    metrics: TrainingMetrics = {
        "train": {
            "loss": res_train_loss,
        },
        "validation": {
            "loss": res_val_loss,
            "accuracy": res_val_acc,
            "f1_score": res_val_f1,
            "precision": res_val_precision,
            "recall": res_val_recall,
        },
        "confusion_matrix": {
            "train": train_confusion,
            "validation": _val_confusion,
        },
    }

    return model, metrics
