"""Visualization module for training, testing, and prediction results.

This module provides functions to generate and save confusion matrix
heatmaps and performance curve visualizations.
"""

import datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from omegaconf import DictConfig


def traingraph(
    x_axis_epoch: list,
    res_train_loss: list,
    res_train_score: list,
    res_val_loss: list,
    res_val_score: list,
    train_confusion: np.ndarray,
    valid_confusion: np.ndarray,
    settings: "DictConfig",
) -> None:
    """Generate and save training result visualizations.

    Creates a 2x2 subplot figure with:
    - Loss curve (training vs validation)
    - F1-score curve (training vs validation)
    - Training confusion matrix heatmap
    - Validation confusion matrix heatmap

    Args:
        x_axis_epoch: List of epoch numbers for x-axis.
        res_train_loss: Training loss values per epoch.
        res_train_score: Training F1 scores per epoch.
        res_val_loss: Validation loss values per epoch.
        res_val_score: Validation F1 scores per epoch.
        train_confusion: Training confusion matrix.
        valid_confusion: Validation confusion matrix.
        settings: Configuration object with output directory and label names.

    """
    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("gray")
    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.plot(x_axis_epoch, res_train_loss, "r", label="Train", marker="o")
    ax1.plot(x_axis_epoch, res_val_loss, "b", label="Valid", marker="+")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("nEpoch")
    ax1.set_ylabel("Loss")
    plt.legend(loc="upper left")

    ax2 = fig.add_subplot(222)
    ax2.plot(x_axis_epoch, res_train_score, "r", label="Train", marker="o")
    ax2.plot(x_axis_epoch, res_val_score, "b", label="Valid", marker="+")
    ax2.set_title("f1-score Curve")
    ax2.set_xlabel("nEpoch")
    ax2.set_ylabel("f1-score")
    plt.legend(loc="upper left")

    ax3 = fig.add_subplot(223)
    train_confusion_df = pd.DataFrame(
        train_confusion,
        index=settings.config["LabelNameList"]["TrueLabel"],
        columns=settings.config["LabelNameList"]["PredLabel"],
    )
    sns.heatmap(
        train_confusion_df,
        linewidth=0.3,
        ax=ax3,
        annot=True,
        square=False,
        fmt=".1f",
    )

    ax3.set_title("Training Confusion Matrix")
    ax3.set_ylabel("True Label")
    ax3.set_xlabel("Predicted Label")

    valid_confusion_df = pd.DataFrame(
        valid_confusion,
        index=settings.config["LabelNameList"]["TrueLabel"],
        columns=settings.config["LabelNameList"]["PredLabel"],
    )
    ax4 = fig.add_subplot(224)
    sns.heatmap(
        valid_confusion_df,
        linewidth=0.3,
        ax=ax4,
        annot=True,
        square=False,
        fmt=".1f",
    )

    ax4.set_title("Validation Confusion Matrix")
    ax4.set_ylabel("True Label")
    ax4.set_xlabel("Predicted Label")
    fig.tight_layout()

    timestamp = (
        str(datetime.datetime.now())
        .replace("-", "")
        .replace(" ", "_")
        .replace(":", "")
        .replace(".", "_")
    )
    output_path = f"{settings.config['System']['OutputFileDir']}/{timestamp}_TrainingResult.png"
    fig.savefig(output_path)
    plt.close(fig)


def testgraph(
    loss: float,
    score: float,
    test_confusion: np.ndarray,
    settings: "DictConfig",
) -> None:
    """Generate and save test result confusion matrix visualization.

    Args:
        loss: Test loss value.
        score: Test F1 score.
        test_confusion: Test confusion matrix.
        settings: Configuration object with output directory and label names.

    """
    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("gray")
    fig = plt.figure()
    ax1 = fig.add_subplot()
    test_confusion_df = pd.DataFrame(
        test_confusion,
        index=settings.config["LabelNameList"]["TrueLabel"],
        columns=settings.config["LabelNameList"]["PredLabel"],
    )
    sns.heatmap(
        test_confusion_df,
        linewidth=0.3,
        ax=ax1,
        annot=True,
        square=False,
        fmt=".1f",
    )

    ax1.set_title(f"Test Confusion Matrix[Score:{score:.3f}, loss:{loss:.3f}]")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    timestamp = (
        str(datetime.datetime.now())
        .replace("-", "")
        .replace(" ", "_")
        .replace(":", "")
        .replace(".", "_")
    )
    output_path = f"{settings.config['System']['OutputFileDir']}/{timestamp}_TestResult.png"
    fig.savefig(output_path)
    plt.close(fig)


def predgraph(
    score: float,
    pred_confusion: np.ndarray,
    settings: "DictConfig",
) -> None:
    """Generate and save prediction result confusion matrix visualization.

    Args:
        score: Prediction F1 score.
        pred_confusion: Prediction confusion matrix.
        settings: Configuration object with output directory and label names.

    """
    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("gray")
    fig = plt.figure()
    ax1 = fig.add_subplot()
    pred_confusion_df = pd.DataFrame(
        pred_confusion,
        index=settings.config["LabelNameList"]["TrueLabel"],
        columns=settings.config["LabelNameList"]["PredLabel"],
    )
    sns.heatmap(
        pred_confusion_df,
        linewidth=0.3,
        ax=ax1,
        annot=True,
        square=False,
        fmt=".1f",
    )

    ax1.set_title(f"Pred Confusion Matrix[Score:{score:.3f}]")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    timestamp = (
        str(datetime.datetime.now())
        .replace("-", "")
        .replace(" ", "_")
        .replace(":", "")
        .replace(".", "_")
    )
    output_path = f"{settings.config['System']['OutputFileDir']}/{timestamp}_PredResult.png"
    fig.savefig(output_path)
    plt.close(fig)
