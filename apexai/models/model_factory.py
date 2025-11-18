"""Model factory for creating different model architectures.

This module provides a factory function for creating models based on configuration.
It supports GRU, LSTM, Transformer, and Informer models.
"""

import logging
from typing import cast

import torch.nn as nn
from omegaconf import DictConfig

from models import gru, informer, lstm, transformer

log = logging.getLogger(__name__)


def create_model(config: DictConfig, feature_size: int, out_dim: int) -> nn.Module:
    """Create model based on config.

    Supports dynamic model selection for NAS.

    Args:
        config: Hydra configuration.
        feature_size: Input feature size.
        out_dim: Output dimension (number of classes).

    Returns:
        Initialized PyTorch model.

    Raises:
        ValueError: If model name is not supported.

    Examples:
        >>> model = create_model(config, feature_size=9, out_dim=21)

    """
    model_name = config.model.name.lower()
    arch_config = config.model.architecture

    if model_name == "gru":
        log.info("Creating GRU model")
        return cast(
            nn.Module,
            gru.GRUwithFC(
                feature_size=feature_size,
                hidden_dim=arch_config.hidden_dim,
                num_layers=arch_config.num_layers,
                out_dim=out_dim,
                dropout_ratio=arch_config.dropout_ratio,
                classification=True,
                batch_first=True,
            ),
        )
    elif model_name == "lstm":
        log.info("Creating LSTM model")
        return cast(
            nn.Module,
            lstm.LSTMwithFC(
                feature_size=feature_size,
                hidden_dim=arch_config.hidden_dim,
                num_layers=arch_config.num_layers,
                out_dim=out_dim,
                dropout_ratio=arch_config.dropout_ratio,
                classification=True,
                batch_first=True,
            ),
        )
    elif model_name == "transformer":
        log.info("Creating Transformer model")
        return cast(
            nn.Module,
            transformer.TransformerModel(
                feature_size=feature_size,
                hidden_size=arch_config.hidden_dim,
                num_layers=arch_config.num_layers,
                num_heads=arch_config.get("num_heads", 8),
                num_classes=out_dim,
                feedforward_multiplier=arch_config.get("feedforward_multiplier", 4),
                dropout_ratio=arch_config.dropout_ratio,
                classification=True,
            ),
        )
    elif model_name == "informer":
        log.info("Creating Informer model")
        return cast(
            nn.Module,
            informer.Informer(
                feature_size=feature_size,
                hidden_dim=arch_config.hidden_dim,
                num_layers=arch_config.num_layers,
                num_heads=arch_config.get("num_heads", 4),
                feedforward_multiplier=arch_config.get("feedforward_multiplier", 4),
                out_dim=out_dim,
                dropout_ratio=arch_config.dropout_ratio,
                classification=True,
            ),
        )
    else:
        msg = f"Unknown model type: {model_name}"
        raise ValueError(msg)
