"""Helper functions for optimization tasks.

This module provides helper functions for optimization tasks,
including hyperparameter tuning and model selection.
"""

import logging
from typing import Any

import optuna
from models.model_factory import create_model  # noqa: F401
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import GridSampler, RandomSampler, TPESampler

log = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial, config: DictConfig) -> dict[str, Any]:
    """Suggest hyperparameters based on config with conditional parameters.

    Dynamically suggests parameters defined in config.optuna.optuna_params.
    Supports int, float, and categorical parameter types.

    For NAS (Neural Architecture Search), model_type is suggested first,
    then only relevant parameters for that model type are suggested:
    - GRU/LSTM: Common parameters only (hidden_dim, num_layers, dropout_ratio, etc.)
    - Transformer/Informer: Common + attention-specific parameters
      (num_heads, feedforward_multiplier)

    Args:
        trial: Optuna trial object.
        config: Hydra configuration containing optuna_params definition.

    Returns:
        Dictionary of suggested parameter names and values.

    """
    params: dict[str, Any] = {}

    if not hasattr(config.optuna, "optuna_params"):
        log.warning("No optuna_params defined in config. Using default model config.")
        return params

    # Step 1: Suggest model_type first if it exists (for NAS)
    model_type = None
    if "model_type" in config.optuna.optuna_params:
        param_config = config.optuna.optuna_params["model_type"]
        model_type = trial.suggest_categorical("model_type", param_config.choices)
        params["model_type"] = model_type
        log.info(f"NAS: Selected model_type = {model_type}")

    # Step 2: Suggest other parameters conditionally
    for param_name, param_config in config.optuna.optuna_params.items():
        # Skip model_type (already processed)
        if param_name == "model_type":
            continue

        # Conditional parameters for Transformer/Informer only
        transformer_only_params = {"num_heads", "feedforward_multiplier"}
        if param_name in transformer_only_params:
            if model_type and model_type not in ["Transformer", "Informer"]:
                log.debug(f"Skipping {param_name} for {model_type} (Transformer/Informer only)")
                continue

        param_type = param_config.get("type")

        if param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                param_config.low,
                param_config.high,
                step=param_config.get("step", 1),
            )
        elif param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                param_config.low,
                param_config.high,
                step=param_config.get("step"),
                log=param_config.get("log", False),
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_config.choices)
        else:
            log.warning(f"Unknown parameter type '{param_type}' for {param_name}")

    return params


def update_config(config: DictConfig, params: dict[str, Any]) -> DictConfig:
    """Update config with suggested parameters.

    Maps suggested parameters to appropriate config locations.
    Handles both model architecture params and training hyperparameters.

    Args:
        config: Original Hydra configuration.
        params: Dictionary of suggested parameters.

    Returns:
        Updated configuration with suggested parameters.

    """
    # Create a mutable copy
    # resolve=False: 変数参照をそのまま保持（InterpolationKeyErrorを回避）
    config_dict = OmegaConf.to_container(config, resolve=False)
    config_mutable = OmegaConf.create(config_dict)
    assert isinstance(config_mutable, DictConfig), "Config must be a DictConfig"
    config = config_mutable

    # Map parameters to config locations
    param_mapping = {
        # Architecture parameters (NAS)
        "hidden_dim": "model.architecture.hidden_dim",
        "num_layers": "model.architecture.num_layers",
        "dropout_ratio": "model.architecture.dropout_ratio",
        "num_heads": "model.architecture.num_heads",  # Transformer/Informer only
        # Transformer/Informer only
        "feedforward_multiplier": "model.architecture.feedforward_multiplier",
        "seq_len": "model.architecture.seq_len",  # Input sequence length
        # Training hyperparameters (HPO)
        "learning_rate": "training.optimizer.learning_rate",
        "batch_size": "training.batch_size",
        "epochs": "training.epochs",
        "weight_decay": "training.optimizer.weight_decay",
        "max_norm": "training.max_norm",
        # Model selection
        "model_type": "model.name",
    }

    for param_name, param_value in params.items():
        if param_name in param_mapping:
            config_path = param_mapping[param_name]
            OmegaConf.update(config, config_path, param_value, merge=False)
            log.info(f"Updated {config_path} = {param_value}")
        else:
            log.warning(f"No mapping found for parameter: {param_name}")

    return config


def create_sampler(config: DictConfig) -> optuna.samplers.BaseSampler:
    """Create Optuna sampler based on config.

    Args:
        config: Hydra configuration with sampler settings.

    Returns:
        Configured Optuna sampler.

    """
    sampler_name = config.optuna.sampler.name.lower()

    if sampler_name == "tpe":
        return TPESampler(
            n_startup_trials=config.optuna.sampler.get("n_startup_trials", 10),
            n_ei_candidates=config.optuna.sampler.get("n_ei_candidates", 24),
            multivariate=config.optuna.sampler.get("multivariate", False),
            seed=config.seed,
        )
    elif sampler_name == "random":
        return RandomSampler(seed=config.seed)
    elif sampler_name == "grid":
        return GridSampler(search_space={})  # Search space defined by suggest methods
    else:
        log.warning(f"Unknown sampler: {sampler_name}. Using TPE.")
        return TPESampler(seed=config.seed)


def create_pruner(config: DictConfig) -> optuna.pruners.BasePruner | None:
    """Create Optuna pruner based on config.

    Args:
        config: Hydra configuration with pruner settings.

    Returns:
        Configured Optuna pruner or None.

    """
    pruner_name = config.optuna.pruner.name.lower()

    if pruner_name == "median":
        return MedianPruner(
            n_startup_trials=config.optuna.pruner.get("n_startup_trials", 5),
            n_warmup_steps=config.optuna.pruner.get("n_warmup_steps", 3),
            interval_steps=config.optuna.pruner.get("interval_steps", 1),
        )
    elif pruner_name == "hyperband":
        return HyperbandPruner(
            min_resource=config.optuna.pruner.get("min_resource", 1),
            max_resource=config.optuna.pruner.get("max_resource", "auto"),
            reduction_factor=config.optuna.pruner.get("reduction_factor", 3),
        )
    elif pruner_name == "none":
        return None
    else:
        log.warning(f"Unknown pruner: {pruner_name}. Using MedianPruner.")
        return MedianPruner()
