"""Optuna optimization entry point for hyperparameter and architecture search.

This module provides a unified interface for both HPO (Hyperparameter Optimization)
and NAS (Neural Architecture Search) using Optuna. The optimization parameters
are configured via Hydra config files (e.g., conf/optuna/nas.yaml or hpo.yaml).

Usage:
    python apexai/optimization_entrypoint.py optuna=nas
    python apexai/optimization_entrypoint.py optuna=hpo
"""

import logging
from typing import Any

import hydra
import mlflow
import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from apexai.engine.train import train_model
from apexai.util.load_dataset import CustomTensorDataset, get_all_file_list
from apexai.util.optimization_helpers import (
    create_model,
    create_pruner,
    create_sampler,
    suggest_params,
    update_config,
)
from apexai.util.preprocessing import generate_batch
from apexai.util.tools import (
    log_confusion_matrix_to_mlflow,
    seed_everything,
    validate_data_shape,
)

log = logging.getLogger(__name__)


def save_top_k_models(
    study: optuna.Study,
    config: DictConfig,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Save Top-K models to MLflow after optimization completes.

    Args:
        study: Completed Optuna study.
        config: Hydra configuration.
        checkpoint_dir: Directory containing trial checkpoints.

    """
    import os
    import shutil

    top_k = config.optuna.model_saving.get("top_k_models", 5)
    if top_k <= 0:
        log.info("Top-K model saving disabled (top_k_models <= 0)")
        return

    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        log.warning("No completed trials to save")
        return

    # Sort by objective value
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        completed_trials.sort(
            key=lambda t: t.value if t.value is not None else float("-inf"),
            reverse=True,
        )
    else:
        completed_trials.sort(key=lambda t: t.value if t.value is not None else float("inf"))

    # Select Top-K trials
    top_k_trials = completed_trials[:top_k]

    log.info(f"Saving Top-{top_k} models to MLflow...")

    for rank, trial in enumerate(top_k_trials, start=1):
        checkpoint_path = f"{checkpoint_dir}/trial_{trial.number}.pth"

        if not os.path.exists(checkpoint_path):
            log.warning(f"Checkpoint not found for trial {trial.number}: {checkpoint_path}")
            continue

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Start a child run for this model
            with mlflow.start_run(run_name=f"Top{rank}_Trial{trial.number}", nested=True):
                # Log trial metadata
                mlflow.log_params(
                    {
                        "trial_number": trial.number,
                        "rank": rank,
                        "objective_value": trial.value,
                        **trial.params,
                    }
                )

                # Log metrics
                if "metrics" in checkpoint:
                    for metric_name, values in checkpoint["metrics"].get("validation", {}).items():
                        if values:
                            mlflow.log_metric(f"final_val_{metric_name}", values[-1])

                # Save model state_dict as artifact
                model_artifact_path = f"models/trial_{trial.number}_rank_{rank}.pth"
                mlflow.log_dict(
                    {
                        "model_state_dict": checkpoint["model_state_dict"],
                        "config": checkpoint.get("config", {}),
                    },
                    artifact_file=model_artifact_path,
                )

                log.info(
                    f"✓ Saved Top-{rank} model: Trial {trial.number} (Objective: {trial.value:.4f})"
                )

        except Exception as e:
            log.error(f"Failed to save model for trial {trial.number}: {e}")

    # Cleanup: Remove all checkpoints
    log.info(f"Cleaning up checkpoints in {checkpoint_dir}...")
    try:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            log.info(f"✓ Checkpoint directory removed: {checkpoint_dir}")
    except Exception as e:
        log.warning(f"Failed to cleanup checkpoints: {e}")


def objective(
    trial: optuna.Trial,
    config: DictConfig,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    device: torch.device,
    actual_feature_size: int,
    out_dim: int,
) -> float:
    """Objective function for Optuna optimization.

    This function is called for each trial. It suggests parameters,
    trains the model, and returns the optimization metric.

    Args:
        trial: Optuna trial object.
        config: Base Hydra configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        device: PyTorch device (CPU/GPU).
        actual_feature_size: Input feature dimension.
        out_dim: Output dimension (number of classes).

    Returns:
        Optimization metric value (higher is better for maximize direction).

    """
    # CRITICAL: Reset seed for reproducibility across trials
    # This ensures that Trainer and Optuna use the same initial model weights
    g_trial, seed_worker_trial = seed_everything(config.seed)

    # Suggest parameters based on config
    params = suggest_params(trial, config)
    log.info(f"Trial {trial.number}: Testing parameters {params}")

    # Update config with suggested parameters
    trial_config = update_config(config, params)

    # Create DataLoaders with fresh generator for this trial
    # This ensures shuffle order is deterministic and reproducible
    train_loader_trial = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=trial_config.training.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker_trial,
        generator=g_trial,
    )
    val_loader_trial = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=trial_config.training.batch_size,
        shuffle=False,
    )

    # Start nested MLflow run for this trial
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Log trial parameters
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        try:
            # Create model with trial configuration
            model = create_model(trial_config, actual_feature_size, out_dim)

            # DEBUG: Log initial model parameters for reproducibility verification
            first_param = next(model.parameters())
            first_100_vals = first_param.flatten()[:100].detach().cpu().numpy()
            log.info("[DEBUG] Model initial params - First 100 values:")
            log.info(f"[DEBUG]   {first_100_vals}")
            param_sum = sum(p.sum().item() for p in model.parameters())
            log.info(f"[DEBUG] Model initial params - Sum: {param_sum:.10f}")

            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=trial_config.training.optimizer.learning_rate,
                weight_decay=trial_config.training.optimizer.get("weight_decay", 0.0),
            )
            loss_fn = nn.CrossEntropyLoss()
            num_epoch = trial_config.training.epochs

            # Create scheduler only if enabled in config
            scheduler = None
            if trial_config.training.scheduler.enabled:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epoch, eta_min=0
                )

            # Train model (use trial-specific loaders for reproducibility)
            trained_model, metrics = train_model(
                trial_config,
                model,
                optimizer,
                loss_fn,
                num_epoch,
                scheduler,
                train_loader_trial,
                val_loader_trial,
                mlflow_logger=None,
            )

            # Log metrics to MLflow
            for metric_name, values in metrics["train"].items():
                for epoch, value in enumerate(values, start=1):
                    mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
                    # Report intermediate value for pruning (use training loss)
                    if metric_name == "loss":
                        trial.report(value, epoch)

            for metric_name, values in metrics["validation"].items():
                for epoch, value in enumerate(values, start=1):
                    mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

            # Log confusion matrices
            log_confusion_matrix_to_mlflow(
                cm=metrics["confusion_matrix"]["validation"],
                prefix="final_validation_",
                normalize=False,
                config=trial_config,
            )

            # Determine optimization metric based on study direction
            metric_name = trial_config.optuna.get("optimization_metric", "f1_score")
            if metric_name in metrics["validation"]:
                objective_value = float(metrics["validation"][metric_name][-1])
            else:
                log.warning(f"Metric {metric_name} not found. Using validation loss instead.")
                # Negative for maximization (loss should be minimized)
                objective_value = float(-metrics["validation"]["loss"][-1])

            mlflow.log_metric("objective_value", objective_value)

            # Save checkpoint for potential Top-K selection
            checkpoint_dir = trial_config.get("checkpoint_dir", "checkpoints")
            checkpoint_path = f"{checkpoint_dir}/trial_{trial.number}.pth"

            try:
                import os

                os.makedirs(checkpoint_dir, exist_ok=True)

                # Convert config to container (resolve=False to avoid interpolation errors)
                try:
                    config_dict = OmegaConf.to_container(trial_config, resolve=False)
                except Exception:
                    # Fallback: save only trial params if config conversion fails
                    config_dict = dict(trial.params)

                checkpoint = {
                    "trial_number": trial.number,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config_dict,
                    "metrics": metrics,
                    "objective_value": objective_value,
                }
                torch.save(checkpoint, checkpoint_path)
                log.info(f"Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                log.warning(f"Failed to save checkpoint: {e}")

            log.info(f"Trial {trial.number} completed. Objective value: {objective_value:.4f}")

            # Check if trial should be pruned
            if trial.should_prune():
                log.info(f"Trial {trial.number} pruned.")
                raise optuna.TrialPruned()

            return objective_value

        except Exception as e:
            log.error(f"Trial {trial.number} failed with error: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise


@hydra.main(config_path="../conf", config_name="config")
def main(config: DictConfig) -> dict[str, Any]:
    """Main entry point for Optuna optimization.

    Loads data, creates Optuna study, and runs optimization.
    Results are logged to both Optuna database and MLflow.

    Args:
        config: Hydra configuration object.

    Returns:
        Dictionary with optimization results (best_params, best_value, etc.).

    """
    # NOTE: Removed seed_everything() here - no random operations until DataLoader creation
    # Seed is set in objective() right before DataLoader creation (Line 179)
    log.info(">>>>> Starting Optuna Optimization <<<<<")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Setup MLflow tracking
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    # Load datasets
    log.info("Loading datasets...")
    x_train_dataset_list = get_all_file_list(config.data_paths.train_dir + "/x_train", "*.csv")
    y_train_dataset_list = get_all_file_list(config.data_paths.train_dir + "/y_train", "*.csv")
    x_valid_dataset_list = get_all_file_list(config.data_paths.valid_dir + "/x_valid", "*.csv")
    y_valid_dataset_list = get_all_file_list(config.data_paths.valid_dir + "/y_valid", "*.csv")

    log.info(f"Training samples: {len(x_train_dataset_list)}")
    log.info(f"Validation samples: {len(x_valid_dataset_list)}")

    # IMPORTANT: For Z-score normalization, calculate statistics from training data first!
    if (
        config.data.preprocessing.normalization.enabled
        and config.data.preprocessing.normalization.method == "zscore"
        and config.data.preprocessing.normalization.zscore.mean is None
    ):
        from util.preprocessing import calculate_global_statistics

        log.info("Calculating global statistics from TRAINING data for Z-score normalization...")
        train_mean, train_std = calculate_global_statistics(
            x_train_dataset_list, config.data.preprocessing.features, config
        )
        log.info("Training data statistics calculated:")
        log.info(f"  Mean: {train_mean}")
        log.info(f"  Std:  {train_std}")

        # Update config with training statistics for validation/test
        config.data.preprocessing.normalization.zscore.mean = train_mean
        config.data.preprocessing.normalization.zscore.std = train_std

    # Generate batches
    x_train_batch, y_train_batch = generate_batch(
        x_train_dataset_list, y_train_dataset_list, config
    )
    log.info("  (Using training data statistics for validation normalization)")
    x_val_batch, y_val_batch = generate_batch(x_valid_dataset_list, y_valid_dataset_list, config)

    # Create datasets
    train_dataset = CustomTensorDataset(x_train_batch, y_train_batch)
    val_dataset = CustomTensorDataset(x_val_batch, y_val_batch)

    # Validate data shapes
    actual_seq_len = x_train_batch.shape[1]
    actual_feature_size = x_train_batch.shape[2]
    out_dim = config.model.architecture.num_classes

    validate_data_shape(
        actual_seq_len,
        actual_feature_size,
        config.model.architecture.get("seq_len", None),
        config.model.architecture.get("feature_size", None),
        log,
    )

    # Note: DataLoaders will be created inside objective() for each trial
    # to ensure reproducibility with fresh generator state

    # Create Optuna study
    log.info(f"Creating Optuna study: {config.optuna.study.name}")
    study = optuna.create_study(
        study_name=config.optuna.study.name,
        direction=config.optuna.study.direction,
        storage=config.optuna.study.storage_url,
        load_if_exists=config.optuna.study.load_if_exists,
        sampler=create_sampler(config),
        pruner=create_pruner(config),
    )

    # Start parent MLflow run
    with mlflow.start_run(run_name=f"Optuna_{config.optuna.study.name}"):
        # Log study configuration
        mlflow.log_params(
            {
                "study_name": config.optuna.study.name,
                "direction": config.optuna.study.direction,
                "n_trials": config.optuna.optimization.n_trials,
                "timeout": config.optuna.optimization.timeout,
                "sampler": config.optuna.sampler.name,
                "pruner": config.optuna.pruner.name,
                "base_model": config.model.name,
            }
        )

        # Run optimization
        log.info("Starting optimization...")
        study.optimize(
            lambda trial: objective(
                trial,
                config,
                train_dataset,
                val_dataset,
                device,
                actual_feature_size,
                out_dim,
            ),
            n_trials=config.optuna.optimization.n_trials,
            timeout=config.optuna.optimization.timeout,
            n_jobs=config.optuna.optimization.get("n_jobs", 1),
            show_progress_bar=True,
        )

        # Log best results
        log.info(">>>>> Optimization Complete <<<<<")
        log.info(f"Best trial: {study.best_trial.number}")
        log.info(f"Best value: {study.best_value:.4f}")
        log.info(f"Best parameters: {study.best_params}")

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_value", study.best_value)
        mlflow.log_metric("n_trials_completed", len(study.trials))

        # Save Top-K models to MLflow
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        save_top_k_models(study, config, checkpoint_dir)

        # Return results
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
        }


if __name__ == "__main__":
    main()
