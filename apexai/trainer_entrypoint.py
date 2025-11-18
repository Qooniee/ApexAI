"""Entry point for training the model."""

import logging

import hydra
import mlflow
import torch
import torch.nn as nn
from engine.train import train_model
from models.model_factory import create_model
from util.load_dataset import CustomTensorDataset, get_all_file_list
from util.preprocessing import generate_batch
from util.tools import (
    log_confusion_matrix_to_mlflow,
    seed_everything,
    validate_data_shape,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def main(config):
    """Main training entry point.

    Loads datasets, creates data loaders, initializes model and optimizer,
    and trains the model while logging all metrics and artifacts to MLflow.

    Args:
        config: Hydra configuration object containing all hyperparameters.

    """
    g, seed_worker = seed_everything(config.seed)
    log.info(">>>>> Starting Main Process <<<<<")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Setup MLflow tracking
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name=config.mlflow.get("run_name", None)):
        # Log important hyperparameters only (avoid Hydra internals)
        mlflow.log_params(
            {
                "seed": config.seed,
                "model_type": config.model.name,
                "hidden_size": config.model.architecture.hidden_dim,
                "num_layers": config.model.architecture.num_layers,
                "num_classes": config.model.architecture.num_classes,
                "batch_size": config.training.batch_size,
                "num_epochs": config.training.epochs,
                "learning_rate": config.training.optimizer.learning_rate,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            }
        )

        log.info("Creating dataset file list...")
        x_train_dataset_list = get_all_file_list(config.data_paths.train_dir + "/x_train", "*.csv")
        y_train_dataset_list = get_all_file_list(config.data_paths.train_dir + "/y_train", "*.csv")

        x_valid_dataset_list = get_all_file_list(config.data_paths.valid_dir + "/x_valid", "*.csv")
        y_valid_dataset_list = get_all_file_list(config.data_paths.valid_dir + "/y_valid", "*.csv")

        x_test_dataset_list = get_all_file_list(config.data_paths.test_dir + "/x_test", "*.csv")
        y_test_dataset_list = get_all_file_list(config.data_paths.test_dir + "/y_test", "*.csv")
        log.info("--- Dataset Information ---")
        log.info(f"Number of x_train dataset: {len(x_train_dataset_list)}")
        log.info(f"Number of y_train dataset: {len(y_train_dataset_list)}")
        log.info(f"Number of x_valid dataset: {len(x_valid_dataset_list)}")
        log.info(f"Number of y_valid dataset: {len(y_valid_dataset_list)}")
        log.info(f"Number of x_test dataset: {len(x_test_dataset_list)}")
        log.info(f"Number of y_test dataset: {len(y_test_dataset_list)}")
        log.info("---------------------------")

        # load datasets
        # IMPORTANT: For Z-score normalization, calculate statistics from training data first!
        if (
            config.data.preprocessing.normalization.enabled
            and config.data.preprocessing.normalization.method == "zscore"
            and config.data.preprocessing.normalization.zscore.mean is None
        ):
            from util.preprocessing import calculate_global_statistics

            log.info(
                "Calculating global statistics from TRAINING data for Z-score normalization..."
            )
            train_mean, train_std = calculate_global_statistics(
                x_train_dataset_list, config.data.preprocessing.features, config
            )
            log.info("Training data statistics calculated:")
            log.info(f"  Mean: {train_mean}")
            log.info(f"  Std:  {train_std}")

            # Update config with training statistics for validation/test
            config.data.preprocessing.normalization.zscore.mean = train_mean
            config.data.preprocessing.normalization.zscore.std = train_std

            # Log to MLflow for reproducibility
            mlflow.log_params(
                {
                    "normalization_method": "zscore",
                    "train_mean": str(train_mean),
                    "train_std": str(train_std),
                }
            )

        log.info("Creating batch for training datasets")
        x_train_batch, y_train_batch = generate_batch(
            x_train_dataset_list, y_train_dataset_list, config
        )
        log.info("Creating train loader...")
        train_dataset = CustomTensorDataset(x_train_batch, y_train_batch)

        log.info("Creating batch for validation set...")
        log.info("  (Using training data statistics for normalization)")
        x_val_batch, y_val_batch = generate_batch(
            x_valid_dataset_list, y_valid_dataset_list, config
        )

        log.info("Creating validation loader...")
        val_dataset = CustomTensorDataset(x_val_batch, y_val_batch)

        actual_seq_len = x_train_batch.shape[1]
        actual_feature_size = x_train_batch.shape[2]
        expected_seq_len = config.model.architecture.get("seq_len", None)
        expected_feature_size = config.model.architecture.get("feature_size", None)

        validate_data_shape(
            actual_seq_len,
            actual_feature_size,
            expected_seq_len,
            expected_feature_size,
            log,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )
        # Create model using factory
        out_dim = config.model.architecture.num_classes
        model = create_model(config, actual_feature_size, out_dim)
        log.info("Model structure:\n%s", model)
        # DEBUG: Log initial model parameters for reproducibility verification
        first_param = next(model.parameters())
        first_100_vals = first_param.flatten()[:100].detach().cpu().numpy()
        log.info("[DEBUG] Model initial params - First 100 values:")
        log.info(f"[DEBUG]   {first_100_vals}")
        param_sum = sum(p.sum().item() for p in model.parameters())
        log.info(f"[DEBUG] Model initial params - Sum: {param_sum:.10f}")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.learning_rate,
            weight_decay=config.training.optimizer.weight_decay,
        )
        num_epoch = config.training.epochs

        # Create scheduler only if enabled in config
        scheduler = None
        if config.training.scheduler.enabled:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epoch, eta_min=0
            )

        # Train model (mlflow_logger=None means use mlflow.active_run() directly)
        trained_model, metrics = train_model(
            config,
            model,
            optimizer,
            loss_fn,
            num_epoch,
            scheduler,
            train_loader,
            val_loader,
            mlflow_logger=None,
        )

        # Log all epoch-wise metrics to MLflow in batch
        log.info("Logging metrics to MLflow...")
        # Log training metrics
        for metric_name, values in metrics["train"].items():
            for epoch, value in enumerate(values, start=1):
                mlflow.log_metric(f"train_{metric_name}", value, step=epoch)

        # Log validation metrics
        for metric_name, values in metrics["validation"].items():
            for epoch, value in enumerate(values, start=1):
                mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

        # Log final confusion matrices
        log.info("Logging confusion matrices to MLflow...")
        log_confusion_matrix_to_mlflow(
            cm=metrics["confusion_matrix"]["train"],
            prefix="final_train_",
            normalize=False,
            config=config,
        )
        log_confusion_matrix_to_mlflow(
            cm=metrics["confusion_matrix"]["train"],
            prefix="final_train_",
            normalize=True,
            config=config,
        )
        log_confusion_matrix_to_mlflow(
            cm=metrics["confusion_matrix"]["validation"],
            prefix="final_validation_",
            normalize=False,
            config=config,
        )
        log_confusion_matrix_to_mlflow(
            cm=metrics["confusion_matrix"]["validation"],
            prefix="final_validation_",
            normalize=True,
            config=config,
        )

        # Log model artifact with signature and input example
        log.info("Logging model to MLflow...")
        # Create a sample input for model signature inference
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch[0][:1]  # Take one sample from batch

        # Set model to eval mode and create signature
        trained_model.eval()
        with torch.no_grad():
            sample_output = trained_model(sample_input.to(device))

        signature = mlflow.models.infer_signature(
            sample_input.cpu().numpy(),
            sample_output.detach().cpu().numpy(),
        )

        mlflow.pytorch.log_model(
            trained_model,
            "model",
            signature=signature,
            input_example=sample_input.cpu().numpy(),
        )

        log.info(">>>>> Training Complete <<<<<")
        log.info(f"Model and metrics logged to MLflow run: {mlflow.active_run().info.run_id}")

        # Return metrics for potential reuse (e.g., in Optuna optimization)
        return metrics


if __name__ == "__main__":
    main()
