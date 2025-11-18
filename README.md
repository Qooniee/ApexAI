# ApexAI - Vehicle Identification Platform

[![CI](https://github.com/Qooniee/apexai/actions/workflows/ci.yml/badge.svg)](https://github.com/Qooniee/apexai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Qooniee/apexai/branch/main/graph/badge.svg)](https://codecov.io/gh/Qooniee/apexai)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AI-powered vehicle identification and race analysis using Toyota GR Cup telemetry data.

## 1. Overview

  ApexAI is a deep learning platform for identifying vehicles from race telemetry data.
  This project is developed for the Hack the Track competition, focusing on vehicle
  classification from time-series sensor data collected during races.

## 2. Features

  - **Vehicle Identification**: Classify vehicles from telemetry data (speed, throttle, brake, steering, acceleration)
  - **Time Series Models**: Support for GRU, LSTM, Transformer, and Informer architectures
  - **MLOps Integration**: Experiment tracking with MLflow, hyperparameter optimization with Optuna
  - **Docker-based**: Fully containerized environment with GPU support



## 3. Installation
### 3-1. Prerequisites

  - Docker and Docker Compose
  - NVIDIA GPU with CUDA support
  - At least 16GB RAM
  - 10GB free disk space

### 3-2. Setup

1. Clone the repository
  ```bash
  git clone https://github.com/Qooniee/ApexAI.git
  ```
  2. Copy .env.templete and rename it to .env
    If you want to change the network settings, change the env file. If not, just copy is OK.
  3. Run the setup script:
  ```bash
  python setup_apexai.py
  ```

  This will:
  - Build Docker images
  - Initialize PostgreSQL databases (MLflow and Optuna)
  - Set up MinIO for artifact storage
  - Verify all services

  ### 3-3. Access Services

  After setup completes, access the following services:

  - **MLflow UI**: http://localhost:5001
  - **Optuna Dashboard**: http://localhost:8081
  - **MinIO Console**: http://localhost:9020
  - **pgAdmin**: http://localhost:5051
  - **Jupyter Lab**: http://localhost:8889
  - **Streamlit**: http://localhost:8501

  Username and password are written in the .env file

## 4. Development Environment Setup

### 4-1. Setup for development environment
  ApexAI uses uv to manage python environment.
  I recommend to reffer pyproject.toml, ruff.toml and .pre-commit-config.yaml.
  Also ApexAI has test and code analysis functions using pytests, ruff, mypy.
  Those precedures automatically are executed when you commit and push.
  To enable tools for development, you have to install ruff, pre-commit, pytest, pytest-cov, and mypy

  ```bash
  uv pip install ruff pre-commit pytest pytest-cov mypy
  ```
  or
  ```bash
  uv pip install -e .[dev]
  ```
### 4-2. Set up pre-commit hooks
  ```bash
  pre-commit install
  ```

### 4-3. Verify installation
  ```bash
  ruff --version
  pre-commit --version
  pytest --version
  mypy --version
  ```

### 4-4. Important Notes

  **Virtual Environment Location:**
  - Docker container uses `/opt/venv` (configured in Dockerfile)
  - **Do NOT use `uv sync --extra dev`** - it creates a local `.venv` that conflicts with `/opt/venv`
  - Use `uv pip install` to install packages directly to the active `/opt/venv` environment
  - If `.venv` is accidentally created, delete it and reinstall with `uv pip install`

**Development Tools:**
  - **Ruff**: Linter and formatter (`ruff check .` / `ruff format .`)
  - **Pre-commit**: Git hooks for code quality (`pre-commit run --all-files`)
  - **PyTest**: Testing framework (`pytest`)
  - **MyPy**: Static type checker (`mypy src/`)

**Why not `uv sync --extra dev`?**
  - In Docker container environments with mounted volumes, `uv sync` ignores `VIRTUAL_ENV=/opt/venv`
  - It creates a new `.venv` in the mounted directory, causing conflicts
  - Use direct `uv pip install` instead for container-based development

## 5. Project Structure

  ```
  apexai/
  ├── apexai/                        # Main Python package
  │   ├── data_generation/          # Automated data pipeline
  │   │   ├── pipeline.py           # End-to-end pipeline orchestration
  │   │   ├── preprocess_telemetry.py  # Resampling & lap extraction
  │   │   ├── generate_training_data.py # X/y file generation
  │   │   ├── split_dataset.py      # Train/valid/test splitting
  │   │   └── vir_loader.py         # VIR dataset loader
  │   ├── models/                   # Neural network architectures
  │   │   ├── lstm.py               # LSTM model
  │   │   ├── gru.py                # GRU model
  │   │   ├── transformer.py        # Transformer model
  │   │   ├── informer.py           # Informer model
  │   │   └── model_factory.py      # Dynamic model instantiation
  │   ├── engine/                   # Training & evaluation logic
  │   │   ├── train.py              # Training loop
  │   │   ├── test.py               # Evaluation logic
  │   │   ├── inference.py          # Inference engine
  │   │   ├── metrics.py            # Performance metrics
  │   │   └── make_graph.py         # Visualization utilities
  │   ├── simulator/                # Real-time inference simulator
  │   │   ├── frontend/             # Streamlit UI
  │   │   │   └── app.py            # Web interface
  │   │   ├── backend/              # Inference engine
  │   │   │   └── engine.py         # Queue-based streaming inference
  │   │   └── model_repository/     # ONNX model storage
  │   ├── util/                     # Utility modules
  │   │   ├── load_dataset.py       # Dataset loaders
  │   │   ├── preprocessing.py      # Normalization & transforms
  │   │   ├── optimization_helpers.py  # BO-NAS parameter suggestion
  │   │   └── tools.py              # Common utilities
  │   ├── signal_processing/        # Signal processing utilities
  │   ├── datasets/                 # Legacy dataset utilities
  │   ├── model_resistory/          # Model repository utilities
  │   ├── trainer_entrypoint.py     # Single model training entry point
  │   ├── optimization_entrypoint.py # HPO/NAS optimization entry point
  │   └── EDA.ipynb                 # Exploratory data analysis notebook
  │
  ├── conf/                         # Hydra configuration files
  │   ├── config.yaml               # Main configuration
  │   ├── data/                     # Data configurations
  │   │   └── toyota_gr86.yaml      # GR86 dataset config
  │   ├── model/                    # Model architecture configs
  │   │   ├── gru.yaml              # GRU architecture
  │   │   ├── lstm.yaml             # LSTM architecture
  │   │   ├── transformer.yaml      # Transformer architecture
  │   │   └── informer.yaml         # Informer architecture
  │   ├── training/                 # Training configurations
  │   │   └── default.yaml          # Default training config
  │   └── optuna/                   # Optuna optimization configs
  │       ├── hpo.yaml              # Hyperparameter optimization
  │       └── nas.yaml              # Neural architecture search
  │
  ├── datasets/                     # Data storage
  │   ├── rawdata/                  # Raw telemetry CSV files
  │   │   └── VIR/                  # Virginia International Raceway
  │   ├── preprocessed_10Hz/        # Resampled lap data (10Hz)
  │   └── drivingdatasets/          # Training-ready datasets
  │       └── input/                # Train/valid/test splits
  │           ├── train/            # Training data
  │           ├── valid/            # Validation data
  │           └── test/             # Test data
  │
  ├── dockerfiles/                  # Docker configurations
  │   └── Dockerfile                # Main container definition
  │
  ├── init_scripts/                 # Infrastructure initialization
  │   ├── setup_database.py         # PostgreSQL setup
  │   ├── setup_mlflow_db.py        # MLflow database setup
  │   ├── setup_optuna.py           # Optuna study initialization
  │   ├── setup_minio.py            # MinIO artifact storage setup
  │   ├── environment_check.py      # Environment validation
  │   ├── gpu_check.py              # GPU availability check
  │   ├── health_check.py           # Service health monitoring
  │   ├── test_experiment.py        # MLflow experiment test
  │   ├── test_modedetection_integration.py  # Integration test
  │   ├── 01-create-databases.sh    # Database creation script
  │   └── wait_for_databases.sh     # Database readiness check
  │
  ├── tests/                        # Unit tests
  │   ├── test_vir_loader.py        # VIR loader tests
  │   ├── test_preprocessing.py     # Preprocessing tests
  │   ├── test_resample.py          # Resampling tests
  │   └── test_filter.py            # Filtering tests
  │
  ├── .github/                      # GitHub Actions workflows
  ├── .devcontainer/                # VS Code devcontainer config
  ├── hydra_experiment_logs/        # Hydra runtime outputs
  ├── postgres_data/                # PostgreSQL data volume
  ├── minio_data/                   # MinIO storage volume
  ├── pgadmin_data/                 # pgAdmin data volume
  │
  ├── docker-compose.yaml           # Docker Compose orchestration
  ├── pyproject.toml                # Python project metadata & dependencies
  ├── ruff.toml                     # Ruff linter configuration
  ├── .pre-commit-config.yaml       # Pre-commit hooks configuration
  ├── .env.template                 # Environment variable template
  ├── .gitignore                    # Git ignore rules
  ├── LICENSE                       # Apache 2.0 License
  ├── README.md                     # This documentation
  ├── setup_apexai.py               # One-click infrastructure setup
  ├── apexai_startup_check.py       # Startup health check script
  └── analyze_abnormal_laps.py      # Lap anomaly detection utility

  ```


## 6. Dataset

  The project uses telemetry data from the Toyota GR Cup races:

  - **Circuit**: Virginia International Raceway (VIR)
  - **Total Vehicles**: 21 unique vehicles
  - **Total Laps**: 836 laps
  - **Features**: 9-dimensional telemetry
      - pbrake_f
      - pbrake_r
      - Steering_Angle
      - accx_can
      - accy_can
      - ath
      - gear
      - nmot
      - speed
  - **Sampling Rate**: ~23Hz. Original sampling rate is converted to 10Hz thorugh preprocessing
  - **Task**: Multi-class classification of vehicle IDs from time-series telemetry

## 7. Data Pipeline

  ApexAI provides a fully automated data generation pipeline (apexai/data_generation/pipeline.py) that
  handles the complete workflow from raw telemetry to training-ready datasets.

### 7-1. Pipeline Overview

  The pipeline performs three main steps automatically:

  1. Resampling & Lap Extraction (preprocess_telemetry.py)
    - Downsample telemetry from ~23Hz to 10Hz with anti-aliasing filter
    - Extract individual lap data from race telemetry
    - Filter invalid laps (minimum length validation)
    - Apply linear interpolation for missing values
  2. Feature/Label Separation (generate_training_data.py)
    - Generate X (features) and y (labels) file pairs
    - Separate directories: x_train/ and y_train/
    - Preserve vehicle ID mapping in labels
  3. Train/Valid/Test Split (split_dataset.py)
    - Split dataset by vehicle ID (default: 70%/15%/15%)
    - Filter out last lap per vehicle (out laps)
    - Shuffle laps for better generalization
    - Create final directory structure: train/, valid/, test/
### 7-2. Quick Start

  Generate training datasets from raw telemetry with a single command:

  ```bash
  python -m apexai.data_generation.pipeline \
    --raw-data-dir datasets/rawdata/VIR \
    --export-dir datasets/drivingdatasets/input
  ```

  This will:
  - Process all race data in datasets/rawdata/VIR/
  - Apply 10Hz resampling with anti-aliasing filter
  - Generate X/y training file pairs
  - Split into train/valid/test sets by vehicle ID
  - Save final dataset to datasets/drivingdatasets/input/
  - Generate pipeline_stats.json with execution statistics

  Expected runtime: 2-5 minutes for ~800 laps (VIR dataset)

### 7-3. Pipeline Options

  Customize the pipeline behavior with command-line arguments:

  ```bash
  python -m apexai.data_generation.pipeline \
    --raw-data-dir datasets/rawdata/VIR \
    --export-dir datasets/drivingdatasets/input \
    --target-frequency 10.0 \
    --original-frequency 23.0 \
    --train-ratio 0.7 \
    --valid-ratio 0.15 \
    --test-ratio 0.15 \
    --filter-last-lap \
    --shuffle \
    --random-seed 42 \
    --cleanup
  ```

  Key Parameters:

  | Parameter            | Type  | Default  | Description                                      |
  |----------------------|-------|----------|--------------------------------------------------|
  | --raw-data-dir       | str   | required | Directory containing raw telemetry CSV files     |
  | --export-dir         | str   | required | Output directory for train/valid/test splits     |
  | --target-frequency   | float | 10.0     | Target sampling frequency [Hz]                   |
  | --original-frequency | float | 23.0     | Original sampling frequency [Hz]                 |
  | --train-ratio        | float | 0.7      | Training set ratio (must sum to 1.0)             |
  | --valid-ratio        | float | 0.15     | Validation set ratio                             |
  | --test-ratio         | float | 0.15     | Test set ratio                                   |
  | --filter-last-lap    | flag  | True     | Remove last lap per vehicle (out laps)           |
  | --no-filter-last-lap | flag  | -        | Keep all laps including last lap                 |
  | --shuffle            | flag  | True     | Shuffle laps before splitting                    |
  | --no-shuffle         | flag  | -        | Sequential split without shuffling               |
  | --random-seed        | int   | 42       | Random seed for reproducibility                  |
  | --cleanup            | flag  | False    | Remove intermediate directories after completion |

### 7-4. Expected Output

  After successful execution, the following directory structure will be created:

  ```
  datasets/drivingdatasets/input/
  ├── train/
  │   ├── x_train/          # Training features (589 files for VIR)
  │   │   └── VIR_R1_vehicle_GR86-002-2_lap_001_x_train.csv
  │   └── y_train/          # Training labels (589 files)
  │       └── VIR_R1_vehicle_GR86-002-2_lap_001_y_train.csv
  ├── valid/
  │   ├── x_valid/          # Validation features (114 files)
  │   └── y_valid/          # Validation labels (114 files)
  ├── test/
  │   ├── x_test/           # Test features (154 files)
  │   └── y_test/           # Test labels (154 files)
  └── pipeline_stats.json   # Pipeline execution statistics
  ```

  File naming convention:
  ```
  {TRACK}_{RACE}_vehicle_{VEHICLE_ID}_lap_{LAP_NUM}_{x|y}_{train|valid|test}.csv
  ```

  Example: `VIR_R1_vehicle_GR86-002-2_lap_001_x_train.csv`

### 7-5. Pipeline Statistics

  The pipeline generates pipeline_stats.json with detailed execution metadata:

  ```json
  {
    "raw_data_dir": "datasets/rawdata/VIR",
    "export_dir": "datasets/drivingdatasets/input",
    "target_frequency": 10.0,
    "original_frequency": 23.0,
    "train_ratio": 0.7,
    "valid_ratio": 0.15,
    "test_ratio": 0.15,
    "steps": {
      "step1_resampling": {
        "total_laps": 836,
        "total_files": 836
      },
      "step2_training_files": {
        "total_x_files": 836,
        "total_y_files": 836
      },
      "step3_split": {
        "split_stats": {
          "train": 589,
          "valid": 114,
          "test": 154
        },
        "train_vehicles": 15,
        "valid_vehicles": 3,
        "test_vehicles": 3
      }
    }
  }
  ```

### 7-6. Advanced Usage

#### Running Individual Pipeline Steps

  For debugging or custom workflows, you can run each step separately:

  **Step 1: Generate resampled lap data**
  ```python
  from apexai.data_generation.preprocess_telemetry import preprocess_telemetry_to_laps

  preprocess_telemetry_to_laps(
      telemetry_path="datasets/rawdata/VIR/Race 1/R1_telemetry_data.csv",
      output_dir="datasets/preprocessed_10Hz/VIR/R1",
      resample_frequency=10.0,
      original_frequency=23.0,
      use_polars=True,
  )
  ```

  **Step 2: Generate X/y training files**
  ```python
  from apexai.data_generation.generate_training_data import generate_training_files

  generate_training_files(
      input_dir="datasets/preprocessed_10Hz/VIR/R1",
      output_dir="datasets/training_dataset_10Hz",
      separate_xy_dirs=True,
  )
  ```

  **Step 3: Split into train/valid/test**
  ```python
  from apexai.data_generation.split_dataset import (
      get_vehicle_files_from_training,
      split_and_copy_files,
  )

  vehicle_files = get_vehicle_files_from_training(
      x_dir="datasets/training_dataset_10Hz/x_train",
      y_dir="datasets/training_dataset_10Hz/y_train",
  )

  split_stats, vehicle_stats = split_and_copy_files(
      vehicle_files=vehicle_files,
      dest_base="datasets/drivingdatasets/input",
      train_ratio=0.7,
      valid_ratio=0.15,
      test_ratio=0.15,
      shuffle=True,
      filter_last_lap=True,
      random_seed=42,
  )
  ```

#### Using the Pipeline Programmatically

  ```python
  from pathlib import Path
  from apexai.data_generation.pipeline import run_pipeline

  pipeline_stats = run_pipeline(
      raw_data_dir=Path("datasets/rawdata/VIR"),
      export_dir=Path("datasets/drivingdatasets/input"),
      target_frequency=10.0,
      original_frequency=23.0,
      train_ratio=0.7,
      valid_ratio=0.15,
      test_ratio=0.15,
      filter_last_lap=True,
      shuffle=True,
      random_seed=42,
      intermediate_cleanup=True,  # Clean up intermediate files
  )

  print(f"Generated {pipeline_stats['steps']['step3_split']['split_stats']['train']} training files")
  ```

### 7-7. Intermediate Files (Optional Cleanup)

  During execution, the pipeline creates intermediate directories:

  ```
  datasets/
  ├── preprocessed_10Hz/
  │   └── VIR/
  │       └── R1/          # Resampled lap files
  └── training_dataset_10Hz/
      ├── x_train/         # Intermediate X files
      └── y_train/         # Intermediate y files
  ```

  Use `--cleanup` flag to automatically remove these after successful completion.





## 8. Using the Dashboard Tools

ApexAI integrates several powerful web-based tools for experiment tracking, hyperparameter optimization, and data management.

### 8-1. MLflow UI - Experiment Tracking

  Access the MLflow UI at http://localhost:5001 to:

  - **View Experiments**: Browse all your training runs with parameters and metrics
  - **Compare Runs**: Side-by-side comparison of different hyperparameter configurations
  - **Visualize Metrics**: Plot training/validation metrics over time
  - **Model Registry**: Download trained models and their artifacts

  **Key Features:**
  - Filter runs by parameters (e.g., `params.hidden_dim > 128`)
  - Sort by metrics to find best performing models
  - Download model checkpoints and training logs
  - Export experiment data to CSV

### 8-2. Optuna Dashboard - Hyperparameter Optimization

  Access the Optuna Dashboard at http://localhost:8081 to:

  - **Optimization History**: Visualize how trials improve over time
  - **Parameter Importance**: See which hyperparameters matter most
  - **Parallel Coordinate Plot**: Understand parameter interactions
  - **Study Management**: Monitor ongoing optimization studies

  **Key Features:**
  - Real-time visualization of optimization progress
  - Hyperparameter importance analysis
  - Pareto front visualization for multi-objective optimization
  - Trial pruning and early stopping insights

### 8-3. pgAdmin - Database Management

  Access pgAdmin at http://localhost:5051 (default credentials in `.env` file) to:

  - **Inspect Experiment Data**: Query MLflow and Optuna databases directly
  - **Custom Analysis**: Write SQL queries for advanced analytics
  - **Database Backup**: Export experiment data for long-term storage
  - **Performance Monitoring**: Check database health and performance

  **Initial Setup:**
  1. Log in with credentials from `.env` file
  2. Right-click "Servers" → "Register" → "Server"
  3. Name: `ApexAI DB`
  4. Connection tab: Host=`mlflow_optuna_pg`, Port=`5432`, Username=`mlflow`, Password from `.env`

  ### 8-4. MinIO Console - Artifact Storage

  Access the MinIO Console at http://localhost:9020 (credentials: minioadmin/minioadmin) to:

  - **Browse Artifacts**: View model checkpoints, logs, and plots
  - **Manage Storage**: Monitor disk usage and bucket organization
  - **Download Models**: Retrieve trained model files
  - **Bucket Management**: Organize experiments by creating buckets

  **Key Features:**
  - S3-compatible API for programmatic access
  - Built-in file browser
  - Access control and bucket policies
  - Object versioning


## 9. Training your AI models

ApexAI provides three powerful training modes to match different research and development needs:

1. **Single Training** - Train a specific model with fixed hyperparameters
2. **Hyperparameter Optimization (HPO)** - Optimize hyperparameters for a specific model architecture
3. **Neural Architecture Search (NAS)** - BO-NAS + HPO AutoML for fully automated model discovery

All training modes are fully integrated with MLflow for experiment tracking and artifact management.

### 9-1. Single Model Training

Train a specific model (GRU, LSTM, Transformer, or Informer) with predefined hyperparameters using `trainer_entrypoint.py`.

#### 9-1-1. Basic Usage

```bash
python -m apexai.trainer_entrypoint
```

This will:
- Load configuration from `conf/config.yaml`
- Use default model specified in `conf/model/gru.yaml`
- Train with settings from `conf/training/default.yaml`
- Log all metrics and artifacts to MLflow

#### 9-1-2. Custom Model Selection

Train different model architectures:

```bash
# Train GRU model (default)
python -m apexai.trainer_entrypoint model=gru

# Train LSTM model
python -m apexai.trainer_entrypoint model=lstm

# Train Transformer model
python -m apexai.trainer_entrypoint model=transformer

# Train Informer model
python -m apexai.trainer_entrypoint model=informer
```

#### 9-1-3. Override Hyperparameters

Customize training parameters on the fly:

```bash
# Override training epochs and batch size
python -m apexai.trainer_entrypoint \
  training.epochs=100 \
  training.batch_size=128

# Override learning rate and model hidden dimension
python -m apexai.trainer_entrypoint \
  training.optimizer.learning_rate=0.001 \
  model.architecture.hidden_dim=512

# Combine multiple overrides
python -m apexai.trainer_entrypoint \
  model=lstm \
  training.epochs=150 \
  training.batch_size=64 \
  model.architecture.num_layers=3 \
  model.architecture.dropout_ratio=0.2
```

#### 9-1-4. MLflow Integration

All training runs automatically log:
- **Parameters**: Model architecture, hyperparameters, optimizer settings
- **Metrics**: Training/validation loss, accuracy, F1-score per epoch
- **Artifacts**:
  - Trained model with signature and input example
  - Confusion matrices (normalized and raw)
  - Training configuration snapshot

Access results at: http://localhost:5001

### 9-2. Hyperparameter Optimization (HPO)

Automatically find optimal hyperparameters for a specific model architecture using Optuna.

#### 9-2-1. Basic Usage

```bash
# HPO for GRU model
python -m apexai.optimization_entrypoint \
  model=gru \
  optuna=gru_hpo
```

This will:
- Search hyperparameters defined in `conf/optuna/gru_hpo.yaml`
- Run up to 1000 trials (or 2 hours, whichever comes first)
- Optimize for F1-score maximization
- Save Top-5 best models to MLflow
- Store optimization history in PostgreSQL

#### 9-2-2. Available HPO Configurations

```bash
# GRU-specific HPO (2 hours, moderate search space)
python -m apexai.optimization_entrypoint model=gru optuna=gru_hpo

# General HPO for any model (10 hours, extensive search)
python -m apexai.optimization_entrypoint model=lstm optuna=default

# Quick test run (5 trials for debugging)
python -m apexai.optimization_entrypoint \
  model=gru \
  optuna=gru_hpo \
  optuna.optimization.n_trials=5 \
  optuna.optimization.timeout=300
```

#### 9-2-3. HPO Search Space

The `gru_hpo.yaml` configuration searches:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `hidden_dim` | int | [256, 512] step 64 | GRU hidden state dimension |
| `num_layers` | int | [1, 4] | Number of stacked GRU layers |
| `dropout_ratio` | float | [0.0, 0.3] step 0.1 | Dropout regularization |
| `epochs` | int | [50, 100] step 5 | Training epochs |
| `batch_size` | int | [64, 256] step 64 | Batch size |
| `learning_rate` | float | [1e-4, 1e-3] log scale | AdamW learning rate |
| `weight_decay` | float | [1e-4, 1e-2] log scale | L2 regularization |

#### 9-2-4. Optimization Strategy

- **Sampler**: TPE (Tree-structured Parzen Estimator) with Bayesian optimization
- **Pruner**: Median pruner to early-stop unpromising trials
- **Objective**: Maximize validation F1-score
- **Parallel Jobs**: 1 (configurable via `optuna.optimization.n_jobs`)

#### 9-2-5. Viewing Results

**Optuna Dashboard** (http://localhost:8081):
- Optimization history plot
- Parameter importance analysis
- Parallel coordinate plot
- Trial filtering and sorting

**MLflow UI** (http://localhost:5001):
- Parent run: `Optuna_ApexAI_GRUHPO_gru_w/Norm_f1Maximize_feature9_v2`
- Child runs: Individual trials with full metrics
- Top-K models: Best performing models saved as artifacts

### 9.3 Neural Architecture Search (NAS) - BO-NAS + HPO AutoML

ApexAI implements a powerful **Bayesian Optimization-based NAS with integrated HPO**, enabling fully automated model discovery and hyperparameter tuning in a single optimization run.

### 9-3. BO-NAS + HPO (AutoML)

This is an advanced AutoML approach that combines:

1. **Neural Architecture Search (NAS)**: Search across multiple model architectures
   - Model types: GRU, LSTM, Transformer, Informer
   - Architecture parameters: hidden_dim, num_layers, dropout_ratio
   - Attention parameters: num_heads, feedforward_multiplier (Transformer/Informer only)

2. **Hyperparameter Optimization (HPO)**: Optimize training hyperparameters
   - Learning rate, batch size, epochs
   - Weight decay (L2 regularization)

3. **Bayesian Optimization**: Efficient search using TPE (Tree-structured Parzen Estimator)
   - Learns from previous trials to suggest better candidates
   - Much faster than random/grid search

4. **Conditional Parameter Suggestion**: Architecture-aware parameter selection
   - Transformer/Informer: Suggests `num_heads`, `feedforward_multiplier`
   - GRU/LSTM: Skips attention-specific parameters
   - Prevents invalid parameter combinations

#### 9-3-1. Key Technical Features

**Hierarchical Search Strategy**:
```python
# Step 1: Select model architecture first
model_type ∈ {GRU, LSTM, Transformer, Informer}

# Step 2: Suggest architecture-specific parameters
if model_type in [Transformer, Informer]:
    num_heads ∈ {2, 4, 8}
    feedforward_multiplier ∈ {2, 4, 6, 8}

# Step 3: Suggest common hyperparameters
hidden_dim, num_layers, dropout, learning_rate, batch_size, ...
```

**Automatic Pruning**:
- MedianPruner stops unpromising trials early (saves ~40% compute time)
- Warmup: First 3 epochs are not pruned (allow initial learning)
- Startup: First 5 trials run to completion (establish baseline)

**Top-K Model Selection**:
- Automatically saves best K models (default: 5) after optimization
- Models ranked by objective value (F1-score) regardless of architecture
- Each model saved with full config and checkpoint to MLflow

#### 9-3-2. Basic Usage

```bash
python -m apexai.optimization_entrypoint optuna=nas
```

This single command will:
- Search 4 architectures × hyperparameter space
- Run Bayesian optimization for 12 hours (or 1000 trials)
- Prune poor performers automatically
- Save Top-5 best models to MLflow
- Generate comprehensive optimization analytics in Optuna Dashboard

#### 9-3-3. Full NAS Search Space

From `conf/optuna/nas.yaml`:

| Category | Parameter | Type | Range/Choices | Models |
|----------|-----------|------|---------------|--------|
| **Architecture Selection** |
| | `model_type` | categorical | [GRU, LSTM, Transformer, Informer] | All |
| **Common Architecture** |
| | `hidden_dim` | int | [128, 512] step 64 | All |
| | `num_layers` | int | [1, 4] | All |
| | `dropout_ratio` | float | [0.0, 0.3] step 0.1 | All |
| | `seq_len` | int | [100, 200] step 100 | All |
| **Attention-Specific** |
| | `num_heads` | categorical | [2, 4, 8] | Transformer, Informer |
| | `feedforward_multiplier` | int | [2, 8] step 2 | Transformer, Informer |
| **Training Hyperparameters** |
| | `epochs` | int | [50, 100] step 1 | All |
| | `batch_size` | int | [64, 256] step 64 | All |
| | `learning_rate` | float | [1e-4, 1e-2] log scale | All |
| | `weight_decay` | float | [1e-4, 1e-2] log scale | All |

**Total Search Space Size**: ~10^9 combinations (Bayesian Optimization makes this tractable)

#### 9-3-4. AutoML Configuration

From `conf/optuna/nas.yaml`:

```yaml
# Bayesian Optimization settings
sampler:
  name: "tpe"                  # Tree-structured Parzen Estimator
  n_startup_trials: 8          # Random trials before BO starts
  n_ei_candidates: 24          # Candidates for Expected Improvement
  multivariate: false          # Independent parameter treatment

# Automatic pruning settings
pruner:
  name: "median"               # Median-based pruning
  n_startup_trials: 5          # No pruning for first 5 trials
  n_warmup_steps: 3            # No pruning for first 3 epochs
  interval_steps: 1            # Check pruning every epoch

# Optimization budget
optimization:
  n_trials: 1000               # Maximum trials (auto-stop at timeout)
  timeout: 43200               # 12 hours (overnight run)
  n_jobs: 1                    # Sequential trials (GPU memory)

# Model selection
model_saving:
  top_k_models: 5              # Save top 5 best models
  optimization_metric: "f1_score"  # Rank by F1-score
```

#### 9-3-5. Example NAS Results

After 12-hour optimization (example output):

```
========================================
OPTIMIZATION COMPLETE
========================================
Best trial: 127
Best value: 0.9234 (F1-score)
Best parameters:
  - model_type: Transformer
  - hidden_dim: 448
  - num_layers: 3
  - num_heads: 8
  - feedforward_multiplier: 4
  - dropout_ratio: 0.1
  - learning_rate: 0.000423
  - batch_size: 128
  - epochs: 85

Top-5 Models Saved to MLflow:
  Rank 1: Transformer (F1: 0.9234) - Trial 127
  Rank 2: GRU (F1: 0.9187) - Trial 89
  Rank 3: Informer (F1: 0.9165) - Trial 203
  Rank 4: LSTM (F1: 0.9142) - Trial 56
  Rank 5: Transformer (F1: 0.9098) - Trial 178

Total trials: 247 (753 pruned)
Time: 12h 00m 45s
========================================
```

#### 9-3-6. Analyzing Results

**Optuna Dashboard** (http://localhost:8081):
1. **Optimization History**: See F1-score improve over trials
2. **Parameter Importance**: Which parameters matter most?
   - Example: `hidden_dim` (importance: 0.42), `model_type` (0.31), `learning_rate` (0.18)
3. **Parallel Coordinate Plot**: Visualize high-performing parameter combinations
4. **Slice Plot**: Understand parameter-objective relationships

**MLflow UI** (http://localhost:5001):
- Parent run: `Optuna_ApexAI_NAS_v2`
- 247 child runs: Individual trials with metrics
- 5 top model runs: Best models with artifacts

#### 9-3-7. Advanced: Custom BO-NAS Configuration

Create your own NAS configuration:

```yaml
# conf/optuna/my_lightweight_nas.yaml
optuna_params:
  # Limit architectures to RNN-based only
  model_type:
    type: "categorical"
    choices:
      - "GRU"
      - "LSTM"

  # Narrow search space for faster optimization
  hidden_dim:
    type: "int"
    low: 256
    high: 512
    step: 128

  num_layers:
    type: "int"
    low: 2
    high: 3

optimization:
  n_trials: 50
  timeout: 3600  # 1 hour for quick NAS
```

Run with:
```bash
python -m apexai.optimization_entrypoint optuna=my_lightweight_nas
```

#### 9-3-8. When to Use BO-NAS + HPO

**Use NAS when**:
- You don't know which architecture is best for your data
- You want to discover optimal model + hyperparameters automatically
- You have 8-24 hours of compute budget
- You need state-of-the-art performance

**Use HPO instead when**:
- You already know the best architecture (e.g., GRU works well)
- You only need to tune hyperparameters
- You have limited compute budget (2-4 hours)

**Use Single Training when**:
- You have good hyperparameters from previous runs
- You're doing ablation studies (changing one thing at a time)
- You need quick iteration for debugging

### 9-4. Monitoring Training Progress

#### 9-4-1. Real-Time Monitoring

**During Single Training**:
```bash
# Terminal shows epoch-by-epoch progress:
Epoch [1/100] - Train Loss: 2.1234, Val Loss: 1.9876, Val F1: 0.7654
Epoch [2/100] - Train Loss: 1.8765, Val Loss: 1.7543, Val F1: 0.8123
...
```

**During HPO/NAS**:
```bash
# Terminal shows trial progress:
[I 2025-01-18 10:30:45] Trial 0 finished with value: 0.8234
[I 2025-01-18 10:35:12] Trial 1 finished with value: 0.8567
[I 2025-01-18 10:36:03] Trial 2 pruned at epoch 5
...
```

#### 9-4-2. Dashboard Monitoring

1. **MLflow UI** (http://localhost:5001):
   - Live metric plots update as training progresses
   - Compare multiple runs side-by-side
   - Filter by parameters or metrics

2. **Optuna Dashboard** (http://localhost:8081):
   - Real-time optimization progress
   - Trial state visualization (running/complete/pruned/failed)
   - Parameter importance updates

### 9-5 Best Practices

#### 9-5-1. For Single Training

1. **Start Simple**: Begin with default configurations
2. **Monitor Overfitting**: Watch train vs. validation gap
3. **Use MLflow**: Tag runs with meaningful names for organization
4. **Experiment Systematically**: Change one parameter at a time

#### 9-5-2. For HPO

1. **Start with Quick Test**: Run 5-10 trials to verify setup
2. **Use Appropriate Timeout**: 2-4 hours for serious optimization
3. **Check Search Space**: Ensure parameter ranges are reasonable
4. **Review Pruned Trials**: High pruning rate indicates poor initialization

#### 9-5-3. For NAS

1. **Allocate Sufficient Time**: 12+ hours for comprehensive search
2. **Monitor Resource Usage**: Check GPU memory and utilization
3. **Analyze Results**: Use Optuna Dashboard for architecture insights
4. **Compare to Baselines**: Ensure NAS improves over manual tuning


## 10. Real-Time Vehicle ID Inference Simulator

  ApexAI provides a powerful **Streamlit-based real-time inference simulator** that emulates production-grade
  vehicle telemetry streaming.
  The simulator is designed to replicate real-world scenarios where telemetry data is continuously transmitted
  from race vehicles via streaming platforms like **Apache Kafka**.

### 10-1. Simulator Overview

#### 10-1-1. Real-World Streaming Architecture

  The simulator emulates the following production pipeline:

  ```
  ┌─────────────────┐   ~22Hz CAN bus data   ┌──────────────────────────────────────────┐
  │  Race Vehicle   │ ────────────────────────> │       ApexAI Backend                   │
  │  (Telemetry)    │   Kafka Stream          │                                         │
  └─────────────────┘   (pbrake_f, pbrake_r,  │  ┌────────────────────────────────────┐ │
                         Steering_Angle, accx, │  │  1. Resampling to 10Hz             │ │
                         accy, ath, gear,      │  │     (from Kafka messages)          │ │
                         nmot, speed)          │  └─────────────┬──────────────────────┘ │
                                               │                ▼                        │
                                               │  ┌────────────────────────────────────┐ │
                                               │  │  2. Sliding Window Buffer          │ │
                                               │  │     (Queue: seq_len=100 samples)   │ │
                                               │  └─────────────┬──────────────────────┘ │
                                               │                ▼                        │
                                               │  ┌────────────────────────────────────┐ │
                                               │  │  3. Model Inference                │ │
                                               │  │     (When buffer full: 100 samples)│ │
                                               │  └─────────────┬──────────────────────┘ │
                                               │                ▼                        │
                                               │  ┌────────────────────────────────────┐ │
                                               │  │  4. TTA Voting Queue               │ │
                                               │  │     (6-vote buffer)                │ │
                                               │  └─────────────┬──────────────────────┘ │
                                               │                ▼                        │
                                               └────────────────────────────────────────┘
                                                                │
                                                                ▼
                                                      ┌─────────────────┐
                                                      │  Vehicle ID     │
                                                      │  Prediction     │
                                                      └─────────────────┘
  ```

  **Design Philosophy:**
  - **Queue-based buffering**: Incoming telemetry samples are stored in a sliding window queue
  - **Fixed-interval inference**: Predictions run when buffer reaches `seq_len` samples
  - **Asynchronous streaming**: Simulates continuous data flow from Kafka stream
  - **Production-ready**: Architecture mirrors actual deployment with Kafka message broker

#### 10-1-2. GR86 Telemetry Data & Kafka Integration

  The VIR (Virginia International Raceway) dataset used in ApexAI was originally collected with **Kafka streaming infrastructure**:

  ```
  # Original data collection setup (GR86 Cup Race)
  CAN Bus (Vehicle) → Kafka Producer → Kafka Topic (telemetry_stream)
                                            ↓
                                Kafka Consumer → CSV Logger
  ```

  **Why Kafka?**
  - **Real-time**: Sub-100ms latency from vehicle to processing
  - **Scalability**: Handle 21 vehicles × 10Hz = 210 messages/sec
  - **Reliability**: Message persistence and replay capability
  - **Standardization**: Industry-standard for automotive telemetry

  The simulator **replicates this Kafka streaming behavior** by:
  1. Reading CSV files as if they were Kafka messages
  2. Streaming samples at fixed intervals (10Hz)
  3. Buffering in a queue (sliding window)
  4. Triggering inference when buffer is full

#### 10-1-3. Simulator Features

  1. **Real-Time Inference**: Process telemetry data at 10Hz with live predictions
  2. **Two Inference Modes**:
    - **Single Model**: Use one trained model with TTA voting
    - **Multi-Model Ensemble**: Combine predictions from multiple models
  3. **Queue-Based Buffering**: Sliding window buffer simulates Kafka consumer behavior
  4. **Test-Time Augmentation (TTA)**: Majority voting over sliding windows for robust predictions
  5. **Live Visualization**: Real-time sensor data plots (steering, throttle, brake)
  6. **Prediction Timeline**: Track prediction changes over time

### 10-2. Quick Start

#### Step 1: Launch Simulator

```bash
streamlit run apexai/simulator/frontend/app.py
```

  The simulator will open in your browser at: http://localhost:8501

#### Step 2: Configure Model Mode

  In the sidebar, select your inference mode:

  **Option A: Single Model**
  - Select "Single Model" mode
  - Config path: `apexai/simulator/model_repository/prodmodel.yaml`
  - Uses TTA with majority voting (default: 6 votes)

  **Option B: Multi-Model Ensemble**
  - Select "Multi-Model Ensemble" mode
  - Model 1: `apexai/simulator/model_repository/prodmodel1.yaml` (e.g., Transformer)
  - Model 2: `apexai/simulator/model_repository/prodmodel2.yaml` (e.g., GRU)
  - Ensemble method: `majority` or `weighted`

#### Step 3: Upload Telemetry Data

  Upload a CSV file containing race telemetry:
  - Format: Must contain feature columns (pbrake_f, pbrake_r, Steering_Angle, etc.)
  - Source: Use files from `datasets/drivingdatasets/input/test/x_test/`

  Example file: `VIR_R1_vehicle_GR86-002-2_lap_001_x_test.csv`

  **Note:** CSV files represent historical Kafka stream data. Each row = one Kafka message.

#### Step 4: Initialize Simulator

  Click **"🚀 Initialize Simulator"** to:
  - Load model(s) from specified config paths
  - Set up data streamer with 10Hz sampling (emulates Kafka consumer)
  - Create sliding window buffer (seq_len=100) as queue
  - Initialize TTA voting mechanism

#### Step 5: Run Simulation

  Control simulation with three buttons:
  - **▶️ Start**: Begin real-time streaming and inference
  - **⏸️ Pause**: Pause stream (resume with Start)
  - **🔄 Restart**: Reset simulation to beginning

### 10-3. Streaming Architecture Deep Dive

#### 10-3-1. Queue-Based Buffering System

  The simulator implements a **sliding window queue** that mimics Kafka consumer behavior:

```python
class SlidingWindowBuffer:
    """Emulates Kafka consumer buffer for real-time streaming."""

    def __init__(self, seq_len: int = 100, feature_size: int = 9):
        self.seq_len = seq_len          # Queue capacity
        self.feature_size = feature_size
        self.buffer = deque(maxlen=seq_len)  # FIFO queue

    def add_sample(self, sample: np.ndarray):
        """Add incoming Kafka message to queue."""
        self.buffer.append(sample)

    def is_ready(self) -> bool:
        """Check if queue has enough samples for inference."""
        return len(self.buffer) == self.seq_len

    def get_sequence(self) -> np.ndarray:
        """Retrieve buffered sequence for model input."""
        return np.array(list(self.buffer))
```

**Queue Behavior:**
1. **FIFO (First-In-First-Out)**: Oldest samples removed when queue is full
2. **Fixed capacity**: `maxlen=seq_len` (default: 100 samples)
3. **Ready signal**: Inference triggered when `len(buffer) == seq_len`
4. **Continuous buffering**: Queue persists across predictions (sliding window)

#### 10-3-2. Real-Time Data Streaming

The `DataStreamer` class emulates Kafka consumer reading from a topic:

```python
class DataStreamer:
    """Emulates Kafka consumer for telemetry stream."""

    def __init__(self, csv_path: str, sampling_rate: int = 10):
        self.sampling_rate = sampling_rate  # 10Hz (Kafka message rate)
        self.interval = 1.0 / sampling_rate  # 0.1 seconds between messages
        self.data = pd.read_csv(csv_path)   # Simulated Kafka topic data
        self.current_index = 0               # Message offset

    def get_next_sample(self) -> np.ndarray | None:
        """Fetch next message from stream (emulates Kafka poll)."""
        if self.current_index >= len(self.data):
            return None  # End of stream

        sample = self.data.iloc[self.current_index].values
        self.current_index += 1  # Increment offset
        return sample
```

**Kafka Message Structure (Simulated):**
```json
{
  "timestamp": 1642345678.123,
  "vehicle_id": "GR86-002-2",
  "features": {
    "pbrake_f": 0.0,
    "pbrake_r": 0.0,
    "Steering_Angle": -15.3,
    "accx_can": 0.45,
    "accy_can": -0.12,
    "ath": 35.2,
    "gear": 3,
    "nmot": 5200,
    "speed": 85.3
  }
}
```

#### 10-3-3. Inference Trigger Logic

**Fixed-Interval Inference** (Production Pattern):

  ```python
  def process_next_sample():
      """Main streaming loop - runs at 10Hz."""
      # 1. Poll next message from stream (Kafka consumer)
      sample = st.session_state.streamer.get_next_sample()

      # 2. Add to queue buffer
      st.session_state.buffer.add_sample(sample)

      # 3. Check if buffer is ready for inference
      if st.session_state.buffer.is_ready():
          sequence = st.session_state.buffer.get_sequence()

          # 4. Run model inference
          predicted_class, probabilities = engine.predict(sequence)

          # 5. Add to TTA voting buffer (secondary queue)
          st.session_state.tta.add_prediction(predicted_class)

          # 6. Clear buffer (prepare for next window)
          st.session_state.buffer.clear()

      # 7. Advance time (0.1s = 10Hz)
      st.session_state.elapsed_time += 0.1
  ```

**Timing Diagram:**
  ```
  Time (s)  | Action                        | Buffer State  | Inference
  ----------|-------------------------------|---------------|----------
  0.0       | Sample 1 arrives (Kafka msg) | 1/100        | Waiting
  0.1       | Sample 2 arrives              | 2/100        | Waiting
  ...       | ...                           | ...          | ...
  9.9       | Sample 100 arrives            | 100/100      | ✓ Ready!
  10.0      | Inference #1 runs             | Cleared      | Running
  10.0      | Sample 101 arrives            | 1/100        | Waiting
  10.1      | Sample 102 arrives            | 2/100        | Waiting
  ...       | ...                           | ...          | ...
  19.9      | Sample 200 arrives            | 100/100      | ✓ Ready!
  20.0      | Inference #2 runs             | Cleared      | Running
  ```

### 10-4. Model Configuration Files

Create YAML config files in `apexai/simulator/model_repository/`:

#### 10-4-1. Single Model Configuration

```yaml
# apexai/simulator/model_repository/prodmodel.yaml
name: "Transformer_Best"
type: "Transformer"
model_path: "path/to/your/model.pth"

# ==========================================
# 📊 Model Architecture
# ==========================================
architecture:
  # Input sequence length (must match streaming buffer size)
  seq_len: 100

  # Expected feature dimension per sample (Kafka message features)
  feature_size: 9  # pbrake_f, pbrake_r, Steering_Angle, accx_can, accy_can, ath, gear, nmot, speed

  # Hidden state dimension of Transformer (d_model)
  hidden_dim: 448

  # Number of stacked Transformer Encoder layers
  num_layers: 3

  # Number of Multi-Head Attention heads
  num_heads: 8

  # Feedforward network dimension multiplier
  # Actual FFN dimension is calculated as hidden_dim * feedforward_multiplier
  # Example: hidden_dim=448 × feedforward_multiplier=4 = FFN_dim=1792
  feedforward_multiplier: 4

  # Dropout ratio (0.0-1.0, for overfitting prevention)
  dropout_ratio: 0.1

  # Output dimension (must match number of classification classes)
  out_dim: 21

  # Number of classification classes
  num_classes: 21

  # Classification task flag (true: classification, false: regression)
  classification: true

  # Batch-first dimension ordering (PyTorch standard format)
  batch_first: true

  # Positional encoding configuration
  max_position_embeddings: 512  # Maximum sequence length

# ==========================================
# 🎛️ Data Preprocessing
# ==========================================
preprocessing:
  features:           # Features extracted from Kafka messages
    - pbrake_f
    - pbrake_r
    - Steering_Angle
    - accx_can
    - accy_can
    - ath
    - gear
    - nmot
    - speed

  normalization:
    enabled: true
    method: "minmax"
    # Min-Max values from training data statistics (EDA.ipynb)
    # CRITICAL: These values MUST match the normalization used during model training
    # Otherwise, inference predictions will be incorrect!
    min: [0.0, 0.0, -467.057242, -2.305603, -3.637093, -16.022081, -1.150162, 0.0, 0.0]
    max: [180.869085, 182.249066, 466.393638, 2.481207, 1.924832, 112.747288, 6.083910, 7851.806179, 217.203306]

# ==========================================
# ⚙️ Inference Configuration
# ==========================================
sampling_rate: 10  # Hz (Kafka message rate / CSV data frequency)
num_votes: 6       # TTA window size (secondary voting queue)

# ==========================================
# 🏷️ Label Configuration
# ==========================================
labels:
  num_classes: 21
  class_names:
    0: "GR86-002-2"
    1: "GR86-006-7"
    2: "GR86-013-80"
    3: "GR86-015-31"
    4: "GR86-016-55"
    5: "GR86-022-13"
    6: "GR86-024-41"
    7: "GR86-026-72"
    8: "GR86-028-89"
    9: "GR86-032-15"
    10: "GR86-033-46"
    11: "GR86-036-98"
    12: "GR86-038-93"
    13: "GR86-040-3"
    14: "GR86-047-21"
    15: "GR86-049-88"
    16: "GR86-051-71"
    17: "GR86-061-51"
    18: "GR86-062-012"
    19: "GR86-063-113"
    20: "GR86-065-5"
```

#### 10-4-2. Multi-Model Ensemble Configuration

**Model 1 (Transformer):**
```yaml
# apexai/simulator/model_repository/prodmodel1.yaml
name: "Transformer_Model1"
type: "Transformer"
model_path: "models/transformer_trial127.pth"
# ... (same structure as single model config)
```

**Model 2 (GRU):**
```yaml
# apexai/simulator/model_repository/prodmodel2.yaml
name: "GRU_Model2"
type: "GRU"
model_path: "models/gru_trial89.pth"
# ... (same structure with GRU architecture params)
```

### 10-5. Understanding the UI

#### 10-5-1. Real-Time Results Panel

**Metrics Row:**
- **⏱️ Time**: Elapsed simulation time (seconds) - simulates Kafka stream duration
- **🔢 Inferences**: Total number of predictions made from buffered sequences
- **🎯 Final Prediction**: Current TTA/Ensemble result

**Latest Inference (Left Column):**

*Single Model Mode:*
- Shows Top-5 predictions with confidence scores
- Updates every 0.1s (10Hz streaming rate)

*Multi-Model Ensemble Mode:*
- Shows predictions from each model
- Top-3 predictions per model
- Arrow (→) indicates selected class

**Final Prediction Details (Right Column):**

*Single Model Mode:*
- **Vote Counts**: TTA voting breakdown
- Shows which vehicles received votes

*Multi-Model Ensemble Mode:*
- **Vote Counts**: Combined ensemble votes
- **Model Agreement**: Percentage of agreement between models
- **Total Votes**: Total predictions considered

#### 10-5-2. Sensor Data Visualization

Three real-time plots (updates every 1 second):
1. **Steering Angle (deg)**: Blue line
2. **Throttle (%)**: Green line
3. **Brake Pressure (front/rear)**: Red/Orange lines

**Prediction Markers:**
- Light gray dashed lines: Individual predictions (every 100 samples)
- Red solid lines: TTA/Ensemble final predictions (every 6 predictions)

### 10-6. Inference Modes Explained

#### 10-6-1. Single Model + TTA

**How it works (with queue buffering):**
  ```python
  # Streaming Loop @ 10Hz:
  1. Poll next Kafka message (CSV row)
  2. Enqueue sample to sliding window buffer (FIFO queue, maxlen=100)
  3. When queue full (100 samples) → Run inference
  4. Add prediction to TTA vote queue (maxlen=6)
  5. When 6 votes collected → Majority vote = Final prediction
  6. Clear window buffer, continue streaming
  ```

**Example Timeline:**
  ```
  Time 0.0s: Queue filling... (0/100 samples in buffer)
  Time 9.9s: Queue full (100/100) → Predict: GR86-002-2 (Vote 1/6)
           → Clear buffer, resume streaming
  Time 10.0s: New queue (1/100) → Wait...
  Time 19.9s: Queue full (100/100) → Predict: GR86-002-2 (Vote 2/6)
  Time 29.9s: Queue full (100/100) → Predict: GR86-006-7 (Vote 3/6)
  ...
  Time 59.9s: 6 votes collected → Final: GR86-002-2 (4/6 votes)
  ```

**Queue Benefits:**
- **Decouples streaming from inference**: Buffer absorbs rate variations
- **Reduces jitter**: Majority voting over multiple buffered predictions
- **Handles bursty streams**: Queue smooths irregular Kafka message arrivals
- **Production-ready**: Standard pattern for stream processing

#### 10-6-2. Multi-Model Ensemble + TTA

**How it works:**
  ```python
  # Streaming Loop @ 10Hz:
  1. Poll next Kafka message
  2. Enqueue to shared sliding window buffer (seq_len=100)
  3. When buffer full:
     - Model 1 (Transformer) → Prediction A (from buffered sequence)
     - Model 2 (GRU) → Prediction B (from same buffered sequence)
  4. Add both predictions to ensemble vote queue (6 per model = 12 total)
  5. When 12 votes collected:
     - Majority voting: Combine all votes
     - Weighted voting: Weight by model confidence
  6. Final prediction = Ensemble result
  7. Clear buffer, continue streaming
  ```

**Ensemble Methods:**

**Majority Voting:**
  - Each model gets equal weight
  - Final prediction = Most voted class across all buffered predictions
  - Simple and robust

**Weighted Voting:**
  - Predictions weighted by confidence scores
  - Final prediction = Highest weighted sum
  - Better when models have different reliability

**Example:**
  ```
  Time 59.9s (after 6 inference cycles):
    Model 1 (Transformer): [GR86-002-2: 4 votes, GR86-006-7: 2 votes]
    Model 2 (GRU):         [GR86-002-2: 5 votes, GR86-015-31: 1 vote]

    Majority Ensemble: GR86-002-2 (9/12 votes = 75%)
    Model Agreement: 83.3% (5/6 buffered predictions agreed)
  ```

### 10-7. Preparing Test Data

#### 10-7-1. Extract Test Files from Pipeline Output

  After running the data pipeline (Section 7), you'll have test files:

  ```bash
  datasets/drivingdatasets/input/test/x_test/
  ├── VIR_R1_vehicle_GR86-002-2_lap_001_x_test.csv  # Historical Kafka data
  ├── VIR_R1_vehicle_GR86-002-2_lap_002_x_test.csv
  ├── VIR_R1_vehicle_GR86-006-7_lap_001_x_test.csv
  └── ... (154 total test files from Kafka stream recordings)
  ```

  Use any of these files in the simulator (they represent historical Kafka stream data).

#### 10-7-2. Create Prediction Data (Optional)

  Generate new test data from raw telemetry:

  ```bash
  python -m apexai.data_generation.pipeline \
    --raw-data-dir datasets/rawdata/VIR \
    --export-dir datasets/drivingdatasets/input
  ```

  Then copy files from `test/x_test/` to test in simulator.

### 10-8. Exporting Trained Models for Simulator

#### 10-8-1. From MLflow UI

1. Go to MLflow UI: http://localhost:5001
2. Navigate to your best run (from training or optimization)
3. In "Artifacts" section → Download `model/data/model.pth`
4. Place in `apexai/simulator/model_repository/models/`

#### 10-8-2. From Optuna Top-K Models

  After NAS/HPO completion, Top-K models are saved in MLflow:

  ```bash
  # Find Top-1 model run in MLflow UI
  # Download artifact: models/trial_127_rank_1.pth
  # Place at: apexai/simulator/model_repository/models/transformer_trial127.pth
  ```

#### 10-8-3. Update Config File

  Create corresponding YAML config with matching architecture:

  ```yaml
  # Must match training configuration exactly
  model_path: "apexai/simulator/model_repository/models/transformer_trial127.pth"
  architecture:
    seq_len: 100         # Must match queue buffer size
    hidden_dim: 448      # From Optuna best trial
    num_layers: 3
    num_heads: 8
    feedforward_multiplier: 4
    dropout_ratio: 0.1
    num_classes: 21
    # ... (all other params from trial)

  preprocessing:
    normalization:
      enabled: true
      method: "minmax"
      # CRITICAL: Use exact same normalization values as training!
      min: [0.0, 0.0, -467.057242, -2.305603, -3.637093, -16.022081, -1.150162, 0.0, 0.0]
      max: [180.869085, 182.249066, 466.393638, 2.481207, 1.924832, 112.747288, 6.083910, 7851.806179, 217.203306]
  ```

**⚠️ CRITICAL: Normalization Parameter Consistency**

  The normalization parameters (`min`, `max`) **MUST match exactly** between training and inference:

  1. **Where to find training normalization values:**
     - Check `conf/data/toyota_gr86.yaml` used during training
     - Or refer to `EDA.ipynb` where statistics were calculated
     - MLflow run parameters may also contain normalization info

  2. **Why this matters:**
     ```python
     # Training: x_normalized = (x - min) / (max - min)
     # If inference uses different min/max:
     # → Model receives different input distribution
     # → Predictions will be completely wrong!
     ```

  3. **How to verify:**
     ```python
     # Test normalization consistency
     import numpy as np

     sample = np.array([0.0, 0.0, 15.3, 0.45, -0.12, 35.2, 3, 5200, 85.3])
     min_vals = np.array([0.0, 0.0, -467.057242, -2.305603,
                          -3.637093, -16.022081, -1.150162, 0.0, 0.0])
     max_vals = np.array([180.869085, 182.249066, 466.393638, 2.481207,
                          1.924832, 112.747288, 6.083910, 7851.806179, 217.203306])

     normalized = (sample - min_vals) / (max_vals - min_vals)
     print(f"Normalized sample: {normalized}")
     # Should be in range [0, 1] for all features
     ```

  4. **Common mistakes:**
     - Using default values instead of training statistics
     - Calculating new min/max from test data (causes distribution shift)
     - Forgetting to enable normalization (`enabled: true`)
     - Using wrong normalization method (`minmax` vs `zscore`)

  5. **Best practice:**
     - Save normalization parameters with model during training
     - Document min/max values in MLflow run
     - Version control your config files
     - Test inference with known samples before deployment


## Troubleshooting

### Services Not Starting

  Check service status:
  ```bash
  docker-compose ps
  ```

  View logs for a specific service:
  ```bash
  docker-compose logs [service_name]
  ```

  Restart all services:
  ```bash
  docker-compose down
  docker-compose up -d
  ```

### GPU Not Detected

  Verify NVIDIA driver:
  ```bash
  nvidia-smi
  ```

  Verify Docker GPU support:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
  ```

### Database Connection Issues

  Reset databases:
  ```bash
  docker-compose down -v
  python setup_apexai.py
  ```

## License

This project is developed for the Hack the Track competition.

## Contributors

Qooniee
