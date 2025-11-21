# ApexAI: Quick Start Guide

This guide provides the minimal steps to launch the platform, run a training demo, and verify the real-time inference simulator.

---

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Visual Studio Code** with **Dev Containers extension** installed
- **NVIDIA GPU** with CUDA support (recommended)
- **16GB+ RAM** and **10GB free disk space**
- **Python 3.11+** (for initial setup script)

---

## Step 1: Initial Setup

**On your host machine:**

```bash
# Clone repository
git clone https://github.com/Qooniee/ApexAI.git
cd ApexAI

# Copy environment template
cp .env.template .env

# Run automated setup (builds containers, initializes databases)
python setup_apexai.py
```

**Expected output:**
```
✓ Docker images built successfully
✓ PostgreSQL databases initialized
✓ MLflow tracking server ready
✓ Optuna dashboard ready
✓ MinIO storage configured
✓ All services healthy
```

---

## Step 2: Open in VS Code Dev Container

1. Open VS Code
2. Open the `ApexAI` folder: **File → Open Folder**
3. VS Code will detect the dev container configuration
4. Click **"Reopen in Container"** (bottom-right notification)
   - Or: Press `F1` → **"Dev Containers: Reopen in Container"**

**Wait for the container to start.** VS Code terminal is now inside the container.

---

## Step 3: Download Dataset

**On your host machine** (outside VS Code):

1. Visit: https://trddev.com/hackathon-2025/
2. Download: `virginia-international-raceway.zip`
3. Extract the ZIP file
4. Create datasets folder (inside Docker container)

```bash
mkdir -p /workspace/datasets/rawdata
```

5. Copy the `VIR` folder to: `<ApexAI-project-root>/datasets/rawdata/VIR`

**Verify in VS Code terminal** (inside container):

```bash
/workspace# ls datasets/rawdata/VIR
# Should show: Race 1, Race 2
```

---

## Step 4: Generate Training Data

**In VS Code terminal** (all commands below are inside the container):

```bash
/workspace# python -m apexai.data_generation.pipeline \
  --raw-data-dir datasets/rawdata/VIR \
  --export-dir datasets/drivingdatasets/input
```

**Expected output:**
```
================================================================================
APEXAI DATA GENERATION PIPELINE
================================================================================
Raw data directory:      datasets/rawdata/VIR
Preprocessed directory:  datasets/preprocessed_10Hz/VIR
Training dataset dir:    datasets/training_dataset_10Hz
Final export directory:  datasets/drivingdatasets/input
Target frequency:        10.0 Hz
Split ratios:            70% / 15% / 15%
Filter last lap:         Yes
Shuffle laps:            Yes
Random seed:             42
================================================================================

[STEP 1/3] Generating 10.0Hz resampled lap data...
  Input:  datasets/rawdata/VIR
  Output: datasets/preprocessed_10Hz/VIR

  Processing Race 1...
Processing laps: 100%|████████████████████████████████████████| 417/417 [00:23<00:00, 17.69it/s]
    ✓ Generated 417 lap files
  Processing Race 2...
Processing laps: 100%|████████████████████████████████████████| 440/440 [00:23<00:00, 18.80it/s]
    ✓ Generated 440 lap files

  ✓ Step 1 complete: 857 laps, 857 files


[STEP 2/3] Generating X/y training files...
  Input:  datasets/preprocessed_10Hz/VIR
  Output: datasets/training_dataset_10Hz

  Processing R1...
Processing laps: 100%|████████████████████████████████████████| 417/417 [00:18<00:00, 23.05it/s]
    ✓ Generated 417 X/y file pairs
  Processing R2...
Processing laps: 100%|████████████████████████████████████████| 440/440 [00:18<00:00, 24.10it/s]
    ✓ Generated 440 X/y file pairs

  ✓ Step 2 complete: 857 X/y file pairs


[STEP 3/3] Splitting into train/valid/test datasets...
  Input:  datasets/training_dataset_10Hz
  Output: datasets/drivingdatasets/input

  Found 21 vehicles

Vehicle GR86_002_2: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_006_7: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_013_80: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_015_31: 23 laps
  Train: 16, Valid: 3, Test: 4

Vehicle GR86_016_55: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_022_13: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_024_41: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_026_72: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_028_89: 25 laps
  Train: 17, Valid: 3, Test: 5

Vehicle GR86_032_15: 35 laps
  Train: 24, Valid: 5, Test: 6

Vehicle GR86_033_46: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_036_98: 28 laps
  Train: 19, Valid: 4, Test: 5

Vehicle GR86_038_93: 43 laps
  Train: 30, Valid: 6, Test: 7

Vehicle GR86_040_3: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_047_21: 23 laps
  Train: 16, Valid: 3, Test: 4

Vehicle GR86_049_88: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_051_71: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_061_51: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_062_012: 43 laps
  Train: 30, Valid: 6, Test: 7

Vehicle GR86_063_113: 44 laps
  Train: 30, Valid: 6, Test: 8

Vehicle GR86_065_5: 44 laps
  Train: 30, Valid: 6, Test: 8

  ✓ Step 3 complete:
    - Train: 572 files (21 vehicles)
    - Valid: 114 files (21 vehicles)
    - Test:  150 files (21 vehicles)

  Pipeline statistics saved to: datasets/drivingdatasets/input/pipeline_stats.json

================================================================================
PIPELINE COMPLETE!
================================================================================
Final dataset ready at: datasets/drivingdatasets/input
```

---

## Step 5a: Quick Training Demo

**In VS Code terminal:**

```bash
/workspace# python -m apexai.trainer_entrypoint \
  model=gru \
  training.epochs=5 \
  training.batch_size=128
```

**Expected output:**
```
/workspace/apexai/trainer_entrypoint.py:23: UserWarning:
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_path="../conf", config_name="config")
[2025-11-20 23:53:32,292][__main__][INFO] - >>>>> Starting Main Process <<<<<
[2025-11-20 23:53:32,647][__main__][INFO] - Using device: cuda:0
[2025-11-20 23:53:33,342][__main__][INFO] - Creating dataset file list...
[2025-11-20 23:53:33,387][__main__][INFO] - --- Dataset Information ---
[2025-11-20 23:53:33,387][__main__][INFO] - Number of x_train dataset: 572
[2025-11-20 23:53:33,388][__main__][INFO] - Number of y_train dataset: 572
[2025-11-20 23:53:33,388][__main__][INFO] - Number of x_valid dataset: 114
[2025-11-20 23:53:33,388][__main__][INFO] - Number of y_valid dataset: 114
[2025-11-20 23:53:33,389][__main__][INFO] - Number of x_test dataset: 150
[2025-11-20 23:53:33,389][__main__][INFO] - Number of y_test dataset: 150
[2025-11-20 23:53:33,389][__main__][INFO] - ---------------------------
[2025-11-20 23:53:33,389][__main__][INFO] - Creating batch for training datasets
[2025-11-20 23:53:44,632][__main__][INFO] - Creating train loader...
[2025-11-20 23:53:44,652][__main__][INFO] - Creating batch for validation set...
[2025-11-20 23:53:44,652][__main__][INFO] -   (Using training data statistics for normalization)
[2025-11-20 23:53:46,306][__main__][INFO] - Creating validation loader...
[2025-11-20 23:53:46,309][__main__][INFO] - --- Data Shape Validation ---
[2025-11-20 23:53:46,310][__main__][INFO] - Actual data shape: seq_len=100, features=9
[2025-11-20 23:53:46,310][__main__][INFO] - Expected shape: seq_len=100, features=9
[2025-11-20 23:53:46,310][__main__][INFO] - ✅ Data shape matches expectations
[2025-11-20 23:53:46,311][__main__][INFO] - ✅ Using actual data shape: seq_len=100, features=9
[2025-11-20 23:53:46,311][__main__][INFO] - -----------------------------
[2025-11-20 23:53:46,312][apexai.models.model_factory][INFO] - Creating GRU model
[2025-11-20 23:53:46,324][__main__][INFO] - Model structure:
  GRUwithFC(
    (gru): GRU(9, 256, num_layers=4, batch_first=True, dropout=0.3)
    (fc): Linear(in_features=256, out_features=21, bias=True)
  )

  [2025-11-20 23:53:53,406] Epoch [1/5] Train Loss: 3.0951 | Val Acc: 0.0341 | Val F1: 0.0094
  [2025-11-20 23:53:57,007] Epoch [2/5] Train Loss: 3.0428 | Val Acc: 0.0418 | Val F1: 0.0164
  ...
  [2025-11-20 23:54:07,808] Epoch [5/5] Train Loss: 2.9904 | Val Acc: 0.0442 | Val F1: 0.0253
  ...

[2025-11-20 23:54:35,436][__main__][INFO] - >>>>> Training Complete <<<<<
[2025-11-20 23:54:35,437][__main__][INFO] - Model and metrics logged to MLflow run: 1772ed87e1024456b8f98f3de50a979e
🏃 View run bittersweet-pug-788 at: http://apexai_mlflow:5000/#/experiments/5/runs/1772ed87e1024456b8f98f3de50a979e
🧪 View experiment at: http://apexai_mlflow:5000/#/experiments/5
```

---

## Step 5b: Quick Hyperparameter Optimization Demo

**In VS Code terminal:**

```bash
/workspace# python -m apexai.optimization_entrypoint \
  model=gru \
  optuna=gru_hpo \
  optuna.optimization.n_trials=5 \
  training.epochs=5 \
  optuna.optimization.timeout=300 \
  optuna.study.name="ApexAI_HPO_Test_GRU"
```

**Expected output:**
```
[I 2025-11-21 00:15:23,456] A new study created in RDB with name: ApexAI_HPO_Test_GRU
[I 2025-11-21 00:15:30,123] Trial 0 finished with value: 0.0523 and parameters: {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2}
[I 2025-11-21 00:15:45,789] Trial 1 finished with value: 0.0687 and parameters: {'hidden_size': 256, 'num_layers': 4, 'dropout': 0.3}
[I 2025-11-21 00:16:01,234] Trial 2 finished with value: 0.0612 and parameters: {'hidden_size': 192, 'num_layers': 3, 'dropout': 0.25}
[I 2025-11-21 00:16:16,567] Trial 3 finished with value: 0.0698 and parameters: {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.15}
[I 2025-11-21 00:16:32,890] Trial 4 finished with value: 0.0734 and parameters: {'hidden_size': 320, 'num_layers': 4, 'dropout': 0.35}
[I 2025-11-21 00:16:33,000] Best trial: Trial 4 with value: 0.0734
```

**What happened:**
- Optuna automatically tested 5 different hyperparameter combinations
- Each trial trained a GRU model for 5 epochs
- Results are logged to both MLflow and Optuna database
- Best parameters are automatically identified

---

## Step 6: Access Dashboards

Open these URLs in your browser:

| Service | URL | Purpose |
|---------|-----|---------|
| **MLflow UI** | http://localhost:5001 | View training runs, metrics, and models |
| **Optuna Dashboard** | http://localhost:8081 | Hyperparameter optimization visualization |
| **MinIO Console** | http://localhost:9020 | Model artifact storage (login: Refer to the `.env` file.) |
| **Streamlit Simulator** | http://localhost:8501 | Real-time inference demo |

**MLflow UI Quick Check:**
1. Navigate to "Experiments"
2. Find your recent runs:
   - Single training run (Step 5a): model=gru, epochs=5
   - HPO trials (Step 5b): 5 optimization trials
3. Compare metrics across different trials
4. View the confusion matrix in "Artifacts"
5. View logged models in the Overview tab

**Optuna Dashboard Quick Check:**
1. Navigate to http://localhost:8081
2. Find study: "ApexAI_HPO_Test_GRU"
3. View optimization history (5 trials)
4. Check parameter importance chart
5. See best hyperparameter combination

---

## Step 7: Real-Time Inference Simulator

**In VS Code terminal:**

```bash
/workspace# streamlit run apexai/simulator/frontend/app.py
```

**Open in browser:** http://localhost:8501

**Demo Steps:**

1. **Select Mode:** Choose "Single Model" in sidebar
2. **Config Path:** `apexai/simulator/model_repository/prodmodel.yaml`
3. **Upload Test Data:**
   - Click "Browse files"
   - Navigate to: `datasets/drivingdatasets/input/test/x_test/`
   - Select any CSV file (e.g., `VIR_R1_vehicle_GR86-002-2_lap_001_x_test.csv`)
4. **Click:** "🚀 Initialize Simulator"
5. **Click:** "▶️ Start"

**What to observe:**
- Real-time sensor plots (steering, throttle, brake)
- Live vehicle ID predictions
- TTA voting mechanism with 6-vote majority (final prediction appears at ~60s)
- Vote counts and prediction confidence

---

## Step 8 (Optional): Advanced Features

**In VS Code terminal:**
### Neural Architecture Search

```bash
python -m apexai.optimization_entrypoint optuna=nas
```

---

## Verification Checklist

- [ ] All Docker services started (`python setup_apexai.py`)
- [ ] VS Code opened in Dev Container
- [ ] Dataset downloaded to `datasets/rawdata/VIR`
- [ ] Data pipeline generated train/valid/test splits
- [ ] Quick training completed (Step 5a: 5 epochs)
- [ ] HPO demo completed (Step 5b: 5 trials)
- [ ] MLflow UI shows training metrics at http://localhost:5001
- [ ] Optuna Dashboard shows HPO results at http://localhost:8081
- [ ] Streamlit simulator running at http://localhost:8501

---

## Cleanup

**In VS Code:**
1. Close VS Code (container will stop automatically)

**On your host machine:**

```bash
# Stop all services
docker-compose down

# Remove all data (optional - WARNING: deletes databases)
docker-compose down -v
```

---

## Key Features Demonstrated

✅ **AutoML Platform:** Automated data pipeline, training, and optimization
✅ **MLOps Integration:** MLflow tracking, Optuna optimization, MinIO storage
✅ **Real-Time Inference:** Streamlit-based simulator with TTA voting
✅ **Scalable Architecture:** Docker-based deployment ready for production
✅ **Multiple Models:** GRU, LSTM, Transformer, Informer support

---

## Next Steps

For detailed documentation, see:
- **Full README:** `README.md`
- **Architecture Details:** Sections 5-10 in README
- **Advanced Training:** Section 9 (HPO, NAS, BO-NAS)
- **Simulator Guide:** Section 10 (Multi-model ensemble, TTA)
