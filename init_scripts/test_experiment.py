#!/usr/bin/env python3
"""AxilDrive Setup Verification - Test Experiment.

Test experiment to verify completed setup.
"""

import os
import sys
import time

import mlflow
import mlflow.pytorch
import optuna
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn


def create_simple_model(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a simple MLP model."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Softmax(dim=1),
    )


def run_test_experiment():
    """Run test experiment."""
    print("🧪 Running test experiment...")

    # 1. Check environment variables
    required_vars = ["MLFLOW_TRACKING_URI", "OPTUNA_DB_URL"]
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ Environment variable {var} not set")
            return False

    # 2. Create dummy data
    print("📊 Creating synthetic test data...")
    features, labels = make_classification(
        n_samples=1000, n_features=10, n_classes=3, random_state=42
    )
    features_train, features_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # 3. Run MLflow experiment
    print("📈 Testing MLflow integration...")
    mlflow.set_experiment("ApexAI_Setup_Test")

    with mlflow.start_run(run_name="Setup_Verification_Test"):
        # Log parameters
        hidden_dim = 32
        learning_rate = 0.01
        epochs = 5

        mlflow.log_params(
            {
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "test_type": "setup_verification",
            }
        )

        # Create and train model
        model = create_simple_model(10, hidden_dim, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Simple training loop
        features_train_tensor = torch.FloatTensor(features_train)
        y_train_tensor = torch.LongTensor(y_train)
        features_test_tensor = torch.FloatTensor(features_test)
        y_test_tensor = torch.LongTensor(y_test)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(features_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Calculate test accuracy
            with torch.no_grad():
                test_outputs = model(features_test_tensor)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == y_test_tensor).float().mean().item()

            mlflow.log_metrics({"train_loss": loss.item(), "test_accuracy": accuracy}, step=epoch)

        # Save model
        mlflow.pytorch.log_model(model, "test_model")

        final_accuracy = accuracy
        print(f"  ✅ MLflow test completed. Final accuracy: {final_accuracy:.3f}")

    # 4. Test Optuna integration
    print("🎯 Testing Optuna integration...")

    try:
        # Connect to Optuna DB
        storage_url = os.getenv("OPTUNA_DB_URL")
        study = optuna.create_study(
            study_name="ApexAI_Setup_Study",
            storage=storage_url,
            direction="minimize",
            load_if_exists=True,
        )

        def objective(trial):
            hidden_dim = trial.suggest_int("hidden_dim", 16, 64)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            # Simple optimization test
            model = create_simple_model(10, hidden_dim, 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train for 1 epoch only
            optimizer.zero_grad()
            outputs = model(features_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            return loss.item()

        # Run simple optimization (3 trials)
        study.optimize(objective, n_trials=3)

        print(f"  ✅ Optuna test completed. Best value: {study.best_value:.3f}")

    except Exception as e:
        print(f"  ❌ Optuna test failed: {e}")
        return False

    print("🎉 All integration tests passed!")
    return True


def main():
    """Main execution."""
    print("🚀" + "=" * 58)
    print("   ApexAI Setup Verification Test")
    print("=" * 60)

    start_time = time.time()

    success = run_test_experiment()

    duration = time.time() - start_time

    if success:
        print(f"\n✅ Setup verification completed successfully in {duration:.1f}s")
        print("\n🌐 Check results at:")
        print("  📊 MLflow: http://localhost:5000")
        print("  🎯 Optuna: http://localhost:8080")
        return True
    print(f"\n❌ Setup verification failed after {duration:.1f}s")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
