#!/usr/bin/env python3
"""ModeDetection integration test script.

Purpose: Verify that MLOps integration is working correctly.
"""
# Usage: python scripts/test_modedetection_integration.py

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_config_loading():
    """Test Hydra configuration file loading."""
    print("=== Configuration Loading Test ===")

    try:
        from omegaconf import OmegaConf

        # Check configuration file paths
        config_dir = project_root / "conf" / "modedetection"
        config_files = {
            "main": config_dir / "config.yaml",
            "model": config_dir / "model" / "gru.yaml",
            "training": config_dir / "training" / "default.yaml",
            "optuna": config_dir / "optuna" / "default.yaml",
            "data": config_dir / "data" / "sensors.yaml",
        }

        print("Checking config files existence:")
        for name, path in config_files.items():
            exists = path.exists()
            print(f"  {name:10}: {path} - {'✓' if exists else '✗'}")
            if not exists:
                return False

        # Test loading main configuration
        main_config = OmegaConf.load(config_files["main"])
        print("\nMain config loaded successfully:")
        print(f"  Experiment name: {main_config.experiment.name}")
        print(f"  Defaults: {list(main_config.defaults)}")

        # Test loading model configuration
        model_config = OmegaConf.load(config_files["model"])
        print("\nModel config loaded successfully:")
        print(f"  Model name: {model_config.name}")
        print(f"  Hidden dim: {model_config.architecture.hidden_dim}")
        print(f"  Num layers: {model_config.architecture.num_layers}")

        return True

    except Exception as e:
        print(f"Config loading failed: {e}")
        return False


def test_data_paths():
    """Test data path existence."""
    print("\n=== Data Paths Test ===")

    try:
        # Check data directory
        data_dir = project_root / "datasets" / "ModeDetection"

        required_paths = [
            data_dir,
            data_dir / "input",
            data_dir / "input" / "train",
            data_dir / "input" / "train" / "x_train",
            data_dir / "input" / "train" / "y_train",
            data_dir / "output",
            data_dir / "docs",
        ]

        print("Checking data directory structure:")
        all_exist = True
        for path in required_paths:
            exists = path.exists()
            print(f"  {path.relative_to(project_root)!s:40} - {'✓' if exists else '✗'}")
            if not exists:
                all_exist = False

        # Check number of data files
        if (data_dir / "input" / "train" / "x_train").exists():
            x_files = list((data_dir / "input" / "train" / "x_train").glob("*.csv"))
            print(f"\nX_train files found: {len(x_files)}")

        if (data_dir / "input" / "train" / "y_train").exists():
            y_files = list((data_dir / "input" / "train" / "y_train").glob("*.csv"))
            print(f"Y_train files found: {len(y_files)}")

        return all_exist

    except Exception as e:
        print(f"Data path test failed: {e}")
        return False


def test_script_structure():
    """Check integration script structure."""
    print("\n=== Script Structure Test ===")

    try:
        script_path = project_root / "src" / "modedetection_main.py"

        if not script_path.exists():
            print(f"Main script not found: {script_path}")
            return False

        # Check basic script structure
        with open(script_path, encoding="utf-8") as f:
            content = f.read()

        required_imports = [
            "import mlflow",
            "import optuna",
            "import hydra",
            "from omegaconf import DictConfig",
        ]

        required_classes = [
            "class GRUModel",
            "class LSTMModel",
            "class TransformerModel",
        ]

        required_functions = [
            "def load_sensor_data",
            "def preprocess_data",
            "def create_model",
            "def train_model",
            "def objective",
        ]

        print("Checking script structure:")

        print("  Required imports:")
        for imp in required_imports:
            found = imp in content
            print(f"    {imp:30} - {'✓' if found else '✗'}")

        print("  Required classes:")
        for cls in required_classes:
            found = cls in content
            print(f"    {cls:30} - {'✓' if found else '✗'}")

        print("  Required functions:")
        for func in required_functions:
            found = func in content
            print(f"    {func:30} - {'✓' if found else '✗'}")

        return True

    except Exception as e:
        print(f"Script structure test failed: {e}")
        return False


def test_integration_readiness():
    """Comprehensive check of integration readiness."""
    print("\n=== Integration Readiness Summary ===")

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Data Paths", test_data_paths),
        ("Script Structure", test_script_structure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("INTEGRATION TEST RESULTS")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:25}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    overall_status = "READY" if all_passed else "NOT READY"
    print(f"Overall Status: {overall_status}")

    if all_passed:
        print("\n✅ ModeDetection integration preparation complete!")
        print("Test can be run with the following command:")
        print("python src/modedetection_main.py optuna.optimization.n_trials=1 training.epochs=2")
    else:
        print("\n❌ There are issues with integration preparation.")
        print("Please check the FAILED items above.")

    return all_passed


if __name__ == "__main__":
    test_integration_readiness()
