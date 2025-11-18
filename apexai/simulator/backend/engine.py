"""ApexAI Simulator - Backend Engine.

Real-time inference simulator backend engine.
"""

import inspect
import os
import sys
import types
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from apexai.models.gru import GRUwithFC
from apexai.models.informer import Informer
from apexai.models.lstm import LSTMwithFC
from apexai.models.transformer import TransformerModel


class DataStreamer:
    """CSV data loader with real-time streaming capability."""

    def __init__(self, csv_path: str, sampling_rate: int = 10, feature_columns: list | None = None):
        """Initialize DataStreamer.

        Args:
            csv_path: Path to CSV file containing sensor data.
            sampling_rate: Sampling rate in Hz (default: 10).
            feature_columns: List of feature column names to use (default: None uses all columns).
        """
        self.csv_path = Path(csv_path)
        self.sampling_rate = sampling_rate
        self.interval = 1.0 / sampling_rate
        self.feature_columns = feature_columns

        # Load CSV data
        self.data = pd.read_csv(csv_path)

        # If feature columns are specified, use only those columns
        if feature_columns is not None:
            # Check if specified columns exist
            missing_cols = [col for col in feature_columns if col not in self.data.columns]
            if missing_cols:
                print(f"[DataStreamer] Warning: Missing columns: {missing_cols}")
            # Select only existing columns
            available_cols = [col for col in feature_columns if col in self.data.columns]
            self.data = self.data[available_cols]
            print(f"[DataStreamer] Using feature columns: {available_cols}")
            print(f"[DataStreamer] DEBUG - First sample from CSV: {self.data.iloc[0].values}")

        self.total_samples = len(self.data)
        self.current_index = 0

        print(f"[DataStreamer] Loaded {self.total_samples} samples from {csv_path}")
        print(f"[DataStreamer] Sampling rate: {sampling_rate} Hz")

    def get_next_sample(self) -> np.ndarray | None:
        """Get the next sample from the data stream.

        Returns:
            Next sample as numpy array, or None if stream is exhausted.
        """
        if self.current_index >= self.total_samples:
            return None

        sample = self.data.iloc[self.current_index].values
        self.current_index += 1
        return sample

    def reset(self):
        """Reset the data stream to the beginning."""
        self.current_index = 0


class SlidingWindowBuffer:
    """Sliding window buffer for sequence-based inference."""

    def __init__(self, seq_len: int = 100, feature_size: int = 9):
        """Initialize SlidingWindowBuffer.

        Args:
            seq_len: Sequence length for inference window (default: 100).
            feature_size: Number of features per sample (default: 9).
        """
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.buffer: deque[np.ndarray] = deque(maxlen=seq_len)

    def add_sample(self, sample: np.ndarray):
        """Add a sample to the buffer.

        Args:
            sample: Sample data as numpy array.
        """
        self.buffer.append(sample)

    def is_ready(self) -> bool:
        """Check if buffer has enough samples for inference.

        Returns:
            True if buffer is full and ready for inference.
        """
        return len(self.buffer) == self.seq_len

    def get_sequence(self) -> np.ndarray:
        """Get the current sequence from the buffer.

        Returns:
            Current sequence as numpy array of shape (seq_len, feature_size).
        """
        return np.array(list(self.buffer))

    def clear(self):
        """Clear all samples from the buffer."""
        self.buffer.clear()


class InferenceEngine:
    """Inference engine for real-time model predictions."""

    def __init__(self, model_path: str, config_path: str, device: str = "cuda"):
        """Initialize InferenceEngine.

        Args:
            model_path: Path to trained PyTorch model file.
            config_path: Path to model configuration YAML file.
            device: Device to use for inference (default: "cuda").
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[InferenceEngine] Using device: {self.device}")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._load_normalization_params()
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"[InferenceEngine] Model loaded: {self.config['type']}")

    @property
    def feature_columns(self) -> list[str]:
        """Get the list of feature column names."""
        return list(self.config["preprocessing"]["features"])

    @property
    def seq_len(self) -> int:
        """Get the sequence length for inference."""
        return int(self.config["architecture"]["seq_len"])

    @property
    def feature_size(self) -> int:
        """Get the number of features per sample."""
        return int(self.config["architecture"]["feature_size"])

    @property
    def sampling_rate(self) -> int:
        """Get the sampling rate in Hz."""
        if "sampling_rate" not in self.config:
            raise ValueError(
                "sampling_rate is not defined in config. "
                "Please add 'sampling_rate: <value>' to your model config YAML."
            )
        return int(self.config["sampling_rate"])

    @property
    def num_votes(self) -> int:
        """Get the number of votes for Test Time Augmentation."""
        if "num_votes" not in self.config:
            raise ValueError(
                "num_votes is not defined in config. "
                "Please add 'num_votes: <value>' to your model config YAML."
            )
        return int(self.config["num_votes"])

    @property
    def class_names(self) -> dict[int, str]:
        """Get the class ID to class name mapping."""
        if "class_names" not in self.config:
            raise ValueError(
                "class_names is not defined in config. "
                "Please add 'class_names: <mapping>' to your model config YAML."
            )
        return dict(self.config["class_names"])

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        return str(self.class_names.get(class_id, f"Unknown-{class_id}"))

    def _load_normalization_params(self):
        """Load Min-Max normalization parameters."""
        norm_config = self.config["preprocessing"]["normalization"]

        min_vals = np.array(norm_config["min_max"]["min"], dtype=np.float32)
        max_vals = np.array(norm_config["min_max"]["max"], dtype=np.float32)

        self.min_vals = torch.tensor(min_vals, dtype=torch.float32, device=self.device)
        self.max_vals = torch.tensor(max_vals, dtype=torch.float32, device=self.device)

        print(f"[InferenceEngine] Normalization: {norm_config['method']}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load PyTorch model (architecture + weights)."""
        model_path_obj = Path(model_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Step 1: Instantiate model architecture
        model_type = self.config["type"]
        arch_config = self.config["architecture"]

        print(f"[InferenceEngine] Creating {model_type} architecture...")

        # Instantiate with correct arguments based on model type
        if model_type == "Transformer":
            model = TransformerModel(
                feature_size=arch_config["feature_size"],
                hidden_dim=arch_config["hidden_dim"],
                num_layers=arch_config["num_layers"],
                num_heads=arch_config["num_heads"],
                out_dim=arch_config["out_dim"],
                feedforward_multiplier=arch_config["feedforward_multiplier"],
                dropout_ratio=arch_config["dropout_ratio"],
                classification=arch_config["classification"],
            )
        elif model_type == "Informer":
            model = Informer(
                feature_size=arch_config["feature_size"],
                hidden_dim=arch_config["hidden_dim"],
                num_layers=arch_config["num_layers"],
                num_heads=arch_config["num_heads"],
                feedforward_multiplier=arch_config["feedforward_multiplier"],
                out_dim=arch_config["out_dim"],
                dropout_ratio=arch_config["dropout_ratio"],
                classification=arch_config["classification"],
            )
        elif model_type == "GRU":
            model = GRUwithFC(
                feature_size=arch_config["feature_size"],
                hidden_dim=arch_config["hidden_dim"],
                num_layers=arch_config["num_layers"],
                out_dim=arch_config["out_dim"],
                dropout_ratio=arch_config["dropout_ratio"],
                classification=arch_config["classification"],
                batch_first=arch_config.get("batch_first", True),
            )
        elif model_type == "LSTM":
            model = LSTMwithFC(
                feature_size=arch_config["feature_size"],
                hidden_dim=arch_config["hidden_dim"],
                num_layers=arch_config["num_layers"],
                out_dim=arch_config["out_dim"],
                dropout_ratio=arch_config["dropout_ratio"],
                classification=arch_config["classification"],
                batch_first=arch_config.get("batch_first", True),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Step 2: Load weights
        print(f"[InferenceEngine] Loading weights from {model_path}...")

        # Alias legacy module paths for old checkpoints
        try:
            import apexai.models.gru as m_gru
            import apexai.models.informer as m_informer
            import apexai.models.lstm as m_lstm
            import apexai.models.transformer as m_transformer

            sys.modules.setdefault("models", types.ModuleType("models"))
            sys.modules["models.transformer"] = m_transformer
            sys.modules["models.informer"] = m_informer
            sys.modules["models.gru"] = m_gru
            sys.modules["models.lstm"] = m_lstm

            sys.modules["models"].transformer = m_transformer  # type: ignore[attr-defined]
            sys.modules["models"].informer = m_informer  # type: ignore[attr-defined]
            sys.modules["models"].gru = m_gru  # type: ignore[attr-defined]
            sys.modules["models"].lstm = m_lstm  # type: ignore[attr-defined]
        except Exception:
            pass

        # Try safe load first (allow necessary classes broadly)
        allowed = {TransformerModel, Informer, GRUwithFC, LSTMwithFC}
        try:
            modules = []
            for modname in [
                "apexai.models.transformer",
                "apexai.models.informer",
                "apexai.models.gru",
                "apexai.models.lstm",
            ]:
                try:
                    modules.append(__import__(modname, fromlist=["*"]))
                except Exception:
                    continue
            for mod in modules:
                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    # Allow classes defined in that module
                    if getattr(cls, "__module__", "") == mod.__name__:
                        allowed.add(cls)

            with torch.serialization.safe_globals(list(allowed)):
                checkpoint = torch.load(model_path_obj, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Safe load failed detail: {e!r}")
            allow_unsafe = os.getenv("APEXAI_ALLOW_UNSAFE_LOAD", "0") == "1"
            if allow_unsafe:
                print(
                    "⚠️ Safe load failed, falling back to weights_only=False. "
                    "Only do this if the file is trusted."
                )
                checkpoint = torch.load(model_path_obj, map_location="cpu", weights_only=False)
            else:
                raise RuntimeError(
                    "Safe load failed. If you trust this checkpoint, set "
                    "APEXAI_ALLOW_UNSAFE_LOAD=1, or re-save it as a pure state_dict."
                ) from e

        # Load according to checkpoint format
        def _load_sd(target_model: torch.nn.Module, sd: dict):
            # Support for DataParallel/DistributedDataParallel
            if any(k.startswith("module.") for k in sd):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            res = target_model.load_state_dict(sd, strict=False)
            try:
                missing = getattr(res, "missing_keys", [])
                unexpected = getattr(res, "unexpected_keys", [])
                if missing or unexpected:
                    print(
                        f"ℹ️ load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}"
                    )
            except Exception:
                pass

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
            print("✅ Loaded model object directly")
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            _load_sd(model, checkpoint["model_state_dict"])
            print(f"✅ Loaded from checkpoint (epoch: {checkpoint.get('epoch', 'N/A')})")
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            _load_sd(model, checkpoint["state_dict"])
            print("✅ Loaded state_dict (key: state_dict)")
        elif isinstance(checkpoint, dict):
            # Treat as pure state_dict
            _load_sd(model, checkpoint)
            print("✅ Loaded state_dict")
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

        model.to(self.device)
        print("[InferenceEngine] Model loaded successfully!")
        return model

    def normalize(self, sequence: np.ndarray) -> torch.Tensor:
        """Apply Min-Max normalization."""
        # Explicitly convert NumPy array to float32 (workaround for object dtype)
        sequence = np.asarray(sequence, dtype=np.float32)
        seq_tensor = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        normalized = (seq_tensor - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized

    def predict(self, sequence: np.ndarray) -> tuple[int, np.ndarray]:
        """Execute inference on a sequence."""
        normalized_seq = self.normalize(sequence)
        input_tensor = normalized_seq.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        probabilities_np = probabilities.cpu().numpy().squeeze()

        return predicted_class, probabilities_np


class TestTimeAugmentation:
    """Test Time Augmentation with majority voting and soft voting support."""

    def __init__(self, window_size: int = 6):
        """Initialize TestTimeAugmentation.

        Args:
            window_size: Number of predictions to collect for voting (default: 6).
        """
        self.window_size = window_size
        self.inference_queue: deque[int] = deque(maxlen=window_size)
        self.probability_queue: deque[np.ndarray] = deque(maxlen=window_size)

    def add_prediction(self, predicted_class: int, probabilities: np.ndarray | None = None):
        """Add a prediction result to the voting queue.

        Args:
            predicted_class: Predicted class index.
            probabilities: Optional probability distribution for soft voting.
        """
        self.inference_queue.append(predicted_class)
        if probabilities is not None:
            self.probability_queue.append(probabilities)

    def is_ready(self) -> bool:
        """Check if TTA has enough predictions for voting."""
        return len(self.inference_queue) == self.window_size

    def get_majority_vote(self) -> tuple[int, dict]:
        """Get the majority vote result (hard voting)."""
        counter = Counter(self.inference_queue)
        majority_class = counter.most_common(1)[0][0]
        vote_counts = dict(counter)
        return majority_class, vote_counts

    def get_average_probabilities(self) -> np.ndarray:
        """Get average probabilities across all predictions (for soft voting)."""
        if len(self.probability_queue) == 0:
            raise ValueError("No probabilities stored in TTA queue")
        return np.mean(list(self.probability_queue), axis=0)

    def reset(self):
        """Reset the voting queue."""
        self.inference_queue.clear()
        self.probability_queue.clear()


class MultiModelTTA:
    """Multi-model Test Time Augmentation with ensemble voting.

    Combines predictions from multiple models (e.g., LSTM + Transformer)
    using TTA for each model, then performs ensemble voting across all predictions.
    """

    def __init__(
        self,
        model_configs: list[dict[str, str]],
        device: str = "cuda",
        tta_window_size: int = 6,
        ensemble_method: str = "majority",
    ):
        """Initialize MultiModelTTA.

        Args:
            model_configs: List of model configuration dictionaries.
                Each dict should have 'model_path' and 'config_path' keys.
                Example: [
                    {'model_path': 'lstm.pth', 'config_path': 'lstm.yaml',
                     'name': 'LSTM'},
                    {'model_path': 'transformer.pth',
                     'config_path': 'transformer.yaml', 'name': 'Transformer'}
                ]
            device: Device to use for inference (default: "cuda").
            tta_window_size: Number of predictions per model for TTA (default: 6).
            ensemble_method: Method for combining predictions (default: "majority").
                - "majority": Simple majority voting across all predictions
                - "weighted": Confidence-weighted ensemble (probabilities)
        """
        self.device = device
        self.tta_window_size = tta_window_size
        self.ensemble_method = ensemble_method

        # Initialize inference engines for each model
        self.engines: list[InferenceEngine] = []
        self.model_names: list[str] = []
        self.ttas: list[TestTimeAugmentation] = []

        for config in model_configs:
            engine = InferenceEngine(config["model_path"], config["config_path"], device)
            tta = TestTimeAugmentation(tta_window_size)

            self.engines.append(engine)
            self.model_names.append(config.get("name", f"Model{len(self.engines)}"))
            self.ttas.append(tta)

        print(f"[MultiModelTTA] Initialized with {len(self.engines)} models:")
        for name in self.model_names:
            print(f"  - {name}")
        print(f"[MultiModelTTA] TTA window size: {tta_window_size} per model")
        print(f"[MultiModelTTA] Total votes: {tta_window_size * len(self.engines)}")
        print(f"[MultiModelTTA] Ensemble method: {ensemble_method}")

    def predict(self, sequence: np.ndarray) -> tuple[list[int], list[np.ndarray], list[str]]:
        """Execute inference on all models.

        Args:
            sequence: Input sequence of shape (seq_len, features).

        Returns:
            Tuple of (predictions, probabilities, model_names) for all models.
        """
        predictions = []
        probabilities = []

        for engine, _name in zip(self.engines, self.model_names, strict=False):
            pred, probs = engine.predict(sequence)
            predictions.append(pred)
            probabilities.append(probs)

            # Add to TTA buffer with probabilities for soft voting
            tta_idx = self.engines.index(engine)
            self.ttas[tta_idx].add_prediction(pred, probs)

        return predictions, probabilities, self.model_names

    def is_ready(self) -> bool:
        """Check if all TTAs have enough predictions for voting."""
        return all(tta.is_ready() for tta in self.ttas)

    def get_ensemble_result(
        self,
    ) -> tuple[int, dict[str, Any]]:
        """Get ensemble result from all models' TTA buffers.

        Returns:
            Tuple of (final_prediction, metadata) where metadata contains:
                - vote_counts: Vote distribution across all predictions
                - per_model_votes: Vote counts for each model
                - agreement_rate: Percentage of models agreeing with final prediction
                - ensemble_method: Method used for ensemble
        """
        if self.ensemble_method == "majority":
            return self._majority_vote_ensemble()
        elif self.ensemble_method == "weighted":
            return self._weighted_ensemble()
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _majority_vote_ensemble(self) -> tuple[int, dict[str, Any]]:
        """Simple majority voting across all model predictions."""
        # Collect all predictions from all TTAs
        all_predictions = []
        per_model_votes = {}

        for tta, name in zip(self.ttas, self.model_names, strict=False):
            majority_class, vote_counts = tta.get_majority_vote()
            per_model_votes[name] = {
                "majority": majority_class,
                "votes": vote_counts,
            }
            # Add all predictions from this model's TTA
            all_predictions.extend(list(tta.inference_queue))

        # Count votes across all predictions
        counter = Counter(all_predictions)
        final_prediction = counter.most_common(1)[0][0]
        vote_counts = dict(counter)

        # Calculate agreement rate
        model_agreements = sum(
            1
            for model_vote in per_model_votes.values()
            if model_vote["majority"] == final_prediction
        )
        agreement_rate = model_agreements / len(self.engines)

        metadata = {
            "vote_counts": vote_counts,
            "per_model_votes": per_model_votes,
            "agreement_rate": agreement_rate,
            "ensemble_method": "majority",
            "total_votes": len(all_predictions),
        }

        return final_prediction, metadata

    def _weighted_ensemble(self) -> tuple[int, dict[str, Any]]:
        """Confidence-weighted ensemble using prediction probabilities (soft voting).

        Averages probability distributions from each model's TTA,
        then averages across models to get final prediction.
        """
        # Collect average probabilities from each model's TTA
        model_avg_probs = []
        per_model_info = {}

        for tta, name in zip(self.ttas, self.model_names, strict=False):
            if not tta.is_ready():
                raise ValueError(f"TTA for {name} is not ready")

            # Get average probabilities across TTA window
            avg_probs = tta.get_average_probabilities()
            model_avg_probs.append(avg_probs)

            # Get majority vote for comparison
            majority_class, _ = tta.get_majority_vote()

            per_model_info[name] = {
                "avg_probabilities": avg_probs,
                "predicted_class": int(np.argmax(avg_probs)),
                "confidence": float(np.max(avg_probs)),
                "majority_vote": majority_class,
            }

        # Average probabilities across all models (soft voting)
        ensemble_probs = np.mean(model_avg_probs, axis=0)
        final_prediction = int(np.argmax(ensemble_probs))
        final_confidence = float(np.max(ensemble_probs))

        # Calculate agreement rate
        model_predictions = [info["predicted_class"] for info in per_model_info.values()]
        model_agreements = sum(1 for pred in model_predictions if pred == final_prediction)
        agreement_rate = model_agreements / len(self.engines)

        metadata = {
            "ensemble_probabilities": ensemble_probs,
            "final_confidence": final_confidence,
            "per_model_info": per_model_info,
            "agreement_rate": agreement_rate,
            "ensemble_method": "weighted",
        }

        return final_prediction, metadata

    def reset(self):
        """Reset all TTA buffers."""
        for tta in self.ttas:
            tta.reset()

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID (using first engine's mapping)."""
        return self.engines[0].get_class_name(class_id)
