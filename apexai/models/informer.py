"""Informer model with attention mechanism for time-series classification."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# NOTE: Seed setting removed from module level
# Seed is managed centrally in entrypoint and train_model() functions
# Module-level seed setting can cause inconsistent behavior between HPO and Trainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer with multi-head self-attention.

    Args:
        input_size: Size of input features (unused, kept for compatibility).
        hidden_size: Dimension of the model's hidden representations.
        num_heads: Number of attention heads.
        feedforward_dim: Dimension of the feedforward network.
        dropout: Dropout probability.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        """Initialize the encoder layer."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(hidden_size, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder layer.

        Args:
            src: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).

        """
        # Pre-norm self-attention with batch_first=True
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)

        # Pre-norm feedforward
        src2 = self.norm2(src)
        src2 = F.relu(self.linear1(src2))
        src2 = self.linear2(src2)
        output: torch.Tensor = src + self.dropout2(src2)
        return output


class Encoder(nn.Module):
    """Transformer encoder with multiple encoder layers.

    Args:
        input_size: Size of input features.
        hidden_size: Dimension of the model's hidden representations.
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads in each layer.
        feedforward_dim: Dimension of the feedforward network.
        dropout: Dropout probability.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        """Initialize the encoder."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, hidden_size, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.input_projection = nn.Linear(input_size, hidden_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            src: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size).

        """
        src = self.input_projection(src)
        for layer in self.layers:
            src = layer(src)
        return src


class Informer(nn.Module):
    """Informer model for time-series classification.

    This model implements an efficient Transformer-based architecture
    with self-attention mechanism for sequence classification.

    Args:
        feature_size: Number of expected features in the input.
        hidden_dim: Dimension of the model's hidden representations.
            Variable name is unified across models (GRU, LSTM, Transformer, Informer)
            to enable Neural Architecture Search (NAS).
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads.
        feedforward_multiplier: Multiplier for feedforward network dimension.
            The actual feedforward dimension is calculated as:
            `feedforward_dim = hidden_dim * feedforward_multiplier`.
            Default is 4, following the original Transformer paper.
            This can be optimized via NAS.
        out_dim: Number of output classes.
        dropout_ratio: Dropout probability.
        classification: Whether this is a classification task.

    Note:
        For NAS compatibility, parameter names are standardized across models:
        - `hidden_dim` = d_model (Transformer) = hidden_dim (RNN)
        - `feedforward_multiplier` allows NAS to explore different ratios

    """

    def __init__(
        self,
        feature_size: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_multiplier: int = 4,
        out_dim: int = 5,
        dropout_ratio: float = 0.1,
        classification: bool = True,
    ) -> None:
        """Initialize the Informer model."""
        super().__init__()
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_size = feature_size
        self.hidden_layer_size = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_multiplier = feedforward_multiplier
        self.out_dim = out_dim
        self.dropout_ratio = dropout_ratio
        self.classification = classification

        # Calculate feedforward dimension
        feedforward_dim = hidden_dim * feedforward_multiplier

        # Encoder with self-attention
        self.encoder = Encoder(
            feature_size,
            hidden_dim,
            num_layers,
            num_heads,
            feedforward_dim,
            dropout_ratio,
        )

        # Output fully connected layer
        self.fc = nn.Linear(hidden_dim, out_dim)
        # Note: No softmax layer - CrossEntropyLoss applies it internally

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Informer network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_size).

        Returns:
            Output logits of shape (batch_size, out_dim).

        Note:
            Returns logits (not probabilities). Use torch.softmax for inference.

        """
        # Encoder forward pass
        x = self.encoder(x)

        # Use the last time step's output
        out = x[:, -1, :]

        # Fully connected layer returns logits
        logits: torch.Tensor = self.fc(out)
        return logits


def main() -> None:
    """Test the Informer model with sample data and export to ONNX format."""
    import onnx
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

    # Model configuration
    batch_size = 8
    num_layers = 2
    input_size = 5
    hidden_size = 24
    num_heads = 4
    feedforward_multiplier = 4  # feedforward_dim = hidden_size * feedforward_multiplier
    seq_len = 100
    output_dim = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Informer(
        feature_size=input_size,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_multiplier=feedforward_multiplier,
        out_dim=output_dim,
        dropout_ratio=0.1,
        classification=True,
    )

    # Generate random test data
    data = torch.rand(batch_size, seq_len, input_size).to(device)
    model.to(device)

    # Training loop test
    for _ in range(1):
        output = model(data)
    logger.info("Model architecture:\n%s", model)

    # Inference loop test
    model.to("cpu")
    model.eval()
    for _ in range(1):
        output = model(data.to("cpu"))
    logger.info("Model output: %s", output)

    # Get predictions
    pred_labels = output.argmax(dim=1).tolist()
    logger.info("Predicted labels: %s", pred_labels)

    # Export to TorchScript
    traced_script_module = torch.jit.trace(model.eval(), data.to("cpu"))
    traced_script_module.save("Informer_test.pt")
    logger.info("Model saved to Informer_test.pt")

    # Export to ONNX
    torch.onnx.export(
        model.eval(),
        (data.to("cpu"),),
        "Informer_test.onnx",
        verbose=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and verify ONNX model
    informer_onnx_model = onnx.load("Informer_test.onnx")
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info("ONNX model structure:")
    logger.info(onnx.helper.printable_graph(informer_onnx_model.graph))

    # Test ONNX inference
    session = InferenceSession(
        "Informer_test.onnx",
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    result = session.run(None, {"input": np.array(data.to("cpu"))})
    logger.info("ONNX inference result: %s", result)


if __name__ == "__main__":
    main()
