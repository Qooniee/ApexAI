"""Transformer model for time-series classification."""

import logging

import numpy as np
import torch
from torch import nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerModel(nn.Module):
    """Transformer model for sequence classification.

    This model implements a Transformer encoder-decoder architecture
    for time-series classification tasks.

    Args:
        feature_size: Number of expected features in the input.
        hidden_dim: Dimension of the model's hidden representations.
            This corresponds to `d_model` in the original Transformer paper.
            Variable name is unified across models (GRU, LSTM, Transformer, Informer)
            to enable Neural Architecture Search (NAS).
        num_layers: Number of encoder and decoder layers.
        num_heads: Number of attention heads in multi-head attention.
        out_dim: Number of output classes.
        feedforward_multiplier: Multiplier for feedforward network dimension.
            The actual feedforward dimension is calculated as:
            `dim_feedforward = hidden_dim * feedforward_multiplier`.
            Default is 4, following the original Transformer paper
            (d_model=512, d_ff=2048). This can be optimized via NAS.
        dropout_ratio: Dropout probability applied throughout the model.
        classification: Whether this is a classification task.

    Note:
        For NAS compatibility, parameter names are standardized across models:
        - `hidden_dim` = d_model (Transformer) = hidden_dim (RNN)
        - `feedforward_multiplier` allows NAS to explore different ratios

    """

    def __init__(
        self,
        feature_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        out_dim: int,
        feedforward_multiplier: int = 4,
        dropout_ratio: float = 0.1,
        classification: bool = True,
    ) -> None:
        """Initialize the Transformer model."""
        super().__init__()
        self.classification = classification
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.feedforward_multiplier = feedforward_multiplier
        self.dropout_ratio = dropout_ratio

        # Calculate feedforward dimension
        dim_feedforward = hidden_dim * feedforward_multiplier

        # Input embedding layer
        self.embedding = nn.Linear(feature_size, hidden_dim)

        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_ratio,
            batch_first=True,
        )

        # Output fully connected layer
        self.fc = nn.Linear(hidden_dim, out_dim)
        # Note: No softmax layer - CrossEntropyLoss applies it internally

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_size).

        Returns:
            Output logits of shape (batch_size, out_dim).

        Note:
            Returns logits (not probabilities). Use torch.softmax for inference.

        """
        # Project input to hidden dimension
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)

        # Transformer forward pass (encoder-decoder) with batch_first=True
        out = self.transformer(x, x)  # (batch, seq_len, hidden_dim)

        # Average pooling over time steps
        out = out.mean(dim=1)  # (batch, hidden_dim)

        # Fully connected layer returns logits
        logits: torch.Tensor = self.fc(out)
        return logits


def main() -> None:
    """Test the Transformer model with sample data and export to ONNX format."""
    import onnx
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

    # Model configuration
    batch_size = 8
    num_layers = 2
    input_size = 5
    num_heads = 2
    hidden_dim = 24
    seq_len = 100
    out_dim = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TransformerModel(
        feature_size=input_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        out_dim=out_dim,
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
        output = model(data.to("cpu"))[-1]
    logger.info("Model output: %s", output)

    # Get prediction
    pred_label = output.argmax().tolist()
    logger.info("Predicted label: %d", pred_label)

    # Export to TorchScript
    traced_script_module = torch.jit.trace(model.eval(), data.to("cpu"))
    traced_script_module.save("Transformer_test.pt")
    logger.info("Model saved to Transformer_test.pt")

    # Export to ONNX
    torch.onnx.export(
        model.eval(),
        (data.to("cpu"),),
        "Transformer_test.onnx",
        verbose=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and verify ONNX model
    transformer_onnx_model = onnx.load("Transformer_test.onnx")
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info("ONNX model structure:")
    logger.info(onnx.helper.printable_graph(transformer_onnx_model.graph))

    # Test ONNX inference
    session = InferenceSession(
        "Transformer_test.onnx",
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    result = session.run(None, {"input": np.array(data.to("cpu"))})
    logger.info("ONNX inference result: %s", result)


if __name__ == "__main__":
    main()
