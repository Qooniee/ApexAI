"""GRU model with fully connected layer for time-series classification."""

import logging

import numpy as np
import torch
from torch import nn

# NOTE: Seed setting removed from module level
# Seed is managed centrally in entrypoint and train_model() functions
# Module-level seed setting can cause inconsistent behavior between HPO and Trainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRUwithFC(nn.Module):
    """GRU model with fully connected output layer.

    This model implements a Gated Recurrent Unit (GRU) network followed by
    a fully connected layer for sequence classification tasks.

    Args:
        feature_size: Number of expected features in the input.
        hidden_dim: Number of features in the hidden state.
        num_layers: Number of recurrent layers.
        out_dim: Output dimension of the fully connected layer.
        dropout_ratio: Dropout probability between GRU layers.
        classification: Whether this is a classification task.
        batch_first: If True, input/output tensors are (batch, seq, feature).

    """

    def __init__(
        self,
        feature_size: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 1,
        out_dim: int = 5,
        dropout_ratio: float = 0,
        classification: bool = True,
        batch_first: bool = True,
    ) -> None:
        """Initialize the GRU model."""
        super().__init__()
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_size = feature_size
        self.hidden_layer_size = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout_ratio = dropout_ratio
        self.classification = classification
        self.batch_first = batch_first

        # Initialize hidden state
        self.hidden_0 = torch.zeros(self.num_layers, 1, self.hidden_layer_size)

        # GRU layer
        self.gru = nn.GRU(
            input_size=self.feature_size,
            hidden_size=self.hidden_layer_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout_ratio,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, self.out_dim)
        # Note: No softmax layer - CrossEntropyLoss applies it internally

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GRU network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_size).

        Returns:
            Output logits of shape (batch_size, out_dim).

        Note:
            Returns logits (not probabilities). Use torch.softmax for inference.

        """
        batch_size = x.shape[0]

        # Determine device from input tensor
        if x.device.type == "cuda":
            self.DEVICE = f"{x.device.type}:{x.device.index}"
        elif x.device.type == "cpu":
            self.DEVICE = x.device.type

        # Initialize hidden state for this batch
        self.hidden_0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(
            self.DEVICE
        )

        # Forward propagation through GRU
        out, _ = self.gru(x, self.hidden_0)

        # Use the last time step output
        out = out[:, -1, :]

        out = self.layer_norm(out)
        out = self.dropout(out)

        # Fully connected layer returns logits
        return self.fc(out)


def main() -> None:
    """Test the GRU model with sample data and export to ONNX format."""
    import onnx
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

    # Model configuration
    batch_size = 8
    num_layers = 2
    input_size = 5
    hidden_size = 24
    seq_len = 100
    output_dim = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = GRUwithFC(
        feature_size=input_size,
        hidden_dim=hidden_size,
        num_layers=num_layers,
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
        output = model(data.to("cpu"))[-1]
    logger.info("Model output: %s", output)

    # Get prediction
    pred_label = output.argmax().tolist()
    logger.info("Predicted label: %d", pred_label)

    # Export to TorchScript
    traced_script_module = torch.jit.trace(model.eval(), data.to("cpu"))
    traced_script_module.save("GRU_test.pt")
    logger.info("Model saved to GRU_test.pt")

    # Export to ONNX
    torch.onnx.export(
        model.eval(),
        data.to("cpu"),
        "GRU_test.onnx",
        verbose=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Load and verify ONNX model
    gru_onnx_model = onnx.load("GRU_test.onnx")
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info("ONNX model structure:")
    logger.info(onnx.helper.printable_graph(gru_onnx_model.graph))

    # Test ONNX inference
    session = InferenceSession(
        "GRU_test.onnx",
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    result = session.run(None, {"input": np.array(data.to("cpu"))})
    logger.info("ONNX inference result: %s", result)


if __name__ == "__main__":
    main()
