"""Dataset loading utilities and custom PyTorch dataset classes.

This module provides functions and classes for:
- Setting random seeds for reproducibility
- Parsing JSON configuration files
- Loading file lists from directories
- Custom tensor datasets for training/validation/testing
"""

import collections
import glob
import json
import logging
import os
from typing import Any

import natsort
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def jsonfileparser(
    filepath: str,
    encoding: str = "utf-8",
) -> collections.OrderedDict:
    """Parse JSON file and return as OrderedDict.

    Args:
        filepath: Path to JSON configuration file.
        encoding: File encoding (default: 'utf-8').

    Returns:
        Parsed JSON as OrderedDict.

    Examples:
        >>> config = jsonfileparser('config.json')
        >>> config['model']['hidden_dim']
        128

    """
    with open(filepath, encoding=encoding) as file:
        result: collections.OrderedDict[str, Any] = json.load(
            file, object_pairs_hook=collections.OrderedDict
        )
        return result


def get_all_file_list(file_path: str, ext: str) -> list[str]:
    """Get all files matching extension pattern from directory.

    Args:
        file_path: Directory path to search.
        ext: File extension pattern (e.g., '*.csv', '*.json').

    Returns:
        Naturally sorted list of file paths.

    Examples:
        >>> files = get_all_file_list('/data/train', '*.csv')
        >>> len(files)
        100

    """
    file_list: list[str] = natsort.natsorted(glob.glob(os.path.join(file_path, ext)))
    return file_list


# Backward compatibility alias (deprecated)
GetAllFileList = get_all_file_list


class CustomTensorDataset(Dataset):
    """Custom PyTorch dataset for converting NumPy arrays to tensors.

    This dataset can be used for training, validation, and test data.
    Converts features and labels to PyTorch tensors on initialization
    for efficiency.

    Args:
        features: Input feature array.
        labels: Corresponding label array.

    Examples:
        >>> features = np.random.randn(100, 50, 10)
        >>> labels = np.random.randint(0, 4, 100)
        >>> dataset = CustomTensorDataset(features, labels)
        >>> len(dataset)
        100
        >>> x, y = dataset[0]
        >>> x.shape
        torch.Size([50, 10])

    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Initialize dataset with features and labels.

        Converts NumPy arrays to tensors once during initialization
        for better performance.
        """
        # Converting to tensor once in __init__ is more efficient
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(int), dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feature-label pair at given index."""
        return self.features[idx], self.labels[idx]
