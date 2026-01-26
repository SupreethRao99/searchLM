"""
Factory for creating dataset loaders.

This module provides a factory function for creating the appropriate
dataset loader based on the dataset name.
"""

from pathlib import Path
from typing import Optional

from searchlm.data.loaders.base import DatasetLoader
from searchlm.data.loaders.nfcorpus import NFCorpusLoader
from searchlm.data.loaders.scifact import SciFactLoader

# Registry of available dataset loaders
DATASET_LOADERS = {
    "nfcorpus": NFCorpusLoader,
    "scifact": SciFactLoader,
}


def create_loader(dataset_name: str, cache_dir: Optional[Path] = None) -> DatasetLoader:
    """
    Create a dataset loader for the specified dataset.

    Args:
        dataset_name: Dataset name ("nfcorpus" or "scifact")
        cache_dir: Optional directory to cache downloaded datasets

    Returns:
        DatasetLoader instance for the specified dataset

    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_LOADERS.keys())}"
        )
    
    return DATASET_LOADERS[dataset_name](cache_dir=cache_dir)
