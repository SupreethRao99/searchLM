"""
Data loaders for IR datasets.

This module provides loaders for various IR datasets from HuggingFace MTEB.
"""

from searchlm.data.loaders.base import DatasetLoader
from searchlm.data.loaders.factory import create_loader
from searchlm.data.loaders.nfcorpus import NFCorpusLoader
from searchlm.data.loaders.scifact import SciFactLoader

__all__ = [
    "DatasetLoader",
    "NFCorpusLoader",
    "SciFactLoader",
    "create_loader",
]
