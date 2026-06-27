"""
Data loaders for IR datasets.

This module provides loaders for various IR datasets from HuggingFace MTEB.
"""

from searchlm.data.loaders.arguana import ArguAnaLoader
from searchlm.data.loaders.base import DatasetLoader
from searchlm.data.loaders.factory import create_loader
from searchlm.data.loaders.fiqa import FiQALoader
from searchlm.data.loaders.hotpotqa import HotpotQALoader
from searchlm.data.loaders.nfcorpus import NFCorpusLoader
from searchlm.data.loaders.nq import NQLoader
from searchlm.data.loaders.scifact import SciFactLoader

__all__ = [
    "DatasetLoader",
    "NFCorpusLoader",
    "SciFactLoader",
    "FiQALoader",
    "ArguAnaLoader",
    "HotpotQALoader",
    "NQLoader",
    "create_loader",
]
