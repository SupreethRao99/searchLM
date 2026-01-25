"""
Data loading module for IR datasets.

Provides loaders for different datasets, data models, and utilities
for working with queries, documents, and relevance judgments.
"""

from searchlm.data.base_loader import DatasetLoader
from searchlm.data.factory import create_loader
from searchlm.data.models import DatasetSplit, Document, Query
from searchlm.data.nfcorpus_loader import NFCorpusLoader
from searchlm.data.scifact_loader import SciFactLoader

__all__ = [
    # Base classes
    "DatasetLoader",
    # Loaders
    "NFCorpusLoader",
    "SciFactLoader",
    # Factory
    "create_loader",
    # Models
    "Document",
    "Query",
    "DatasetSplit",
]
