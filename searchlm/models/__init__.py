"""
Models module for SearchLM.

This module provides domain models and evaluation models used throughout
the SearchLM system.
"""

from searchlm.models.domain import DatasetSplit, Document, Query
from searchlm.models.evaluation import QuerySearchResult, SearchResult

__all__ = [
    # Domain models
    "Document",
    "Query",
    "DatasetSplit",
    # Evaluation models
    "SearchResult",
    "QuerySearchResult",
]
