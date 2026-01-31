"""
Services module for SearchLM.

This module provides high-level services for search and evaluation.
"""

from searchlm.services.evaluator import SearchEvaluator
from searchlm.services.metrics import (calculate_map, calculate_mrr,
                                       calculate_ndcg,
                                       calculate_precision_at_k,
                                       calculate_recall_at_k)
from searchlm.services.search import SearchEngine

__all__ = [
    # Search
    "SearchEngine",
    # Evaluator
    "SearchEvaluator",
    # Metrics
    "calculate_ndcg",
    "calculate_mrr",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_map",
]
