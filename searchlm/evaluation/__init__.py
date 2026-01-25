"""
Evaluation module for measuring search quality using IR metrics.

Provides evaluator classes, result models, and metric calculation
functions for assessing search engine performance.
"""

from searchlm.evaluation.evaluator import SearchEvaluator
from searchlm.evaluation.metrics import (
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from searchlm.evaluation.models import QuerySearchResult, SearchResult

__all__ = [
    # Evaluator
    "SearchEvaluator",
    # Models
    "SearchResult",
    "QuerySearchResult",
    # Metrics
    "calculate_ndcg",
    "calculate_mrr",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_map",
]
