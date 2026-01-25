"""
Information Retrieval (IR) evaluation metrics.

This module provides implementations of standard IR metrics:
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Precision@K
- Recall@K
- MAP (Mean Average Precision)
"""

from typing import List, Optional

import numpy as np


def calculate_ndcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).

    NDCG measures ranking quality by considering the position of relevant items.
    It uses a logarithmic discount factor that reduces the contribution of
    relevant items appearing lower in the ranking.

    Args:
        relevance_scores: List of relevance scores for ranked documents (ordered by rank)
                         Higher scores indicate more relevant documents.
        k: Cutoff rank (if None, uses all scores)

    Returns:
        NDCG score (0.0 to 1.0), where 1.0 is perfect ranking

    Example:
        >>> scores = [1.0, 0.5, 0.0, 1.0, 0.0]  # Relevant docs at positions 1, 2, 4
        >>> calculate_ndcg(scores, k=5)
        0.869...
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    if not relevance_scores:
        return 0.0

    # Calculate DCG (Discounted Cumulative Gain)
    # DCG = sum(rel_i / log2(i + 2)) for i in range(len(scores))
    # Using i+2 because log2(1) = 0, so we start from position 2
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

    # Calculate IDCG (Ideal DCG - sorted in descending order)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_mrr(relevance_scores: List[float]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR measures the rank of the first relevant document.
    It's the reciprocal of the position of the first relevant item.

    Args:
        relevance_scores: List of relevance scores for ranked documents
                         (binary: 0 = not relevant, >0 = relevant)

    Returns:
        MRR score (0.0 to 1.0), where 1.0 means first result is relevant

    Example:
        >>> scores = [0.0, 0.0, 1.0, 0.0]  # First relevant at position 3
        >>> calculate_mrr(scores)
        0.333...  # 1/3
    """
    for i, score in enumerate(relevance_scores):
        if score > 0:
            return 1.0 / (i + 1)
    return 0.0


def calculate_precision_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Precision@K.

    Precision@K is the proportion of relevant documents in the top K results.

    Args:
        relevance_scores: List of relevance scores (binary: 0 or 1)
        k: Cutoff rank

    Returns:
        Precision@K score (0.0 to 1.0)

    Example:
        >>> scores = [1, 1, 0, 1, 0]  # 3 relevant out of 5
        >>> calculate_precision_at_k(scores, k=5)
        0.6  # 3/5
    """
    if k > len(relevance_scores):
        k = len(relevance_scores)

    if k == 0:
        return 0.0

    relevant_count = sum(1 for score in relevance_scores[:k] if score > 0)
    return relevant_count / k


def calculate_recall_at_k(
    relevance_scores: List[float], k: int, total_relevant: int
) -> float:
    """
    Calculate Recall@K.

    Recall@K is the proportion of relevant documents retrieved in the top K results
    out of all relevant documents in the collection.

    Args:
        relevance_scores: List of relevance scores (binary: 0 or 1)
        k: Cutoff rank
        total_relevant: Total number of relevant documents in the collection

    Returns:
        Recall@K score (0.0 to 1.0)

    Example:
        >>> scores = [1, 1, 0, 0, 0]  # 2 relevant retrieved
        >>> calculate_recall_at_k(scores, k=5, total_relevant=10)
        0.2  # 2/10
    """
    if total_relevant == 0:
        return 0.0

    if k > len(relevance_scores):
        k = len(relevance_scores)

    relevant_retrieved = sum(1 for score in relevance_scores[:k] if score > 0)
    return relevant_retrieved / total_relevant


def calculate_map(relevance_scores: List[float]) -> float:
    """
    Calculate Mean Average Precision (MAP).

    MAP is the mean of the average precision scores for each relevant document.
    It considers both precision and the position of relevant documents.

    Args:
        relevance_scores: List of relevance scores (binary: 0 or 1)

    Returns:
        MAP score (0.0 to 1.0)

    Example:
        >>> scores = [1, 0, 1, 0, 1]  # Relevant at positions 1, 3, 5
        >>> calculate_map(scores)
        0.766...  # Average of precisions at each relevant position
    """
    if not relevance_scores:
        return 0.0

    total_relevant = sum(1 for score in relevance_scores if score > 0)
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, score in enumerate(relevance_scores):
        if score > 0:
            relevant_count += 1
            # Precision at this position
            precision_sum += relevant_count / (i + 1)

    return precision_sum / total_relevant
