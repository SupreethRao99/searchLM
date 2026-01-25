"""
Data models for search evaluation results.

This module provides dataclasses for representing search results
and query evaluation results with metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SearchResult:
    """
    A single search result document.
    
    Represents a document returned from a search query with its
    relevance score and optional ground truth relevance.
    
    Attributes:
        doc_id: Document identifier
        title: Document title
        text: Document text content
        score: Search relevance score
        dataset: Dataset name
        relevance: Ground truth relevance (if available)
        rank: Position in ranking (1-based)
    """
    
    doc_id: str
    title: str
    text: str
    score: float
    dataset: str
    relevance: float = 0.0  # Ground truth relevance (if available)
    rank: int = 0  # Position in ranking (1-based)


@dataclass
class QuerySearchResult:
    """
    Search results for a single query with evaluation metrics.
    
    Encapsulates all information about a query's search results,
    including retrieved documents and calculated evaluation metrics.
    
    Attributes:
        query_id: Query identifier
        query_text: Query text
        dataset_name: Dataset name
        results: List of search results
        metrics: Calculated evaluation metrics
        num_retrieved: Number of documents retrieved
        num_relevant_total: Total number of relevant documents
    """
    
    query_id: str
    query_text: str
    dataset_name: str
    results: List[SearchResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    num_retrieved: int = 0
    num_relevant_total: int = 0
    
    @property
    def has_metrics(self) -> bool:
        """Check if metrics have been calculated."""
        return len(self.metrics) > 0
    
    @property
    def ndcg_at_10(self) -> float:
        """Get NDCG@10 metric."""
        return self.metrics.get("ndcg@10", 0.0)
    
    @property
    def recall_at_10(self) -> float:
        """Get Recall@10 metric."""
        return self.metrics.get("recall@10", 0.0)
    
    @property
    def mrr(self) -> float:
        """Get MRR metric."""
        return self.metrics.get("mrr", 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the query search result
        """
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "dataset_name": self.dataset_name,
            "num_retrieved": self.num_retrieved,
            "num_relevant_total": self.num_relevant_total,
            "metrics": self.metrics,
            "results": [
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                    "score": r.score,
                    "relevance": r.relevance,
                    "rank": r.rank,
                }
                for r in self.results
            ],
        }
