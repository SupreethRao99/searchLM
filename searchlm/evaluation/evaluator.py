"""
Search evaluator for measuring search quality using IR metrics.

This module provides the SearchEvaluator class for evaluating search
results against ground truth relevance judgments.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from searchlm.data import create_loader
from searchlm.evaluation.metrics import (
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from searchlm.evaluation.models import QuerySearchResult, SearchResult
from searchlm.search_engine import SearchEngine


class SearchEvaluator:
    """
    Unified search evaluator for measuring search quality using IR metrics.
    
    Provides methods for:
    - Loading queries and relevance judgments from MTEB datasets
    - Evaluating single queries or entire datasets
    - Calculating IR metrics (NDCG, MRR, Precision, Recall, MAP)
    - Optionally returning search results along with metrics
    
    This is the primary interface for search evaluation. Use:
    - `evaluate_single_query()` for single query evaluation (returns metrics dict)
    - `evaluate_single_query_with_results()` if you also need the retrieved documents
    - `evaluate_batch()` for batch evaluation (returns aggregate metrics)
    """
    
    def __init__(self, index_path: str = "./search_index"):
        """
        Initialize the evaluator.
        
        Args:
            index_path: Path to the tantivy index
        """
        self.search_engine = SearchEngine(index_path=index_path)
        self.index_path = index_path
    
    def load_qrels(
        self,
        dataset_name: str,
        split: str = "test"
    ) -> Dict[str, Dict[str, float]]:
        """
        Load relevance judgments (qrels) from MTEB dataset.
        
        Args:
            dataset_name: Dataset name ("nfcorpus" or "scifact")
            split: Dataset split ("train", "dev", or "test")
        
        Returns:
            Dictionary mapping query_id -> {doc_id: relevance_score}
        """
        print(f"Loading qrels for {dataset_name} ({split} split)...")
        
        loader = create_loader(dataset_name)
        dataset_split = loader.load_split(split=split)
        
        print(f"Loaded {len(dataset_split.qrels)} queries with relevance judgments")
        return dataset_split.qrels
    
    def load_queries(
        self,
        dataset_name: str,
        split: str = "test"
    ) -> Dict[str, str]:
        """
        Load queries from MTEB dataset.
        
        Args:
            dataset_name: Dataset name ("nfcorpus" or "scifact")
            split: Dataset split ("train", "dev", or "test")
        
        Returns:
            Dictionary mapping query_id -> query_text
        """
        print(f"Loading queries for {dataset_name} ({split} split)...")
        
        loader = create_loader(dataset_name)
        dataset_split = loader.load_split(split=split)
        
        queries = {
            query_id: query.text
            for query_id, query in dataset_split.queries.items()
        }
        
        print(f"Loaded {len(queries)} queries")
        return queries
    
    def evaluate_single_query(
        self,
        query_text: str,
        query_id: str,
        dataset_name: str,
        split: str = "test",
        k: int = 100,
        dataset_filter: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """
        Evaluate a single query by loading its qrels from the dataset.
        
        Convenient for evaluating generated queries.
        
        Args:
            query_text: Search query to evaluate (tantivy query string)
            query_id: Query ID to look up in qrels
            dataset_name: Dataset name ("nfcorpus" or "scifact")
            split: Dataset split ("train", "dev", or "test")
            k: Maximum number of results to retrieve
            dataset_filter: Filter by dataset name (optional, defaults to dataset_name)
        
        Returns:
            Tuple of (metrics_dict, error_message):
            - metrics_dict: Dictionary with evaluation metrics (None if error occurred)
            - error_message: Error message if query failed (None if successful)
            
            Metrics dictionary contains:
            - ndcg@10, ndcg@100: NDCG at ranks 10 and 100
            - mrr: Mean Reciprocal Rank
            - precision@10: Precision at rank 10
            - recall@10: Recall at rank 10
            - map: Mean Average Precision
            - retrieved: Number of documents retrieved
            - relevant_in_collection: Total relevant documents
        """
        # Load qrels for this specific query
        qrels_all = self.load_qrels(dataset_name, split=split)
        qrels = qrels_all.get(query_id, {})
        
        if not qrels:
            print(f"Warning: No qrels found for query_id {query_id}")
            return None, "No qrels found"
        
        # Use dataset_name as filter if not specified
        if dataset_filter is None:
            dataset_filter = dataset_name
        
        return self.evaluate_query(
            query_text, qrels, k=k, dataset_filter=dataset_filter
        )
    
    def evaluate_query(
        self,
        query_text: str,
        qrels: Dict[str, float],
        k: int = 100,
        dataset_filter: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """
        Evaluate a single query against relevance judgments.
        
        Args:
            query_text: Search query
            qrels: Dictionary mapping doc_id -> relevance_score for this query
            k: Maximum number of results to retrieve
            dataset_filter: Filter by dataset name (optional)
        
        Returns:
            Tuple of (metrics_dict, error_message):
            - metrics_dict: Dictionary with evaluation metrics (None if error occurred)
            - error_message: Error message if query failed (None if successful)
            
            Metrics dictionary contains:
            - ndcg@10, ndcg@100: NDCG at ranks 10 and 100
            - mrr: Mean Reciprocal Rank
            - precision@10: Precision at rank 10
            - recall@10: Recall at rank 10
            - map: Mean Average Precision
            - retrieved: Number of documents retrieved
            - relevant_in_collection: Total relevant documents
        """
        # Perform search with error handling
        try:
            results = self.search_engine.search(
                query_text, limit=k, dataset_filter=dataset_filter
            )
        except ValueError as e:
            # Return error message for syntax errors
            return None, str(e)
        except Exception as e:
            # Catch any other unexpected errors
            return None, f"Unexpected error: {str(e)}"
        
        # Create relevance score list for retrieved documents
        relevance_scores = []
        for result in results:
            doc_id = result["doc_id"]
            # Get relevance score (0 if not in qrels)
            relevance = qrels.get(doc_id, 0.0)
            relevance_scores.append(relevance)
        
        # Calculate metrics
        total_relevant = sum(1 for score in qrels.values() if score > 0)
        
        metrics = {
            "ndcg@10": calculate_ndcg(relevance_scores, k=10),
            "ndcg@100": calculate_ndcg(relevance_scores, k=100),
            "mrr": calculate_mrr(relevance_scores),
            "precision@10": calculate_precision_at_k(relevance_scores, k=10),
            "recall@10": calculate_recall_at_k(
                relevance_scores, k=10, total_relevant=total_relevant
            ),
            "map": calculate_map(relevance_scores),
            "retrieved": len(results),
            "relevant_in_collection": total_relevant,
        }
        
        return metrics, None
    
    def evaluate_single_query_with_results(
        self,
        query_text: str,
        query_id: str,
        dataset_name: str,
        split: str = "test",
        k: int = 100,
        dataset_filter: Optional[str] = None,
    ) -> Tuple[Optional[QuerySearchResult], Optional[str]]:
        """
        Evaluate a single query and return both metrics and search results.
        
        Args:
            query_text: Search query to evaluate (tantivy query string)
            query_id: Query ID to look up in qrels
            dataset_name: Dataset name ("nfcorpus" or "scifact")
            split: Dataset split ("train", "dev", or "test")
            k: Maximum number of results to retrieve
            dataset_filter: Filter by dataset name (optional, defaults to dataset_name)
        
        Returns:
            Tuple of (QuerySearchResult, error_message):
            - QuerySearchResult: Search results and calculated metrics (None if error occurred)
            - error_message: Error message if query failed (None if successful)
        """
        # Load qrels for this specific query
        qrels_all = self.load_qrels(dataset_name, split=split)
        qrels = qrels_all.get(query_id, {})
        
        if not qrels:
            print(f"Warning: No qrels found for query_id {query_id}")
            return None, "No qrels found"
        
        # Use dataset_name as filter if not specified
        if dataset_filter is None:
            dataset_filter = dataset_name
        
        return self.evaluate_query_with_results(
            query_text, query_id, qrels, dataset_name, k=k, dataset_filter=dataset_filter
        )
    
    def evaluate_query_with_results(
        self,
        query_text: str,
        query_id: str,
        qrels: Dict[str, float],
        dataset_name: str,
        k: int = 100,
        dataset_filter: Optional[str] = None,
    ) -> Tuple[Optional[QuerySearchResult], Optional[str]]:
        """
        Evaluate a single query and return both metrics and search results.
        
        Args:
            query_text: Search query
            query_id: Query identifier
            qrels: Dictionary mapping doc_id -> relevance_score for this query
            dataset_name: Name of the dataset
            k: Maximum number of results to retrieve
            dataset_filter: Filter by dataset name (optional)
        
        Returns:
            Tuple of (QuerySearchResult, error_message):
            - QuerySearchResult: Search results and calculated metrics (None if error occurred)
            - error_message: Error message if query failed (None if successful)
        """
        # Perform search with error handling
        try:
            raw_results = self.search_engine.search(
                query_text, limit=k, dataset_filter=dataset_filter
            )
        except ValueError as e:
            return None, str(e)
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
        
        # Convert to SearchResult objects and attach relevance
        results = []
        relevance_scores = []
        for rank, result in enumerate(raw_results, start=1):
            doc_id = result["doc_id"]
            relevance = qrels.get(doc_id, 0.0)
            relevance_scores.append(relevance)
            
            search_result = SearchResult(
                doc_id=doc_id,
                title=result["title"],
                text=result["text"],
                score=result["score"],
                dataset=result["dataset"],
                relevance=relevance,
                rank=rank,
            )
            results.append(search_result)
        
        # Calculate metrics
        total_relevant = sum(1 for score in qrels.values() if score > 0)
        
        metrics = {
            "ndcg@10": calculate_ndcg(relevance_scores, k=10),
            "ndcg@100": calculate_ndcg(relevance_scores, k=100),
            "mrr": calculate_mrr(relevance_scores),
            "precision@10": calculate_precision_at_k(relevance_scores, k=10),
            "recall@10": calculate_recall_at_k(
                relevance_scores, k=10, total_relevant=total_relevant
            ),
            "recall@100": calculate_recall_at_k(
                relevance_scores, k=100, total_relevant=total_relevant
            ),
            "map": calculate_map(relevance_scores),
        }
        
        query_result = QuerySearchResult(
            query_id=query_id,
            query_text=query_text,
            dataset_name=dataset_name,
            results=results,
            metrics=metrics,
            num_retrieved=len(results),
            num_relevant_total=total_relevant,
        )
        
        return query_result, None
    
    def search(
        self,
        query_text: str,
        limit: int = 100,
        dataset_filter: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Perform a search query without evaluation.
        
        Args:
            query_text: Search query string
            limit: Maximum number of results to return
            dataset_filter: Filter by dataset name ("nfcorpus" or "scifact")
            fields: Fields to search in (default: ["title", "text"])
        
        Returns:
            List of SearchResult objects ordered by relevance score
        """
        # Execute search
        raw_results = self.search_engine.search(
            query=query_text,
            limit=limit,
            fields=fields,
            dataset_filter=dataset_filter,
        )
        
        # Convert to SearchResult objects
        results = []
        for rank, result in enumerate(raw_results, start=1):
            search_result = SearchResult(
                doc_id=result["doc_id"],
                title=result["title"],
                text=result["text"],
                score=result["score"],
                dataset=result["dataset"],
                rank=rank,
            )
            results.append(search_result)
        
        return results
    
    def evaluate(
        self,
        dataset_name: str,
        split: str = "test",
        k: int = 100,
        dataset_filter: Optional[str] = None,
        max_queries: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate search quality on a dataset.
        
        Args:
            dataset_name: Dataset name ("nfcorpus" or "scifact")
            split: Dataset split ("train", "dev", or "test")
            k: Maximum number of results to retrieve per query
            dataset_filter: Filter by dataset name (optional)
            max_queries: Maximum number of queries to evaluate (None for all)
        
        Returns:
            Dictionary with evaluation results including metrics and failed queries
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name} ({split} split)")
        print(f"{'=' * 60}\n")
        
        # Load queries and qrels
        queries = self.load_queries(dataset_name, split=split)
        qrels_all = self.load_qrels(dataset_name, split=split)
        
        # Filter to queries that have both query text and qrels
        valid_query_ids = set(queries.keys()) & set(qrels_all.keys())
        
        if max_queries:
            valid_query_ids = list(valid_query_ids)[:max_queries]
        else:
            valid_query_ids = list(valid_query_ids)
        
        print(f"Evaluating {len(valid_query_ids)} queries...\n")
        
        # Evaluate each query
        all_metrics = defaultdict(list)
        failed_queries = []
        
        for query_id in tqdm(valid_query_ids, desc="Evaluating queries"):
            query_text = queries[query_id]
            qrels = qrels_all[query_id]
            
            metrics, error = self.evaluate_query(
                query_text, qrels, k=k, dataset_filter=dataset_filter
            )
            
            if error:
                # Query failed due to syntax error or other issue
                failed_queries.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "error": error
                })
                continue
            
            # Accumulate metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        # Calculate averages
        avg_metrics = {
            metric_name: np.mean(values)
            for metric_name, values in all_metrics.items()
            if metric_name not in ["retrieved", "relevant_in_collection"]
        }
        
        # Add summary stats
        avg_metrics["avg_retrieved"] = np.mean(all_metrics.get("retrieved", [0]))
        avg_metrics["avg_relevant"] = np.mean(
            all_metrics.get("relevant_in_collection", [0])
        )
        avg_metrics["num_queries"] = len(all_metrics.get("retrieved", []))
        avg_metrics["num_failed"] = len(failed_queries)
        avg_metrics["failed_queries"] = failed_queries
        
        return avg_metrics
    
    def evaluate_batch(
        self,
        queries: List[Tuple[str, str]],
        dataset_name: str,
        split: str = "test",
        k: int = 100,
        dataset_filter: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict:
        """
        Evaluate a batch of queries efficiently.
        
        Args:
            queries: List of (query_text, query_id) tuples to evaluate
            dataset_name: Dataset name ("nfcorpus" or "scifact")
            split: Dataset split ("train", "dev", or "test")
            k: Maximum number of results to retrieve per query
            dataset_filter: Filter by dataset name (optional, defaults to dataset_name)
            show_progress: Show progress bar during evaluation
        
        Returns:
            Dictionary with evaluation results:
            - ndcg@10, ndcg@100: Average NDCG at ranks 10 and 100
            - mrr: Average Mean Reciprocal Rank
            - precision@10: Average Precision at rank 10
            - recall@10: Average Recall at rank 10
            - map: Average Mean Average Precision
            - avg_retrieved: Average number of documents retrieved
            - avg_relevant: Average number of relevant documents
            - num_queries: Number of queries successfully evaluated
            - num_failed: Number of queries that failed
            - failed_queries: List of dicts with 'query_id', 'query_text', and 'error' for failed queries
        """
        # Load all qrels once for efficiency
        qrels_all = self.load_qrels(dataset_name, split=split)
        
        # Use dataset_name as filter if not specified
        if dataset_filter is None:
            dataset_filter = dataset_name
        
        # Evaluate each query
        all_metrics = defaultdict(list)
        failed_queries = []
        
        iterator = tqdm(queries, desc="Evaluating queries") if show_progress else queries
        
        for query_text, query_id in iterator:
            qrels = qrels_all.get(query_id, {})
            
            if not qrels:
                failed_queries.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "error": "No qrels found"
                })
                continue
            
            metrics, error = self.evaluate_query(
                query_text, qrels, k=k, dataset_filter=dataset_filter
            )
            
            if error:
                # Query failed due to syntax error or other issue
                failed_queries.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "error": error
                })
                continue
            
            # Accumulate metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        if not all_metrics:
            return {
                "num_queries": 0,
                "num_failed": len(failed_queries),
                "failed_queries": failed_queries,
            }
        
        # Calculate averages
        avg_metrics = {
            metric_name: np.mean(values)
            for metric_name, values in all_metrics.items()
            if metric_name not in ["retrieved", "relevant_in_collection"]
        }
        
        # Add summary stats
        avg_metrics["avg_retrieved"] = np.mean(all_metrics.get("retrieved", [0]))
        avg_metrics["avg_relevant"] = np.mean(
            all_metrics.get("relevant_in_collection", [0])
        )
        avg_metrics["num_queries"] = len(all_metrics.get("retrieved", []))
        avg_metrics["num_failed"] = len(failed_queries)
        avg_metrics["failed_queries"] = failed_queries
        
        return avg_metrics
    
    def print_metrics(self, metrics: Dict):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Number of queries: {metrics.get('num_queries', 0):.0f}")
        
        # Print failed queries info if present
        num_failed = metrics.get('num_failed', 0)
        if num_failed > 0:
            print(f"Number of failed queries: {num_failed}")
        
        print(f"Average relevant docs per query: {metrics.get('avg_relevant', 0):.2f}")
        print(
            f"Average retrieved docs per query: {metrics.get('avg_retrieved', 0):.2f}"
        )
        print("\nMetrics:")
        print(f"  NDCG@10:  {metrics.get('ndcg@10', 0):.4f}")
        print(f"  NDCG@100: {metrics.get('ndcg@100', 0):.4f}")
        print(f"  MRR:      {metrics.get('mrr', 0):.4f}")
        print(f"  Precision@10: {metrics.get('precision@10', 0):.4f}")
        print(f"  Recall@10:    {metrics.get('recall@10', 0):.4f}")
        print(f"  MAP:      {metrics.get('map', 0):.4f}")
        
        # Print failed queries details if present
        failed_queries = metrics.get('failed_queries', [])
        if failed_queries:
            print("\n" + "=" * 60)
            print(f"Failed Queries ({len(failed_queries)}):")
            print("=" * 60)
            for i, failed in enumerate(failed_queries[:10], 1):  # Show first 10
                print(f"\n{i}. Query ID: {failed['query_id']}")
                print(f"   Query: {failed['query_text'][:80]}...")
                print(f"   Error: {failed['error']}")
            
            if len(failed_queries) > 10:
                print(f"\n... and {len(failed_queries) - 10} more failed queries")
        
        print("=" * 60 + "\n")
