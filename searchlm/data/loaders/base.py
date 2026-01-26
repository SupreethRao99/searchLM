"""
Base class for loading IR datasets from HuggingFace.

This module provides the DatasetLoader abstract base class that defines
the interface for loading queries, documents, and relevance judgments.
"""

from pathlib import Path
from typing import Dict, Optional

from searchlm.models.domain import DatasetSplit, Document, Query


class DatasetLoader:
    """
    Abstract base class for loading IR datasets from HuggingFace.

    Handles downloading queries, documents (corpus), and relevance
    judgments (qrels) from MTEB datasets.

    Subclasses must implement:
    - load_corpus(): Load documents
    - load_queries(): Load queries for a split
    - load_qrels(): Load relevance judgments

    Attributes:
        DATASET_NAME: Human-readable dataset name
        DATASET_SOURCE: HuggingFace dataset identifier
    """

    DATASET_NAME: str = ""
    DATASET_SOURCE: str = ""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the dataset loader.

        Args:
            cache_dir: Optional directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset_name = self.DATASET_NAME

    def load_corpus(self) -> Dict[str, Document]:
        """
        Load the document corpus.

        Returns:
            Dictionary mapping doc_id -> Document

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement load_corpus()")

    def load_queries(
        self, split: str = "test", qrels: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Query]:
        """
        Load queries for a specific split.

        Args:
            split: Dataset split ("train", "dev", or "test")
            qrels: Optional pre-loaded qrels to filter queries.
                If None, will load qrels.

        Returns:
            Dictionary mapping query_id -> Query

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement load_queries()")

    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, float]]:
        """
        Load relevance judgments (qrels) for a specific split.

        Args:
            split: Dataset split ("train", "dev", or "test")

        Returns:
            Dictionary mapping query_id -> {doc_id -> relevance_score}

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement load_qrels()")

    def load_split(self, split: str = "test") -> DatasetSplit:
        """
        Load a complete dataset split with queries, documents, and qrels.

        This method orchestrates loading all components of a dataset split
        and combines them into a DatasetSplit object.

        Args:
            split: Dataset split ("train", "dev", or "test")

        Returns:
            DatasetSplit object containing all data for the split
        """
        print(f"\n{'=' * 60}")
        print(f"Loading {self.dataset_name} ({split} split)")
        print(f"{'=' * 60}\n")

        # Load corpus (shared across all splits)
        documents = self.load_corpus()

        # Load qrels first (from "default" subset with specific split)
        qrels = self.load_qrels(split=split)

        # Load queries for this split (pass qrels to avoid reloading)
        queries = self.load_queries(split=split, qrels=qrels)

        # Attach qrels to queries
        for query_id, query in queries.items():
            query.qrels = qrels.get(query_id, {})

        dataset_split = DatasetSplit(
            name=split,
            dataset_name=self.dataset_name,
            queries=queries,
            documents=documents,
            qrels=qrels,
        )

        print(f"✓ Loaded {dataset_split.num_queries} queries")
        print(f"✓ Loaded {dataset_split.num_documents} documents")
        print(f"✓ Loaded {len(qrels)} query-document relevance judgments\n")

        return dataset_split
