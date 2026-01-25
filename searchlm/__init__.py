"""
searchLM - A search system for PubMed NFCorpus and SciFact datasets.

This package provides:
- Dataset loading with train/test splits and qrels
- Dataset ingestion into tantivy search index
- Full-text search capabilities
- Unified search evaluator with automatic metrics calculation
- Evaluation metrics (NDCG, MRR, Precision, Recall, MAP)
"""

# Data loading
from searchlm.data import (
    DatasetLoader,
    DatasetSplit,
    Document,
    NFCorpusLoader,
    Query,
    SciFactLoader,
    create_loader,
)

# Evaluation
from searchlm.evaluation import (
    QuerySearchResult,
    SearchEvaluator,
    SearchResult,
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)

# Ingestion
from searchlm.ingestion import (
    DatasetIngester,
    NFCorpusIngester,
    SciFactIngester,
    ingest_all_datasets,
)

# Schema
from searchlm.schema import (
    FIELD_DATASET,
    FIELD_DOC_ID,
    FIELD_SOURCE_ID,
    FIELD_TEXT,
    FIELD_TITLE,
    IndexSchema,
    SEARCHABLE_FIELDS,
)

# Search
from searchlm.search_engine import SearchEngine

__version__ = "0.0.1"


def load_dataset_split(dataset_name: str, split: str = "test") -> DatasetSplit:
    """
    Convenience function to load a dataset split.

    Args:
        dataset_name: Dataset name ("nfcorpus" or "scifact")
        split: Dataset split ("train", "dev", or "test")

    Returns:
        DatasetSplit object with queries, documents, and qrels
    """
    loader = create_loader(dataset_name)
    return loader.load_split(split=split)


__all__ = [
    # Data loading
    "DatasetLoader",
    "NFCorpusLoader",
    "SciFactLoader",
    "create_loader",
    "load_dataset_split",
    "Document",
    "Query",
    "DatasetSplit",
    # Search
    "SearchEngine",
    # Ingestion
    "DatasetIngester",
    "NFCorpusIngester",
    "SciFactIngester",
    "ingest_all_datasets",
    # Evaluation
    "SearchEvaluator",
    "SearchResult",
    "QuerySearchResult",
    "calculate_ndcg",
    "calculate_mrr",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_map",
    # Schema
    "IndexSchema",
    "FIELD_DOC_ID",
    "FIELD_TITLE",
    "FIELD_TEXT",
    "FIELD_DATASET",
    "FIELD_SOURCE_ID",
    "SEARCHABLE_FIELDS",
]
