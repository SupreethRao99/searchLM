"""
searchLM - Search system with RLHF training capabilities.

This package provides:
- Dataset loading with train/test splits and qrels
- Dataset ingestion into tantivy search index
- Full-text search capabilities
- Unified search evaluator with automatic metrics calculation
- Evaluation metrics (NDCG, MRR, Precision, Recall, MAP)
- RLHF training workflows for query generation
"""

# Configuration
from searchlm.config import get_config, load_config, merge_config
from searchlm.data.ingesters import (
    DatasetIngester,
    NFCorpusIngester,
    SciFactIngester,
    ingest_all_datasets,
)

# Data layer
from searchlm.data.loaders import (
    DatasetLoader,
    NFCorpusLoader,
    SciFactLoader,
    create_loader,
)
from searchlm.data.schemas import (
    FIELD_DATASET,
    FIELD_DOC_ID,
    FIELD_SOURCE_ID,
    FIELD_TEXT,
    FIELD_TITLE,
    SEARCHABLE_FIELDS,
    IndexSchema,
)

# Models
from searchlm.models.domain import DatasetSplit, Document, Query
from searchlm.models.evaluation import QuerySearchResult, SearchResult
from searchlm.services.evaluator import SearchEvaluator
from searchlm.services.metrics import (
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)

# Services
from searchlm.services.search import SearchEngine

__version__ = "0.1.0"


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
    # Config
    "load_config",
    "get_config",
    "merge_config",
    # Models
    "Document",
    "Query",
    "DatasetSplit",
    "SearchResult",
    "QuerySearchResult",
    # Data loaders
    "DatasetLoader",
    "NFCorpusLoader",
    "SciFactLoader",
    "create_loader",
    "load_dataset_split",
    # Data ingesters
    "DatasetIngester",
    "NFCorpusIngester",
    "SciFactIngester",
    "ingest_all_datasets",
    # Schemas
    "IndexSchema",
    "FIELD_DOC_ID",
    "FIELD_TITLE",
    "FIELD_TEXT",
    "FIELD_DATASET",
    "FIELD_SOURCE_ID",
    "SEARCHABLE_FIELDS",
    # Services
    "SearchEngine",
    "SearchEvaluator",
    # Metrics
    "calculate_ndcg",
    "calculate_mrr",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_map",
]
