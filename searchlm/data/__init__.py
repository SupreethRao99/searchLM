"""
Data module for SearchLM.

This module provides data loading, ingestion, and schema management
for IR datasets.
"""

# Ingesters
from searchlm.data.ingesters import (
    DatasetIngester,
    NFCorpusIngester,
    SciFactIngester,
    ingest_all_datasets,
)

# Loaders
from searchlm.data.loaders import (
    DatasetLoader,
    NFCorpusLoader,
    SciFactLoader,
    create_loader,
)

# Schemas
from searchlm.data.schemas import (
    FIELD_DATASET,
    FIELD_DOC_ID,
    FIELD_SOURCE_ID,
    FIELD_TEXT,
    FIELD_TITLE,
    SEARCHABLE_FIELDS,
    IndexSchema,
)

__all__ = [
    # Loaders
    "DatasetLoader",
    "NFCorpusLoader",
    "SciFactLoader",
    "create_loader",
    # Ingesters
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
]
