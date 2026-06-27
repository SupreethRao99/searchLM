"""
Data module for SearchLM.

This module provides data loading, ingestion, and schema management
for IR datasets.
"""

# Ingesters
from searchlm.data.ingesters import (ArguAnaIngester, DatasetIngester,
                                     FiQAIngester, HotpotQAIngester,
                                     NFCorpusIngester, NQIngester,
                                     SciFactIngester, ingest_all_datasets)
# Loaders
from searchlm.data.loaders import (ArguAnaLoader, DatasetLoader, FiQALoader,
                                   HotpotQALoader, NFCorpusLoader, NQLoader,
                                   SciFactLoader, create_loader)
# Schemas
from searchlm.data.schemas import (FIELD_DATASET, FIELD_DOC_ID,
                                   FIELD_SOURCE_ID, FIELD_TEXT, FIELD_TITLE,
                                   SEARCHABLE_FIELDS, IndexSchema)

__all__ = [
    # Loaders
    "DatasetLoader",
    "NFCorpusLoader",
    "SciFactLoader",
    "FiQALoader",
    "ArguAnaLoader",
    "HotpotQALoader",
    "NQLoader",
    "create_loader",
    # Ingesters
    "DatasetIngester",
    "NFCorpusIngester",
    "SciFactIngester",
    "FiQAIngester",
    "ArguAnaIngester",
    "HotpotQAIngester",
    "NQIngester",
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
