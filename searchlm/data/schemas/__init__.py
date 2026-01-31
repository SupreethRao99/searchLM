"""
Schema module for tantivy search index.

Provides constants, schema builders, and utilities for managing
the search index schema.
"""

from searchlm.data.schemas.constants import (FIELD_DATASET, FIELD_DOC_ID,
                                             FIELD_SOURCE_ID, FIELD_TEXT,
                                             FIELD_TITLE, SEARCHABLE_FIELDS)
from searchlm.data.schemas.index import IndexSchema

__all__ = [
    # Constants
    "FIELD_DOC_ID",
    "FIELD_TITLE",
    "FIELD_TEXT",
    "FIELD_DATASET",
    "FIELD_SOURCE_ID",
    "SEARCHABLE_FIELDS",
    # Classes
    "IndexSchema",
]
