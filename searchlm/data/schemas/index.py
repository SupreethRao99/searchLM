"""
Schema builder for tantivy search index.

This module provides the IndexSchema class for building and managing
the tantivy schema used for document indexing.
"""

from typing import List

import tantivy

from searchlm.data.schemas.constants import (
    FIELD_DATASET,
    FIELD_DOC_ID,
    FIELD_SOURCE_ID,
    FIELD_TEXT,
    FIELD_TITLE,
    SEARCHABLE_FIELDS,
)


class IndexSchema:
    """
    Manages the tantivy schema for the search index.

    This class encapsulates schema building logic and provides
    a consistent interface for creating tantivy schemas.
    """

    def __init__(self):
        """Initialize the schema builder."""
        self._schema = None

    def build(self) -> tantivy.Schema:
        """
        Build the tantivy schema for indexing documents.

        The schema defines:
        - doc_id: Document identifier (stored, raw tokenizer)
        - title: Document title (stored, English stemmer)
        - text: Document content (stored, English stemmer)
        - dataset: Dataset name (stored, raw tokenizer)
        - source_id: Original source ID (stored, raw tokenizer)

        Returns:
            Configured tantivy Schema object
        """
        if self._schema is None:
            schema_builder = tantivy.SchemaBuilder()

            # Document identifier - stored as-is without tokenization
            schema_builder.add_text_field(
                FIELD_DOC_ID, stored=True, tokenizer_name="raw"
            )

            # Title field - stored with English stemming for better search
            schema_builder.add_text_field(
                FIELD_TITLE, stored=True, tokenizer_name="en_stem"
            )

            # Text content field - stored with English stemming
            schema_builder.add_text_field(
                FIELD_TEXT, stored=True, tokenizer_name="en_stem"
            )

            # Dataset identifier - stored as-is
            schema_builder.add_text_field(
                FIELD_DATASET, stored=True, tokenizer_name="raw"
            )

            # Source ID - original document ID from source dataset
            schema_builder.add_text_field(
                FIELD_SOURCE_ID, stored=True, tokenizer_name="raw"
            )

            self._schema = schema_builder.build()

        return self._schema

    @staticmethod
    def get_searchable_fields() -> List[str]:
        """
        Get the list of fields that should be searched by default.

        Returns:
            List of field names to search
        """
        return SEARCHABLE_FIELDS.copy()
