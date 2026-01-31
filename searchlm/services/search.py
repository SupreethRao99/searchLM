"""
Search engine for querying the tantivy index.

This module provides the SearchEngine class for performing full-text
searches on indexed documents.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import tantivy

from searchlm.data.schemas import (FIELD_DATASET, FIELD_DOC_ID,
                                   FIELD_SOURCE_ID, FIELD_TEXT, FIELD_TITLE,
                                   IndexSchema)


def get_field_from_doc(
    doc: tantivy.Document, field_name: str, default: str = ""
) -> str:
    """
    Safely extract a field value from a tantivy Document.

    Args:
        doc: tantivy Document object
        field_name: Name of the field to extract
        default: Default value if field is missing or empty

    Returns:
        Field value as string, or default if not found
    """
    try:
        field_value = doc[field_name]
        return field_value[0] if field_value else default
    except (KeyError, IndexError):
        return default


class SearchEngine:
    """
    Search engine for querying the tantivy index.

    Provides methods for:
    - Full-text search across indexed documents
    - Filtering by dataset
    - Configurable search fields
    - Result ranking and scoring
    """

    def __init__(self, index_path: str = "./search_index"):
        """
        Initialize the search engine.

        Args:
            index_path: Path to the tantivy index

        Raises:
            ValueError: If index path doesn't exist
        """
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise ValueError(
                f"Index not found at {index_path}. Please run ingestion first."
            )

        # Build schema (must match the ingestion schema)
        self.schema_builder = IndexSchema()
        self.schema = self.schema_builder.build()
        self.index = tantivy.Index(self.schema, path=str(self.index_path))
        self.index.reload()
        self.searcher = self.index.searcher()

    def search(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[List[str]] = None,
        dataset_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the index.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            fields: Fields to search in (default: ["title", "text"])
            dataset_filter: Filter by dataset name ("nfcorpus" or "scifact")

        Returns:
            List of search results, each containing:
            - score: Relevance score
            - doc_id: Document identifier
            - title: Document title
            - text: Document text
            - dataset: Dataset name
            - source_id: Original source ID
        """
        if fields is None:
            fields = IndexSchema.get_searchable_fields()

        # Parse query
        tantivy_query = self.index.parse_query(query, fields)

        # Search
        search_results = self.searcher.search(tantivy_query, limit=limit)

        # Process results
        results = []
        for score, doc_address in search_results.hits:
            doc = self.searcher.doc(doc_address)

            result = {
                "score": score,
                FIELD_DOC_ID: get_field_from_doc(doc, FIELD_DOC_ID),
                FIELD_TITLE: get_field_from_doc(doc, FIELD_TITLE),
                FIELD_TEXT: get_field_from_doc(doc, FIELD_TEXT),
                FIELD_DATASET: get_field_from_doc(doc, FIELD_DATASET),
                FIELD_SOURCE_ID: get_field_from_doc(doc, FIELD_SOURCE_ID),
            }

            # Apply dataset filter if specified
            if dataset_filter is None or result[FIELD_DATASET] == dataset_filter:
                results.append(result)

        return results

    def search_by_dataset(
        self, query: str, dataset: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific dataset.

        Args:
            query: Search query string
            dataset: Dataset name ("nfcorpus" or "scifact")
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        return self.search(query, limit=limit, dataset_filter=dataset)

    def reload_index(self):
        """
        Reload the search index.

        Useful when the index has been updated and you want to
        refresh the search engine without creating a new instance.
        """
        self.index.reload()
        self.searcher = self.index.searcher()
