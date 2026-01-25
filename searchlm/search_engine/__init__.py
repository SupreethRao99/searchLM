"""
Search engine module for querying the tantivy index.

Provides the SearchEngine class for full-text search and utilities
for working with search results.
"""

from searchlm.search_engine.engine import SearchEngine
from searchlm.search_engine.utils import get_field_from_doc

__all__ = [
    "SearchEngine",
    "get_field_from_doc",
]
