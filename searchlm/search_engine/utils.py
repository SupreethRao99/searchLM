"""
Utility functions for search engine operations.

This module provides helper functions for working with tantivy
documents and extracting field values safely.
"""

import tantivy


def get_field_from_doc(
    doc: tantivy.Document,
    field_name: str,
    default: str = ""
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
