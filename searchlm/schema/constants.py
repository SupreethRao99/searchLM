"""
Constants for field names and configuration used in the search index.
"""

# Field names used in the index
FIELD_DOC_ID = "doc_id"
FIELD_TITLE = "title"
FIELD_TEXT = "text"
FIELD_DATASET = "dataset"
FIELD_SOURCE_ID = "source_id"

# Searchable fields (used for query parsing)
SEARCHABLE_FIELDS = [FIELD_TITLE, FIELD_TEXT]
