"""
Base class for dataset ingestion into tantivy index.

This module provides the DatasetIngester base class with common functionality
for initializing indices, adding documents, and committing changes.
"""

from pathlib import Path
from typing import Any, Dict, List

import tantivy

from searchlm.schema import (
    FIELD_DOC_ID,
    FIELD_TEXT,
    FIELD_TITLE,
    IndexSchema,
)


class DatasetIngester:
    """
    Base class for ingesting datasets into tantivy index.
    
    Provides common functionality for:
    - Index initialization
    - Document addition with field validation
    - Index committing and reloading
    
    Subclasses should implement the `ingest()` method to define
    dataset-specific ingestion logic.
    """
    
    def __init__(self, index_path: str = "./search_index"):
        """
        Initialize the ingester with an index path.
        
        Args:
            index_path: Path where the tantivy index will be stored
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.schema = None
        self.index = None
        self.writer = None
        self.schema_builder = IndexSchema()
    
    def build_schema(self) -> tantivy.Schema:
        """
        Build the tantivy schema for indexing documents.
        
        Returns:
            Configured tantivy Schema object
        """
        return self.schema_builder.build()
    
    def initialize_index(self):
        """Initialize the tantivy index with the schema."""
        if self.schema is None:
            self.schema = self.build_schema()
        
        # Create persistent index
        if self.index_path.exists() and any(self.index_path.iterdir()):
            # Load existing index
            self.index = tantivy.Index(self.schema, path=str(self.index_path))
            print(f"Loaded existing index at {self.index_path}")
        else:
            # Create new index
            self.index = tantivy.Index(self.schema, path=str(self.index_path))
            print(f"Created new index at {self.index_path}")
        
        self.writer = self.index.writer()
    
    def _prepare_document_fields(self, doc: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Prepare document fields for tantivy indexing.
        
        Converts field values to the format expected by tantivy
        (lists of strings) and filters out empty values.
        
        Args:
            doc: Dictionary with fields matching the schema
        
        Returns:
            Dictionary with field names mapped to lists of strings (tantivy format)
        """
        doc_fields = {}
        
        # Process fields - tantivy expects text fields as lists of strings
        for field_name, value in doc.items():
            if value is not None:
                if isinstance(value, list):
                    # Filter out None and empty strings
                    filtered_value = [
                        str(v) for v in value if v is not None and str(v).strip()
                    ]
                    if filtered_value:
                        doc_fields[field_name] = filtered_value
                elif isinstance(value, str):
                    if value.strip():  # Only add non-empty strings
                        doc_fields[field_name] = [value]
                else:
                    str_value = str(value).strip()
                    if str_value:
                        doc_fields[field_name] = [str_value]
        
        return doc_fields
    
    def add_document(self, doc: Dict[str, Any]):
        """
        Add a document to the index.
        
        Documents must have at least a doc_id and either title or text
        to be added to the index.
        
        Args:
            doc: Dictionary with fields matching the schema
        """
        doc_fields = self._prepare_document_fields(doc)
        
        # Only add document if it has at least doc_id and some content
        if FIELD_DOC_ID in doc_fields and (
            doc_fields.get(FIELD_TITLE) or doc_fields.get(FIELD_TEXT)
        ):
            # Create tantivy.Document with keyword arguments
            tantivy_doc = tantivy.Document(**doc_fields)
            self.writer.add_document(tantivy_doc)
    
    def commit(self):
        """Commit the index and wait for merging to complete."""
        self.writer.commit()
        self.writer.wait_merging_threads()
        self.index.reload()
        print("Index committed and reloaded")
    
    def ingest(self):
        """
        Main ingestion method to be implemented by subclasses.
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement ingest() method")
