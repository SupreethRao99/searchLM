"""
SciFact dataset ingester.

This module provides the SciFactIngester class for ingesting
the SciFact dataset from HuggingFace into a tantivy search index.
"""

from pathlib import Path
from typing import Optional

from tqdm import tqdm

from searchlm.data.ingesters.base import DatasetIngester
from searchlm.data.loaders import SciFactLoader
from searchlm.data.schemas import (
    FIELD_DATASET,
    FIELD_DOC_ID,
    FIELD_SOURCE_ID,
    FIELD_TEXT,
    FIELD_TITLE,
)


class SciFactIngester(DatasetIngester):
    """
    Ingester for SciFact dataset from HuggingFace.

    Downloads and indexes documents from the SciFact dataset
    into a tantivy search index.
    """

    DATASET_NAME = "scifact"

    def __init__(
        self, index_path: str = "./search_index", cache_dir: Optional[Path] = None
    ):
        """
        Initialize the SciFact ingester.

        Args:
            index_path: Path where the tantivy index will be stored
            cache_dir: Optional directory to cache downloaded datasets
        """
        super().__init__(index_path)
        self.dataset_name = self.DATASET_NAME
        self.loader = SciFactLoader(cache_dir=cache_dir)

    def ingest(self):
        """
        Ingest SciFact dataset into the search index.

        Loads the corpus, converts documents to the indexing format,
        and adds them to the tantivy index.
        """
        print(f"Starting {self.DATASET_NAME} ingestion...")

        # Initialize index
        self.initialize_index()

        # Load corpus using the dataset loader
        corpus_dict = self.loader.load_corpus()

        # Convert to indexing format
        documents = []
        for doc_id, doc in tqdm(
            corpus_dict.items(), desc=f"Processing {self.DATASET_NAME} documents"
        ):
            index_doc = {
                FIELD_DOC_ID: doc.doc_id,
                FIELD_TITLE: doc.title,
                FIELD_TEXT: doc.text,
                FIELD_DATASET: self.DATASET_NAME,
                FIELD_SOURCE_ID: doc.doc_id,
            }
            documents.append(index_doc)

        print(f"Indexing {len(documents)} {self.DATASET_NAME} documents...")
        for doc in tqdm(documents, desc="Indexing documents"):
            self.add_document(doc)

        # Commit
        self.commit()
        print(f"Successfully indexed {len(documents)} {self.DATASET_NAME} documents")
