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
        documents = self.prepare_documents(corpus_dict, self.DATASET_NAME)

        print(f"Indexing {len(documents)} {self.DATASET_NAME} documents...")
        for doc in tqdm(documents, desc="Indexing documents"):
            self.add_document(doc)

        # Commit
        self.commit()
        print(f"Successfully indexed {len(documents)} {self.DATASET_NAME} documents")
