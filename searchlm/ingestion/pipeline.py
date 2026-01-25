"""
Ingestion pipeline utilities.

This module provides convenience functions for ingesting multiple
datasets into a single search index.
"""

from searchlm.ingestion.nfcorpus_ingester import NFCorpusIngester
from searchlm.ingestion.scifact_ingester import SciFactIngester


def ingest_all_datasets(index_path: str = "./search_index"):
    """
    Ingest both NFCorpus and SciFact datasets into a single index.
    
    This function sequentially ingests both datasets, allowing
    them to be searched together in a unified index.
    
    Args:
        index_path: Path where the tantivy index will be stored
    """
    print("=" * 60)
    print("Starting ingestion pipeline for all datasets")
    print("=" * 60)
    
    # Ingest NFCorpus
    nfcorpus_ingester = NFCorpusIngester(index_path=index_path)
    nfcorpus_ingester.ingest()
    
    print("\n" + "=" * 60 + "\n")
    
    # Ingest SciFact (will append to the same index)
    scifact_ingester = SciFactIngester(index_path=index_path)
    scifact_ingester.ingest()
    
    print("\n" + "=" * 60)
    print("Ingestion pipeline completed successfully!")
    print(f"Index location: {index_path}")
    print("=" * 60)
