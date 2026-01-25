"""
Ingestion module for loading datasets into tantivy search index.

Provides ingesters for different datasets and pipeline utilities
for batch ingestion.
"""

from searchlm.ingestion.base_ingester import DatasetIngester
from searchlm.ingestion.nfcorpus_ingester import NFCorpusIngester
from searchlm.ingestion.pipeline import ingest_all_datasets
from searchlm.ingestion.scifact_ingester import SciFactIngester

__all__ = [
    "DatasetIngester",
    "NFCorpusIngester",
    "SciFactIngester",
    "ingest_all_datasets",
]
