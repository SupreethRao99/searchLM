"""
Dataset ingesters for tantivy search index.

This module provides ingesters for various IR datasets.
"""

from searchlm.data.ingesters.base import DatasetIngester
from searchlm.data.ingesters.nfcorpus import NFCorpusIngester
from searchlm.data.ingesters.pipeline import ingest_all_datasets
from searchlm.data.ingesters.scifact import SciFactIngester

__all__ = [
    "DatasetIngester",
    "NFCorpusIngester",
    "SciFactIngester",
    "ingest_all_datasets",
]
