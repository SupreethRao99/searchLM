"""
Dataset ingesters for tantivy search index.

This module provides ingesters for various IR datasets.
"""

from searchlm.data.ingesters.arguana import ArguAnaIngester
from searchlm.data.ingesters.base import DatasetIngester
from searchlm.data.ingesters.fiqa import FiQAIngester
from searchlm.data.ingesters.hotpotqa import HotpotQAIngester
from searchlm.data.ingesters.nfcorpus import NFCorpusIngester
from searchlm.data.ingesters.nq import NQIngester
from searchlm.data.ingesters.pipeline import ingest_all_datasets
from searchlm.data.ingesters.scifact import SciFactIngester

__all__ = [
    "DatasetIngester",
    "NFCorpusIngester",
    "SciFactIngester",
    "FiQAIngester",
    "ArguAnaIngester",
    "HotpotQAIngester",
    "NQIngester",
    "ingest_all_datasets",
]
