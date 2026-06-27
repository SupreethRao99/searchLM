"""
Ingestion pipeline utilities.

This module provides convenience functions for ingesting multiple
datasets into a single search index.
"""

from searchlm.data.ingesters.arguana import ArguAnaIngester
from searchlm.data.ingesters.fiqa import FiQAIngester
from searchlm.data.ingesters.hotpotqa import HotpotQAIngester
from searchlm.data.ingesters.nfcorpus import NFCorpusIngester
from searchlm.data.ingesters.nq import NQIngester
from searchlm.data.ingesters.scifact import SciFactIngester

# Ordered by corpus size (small → large) so the index grows incrementally.
# HotpotQA (~5M docs) and NQ (~2.7M docs) run last and take the longest.
_INGESTERS = [
    ("nfcorpus", NFCorpusIngester),
    ("scifact", SciFactIngester),
    ("fiqa", FiQAIngester),
    ("arguana", ArguAnaIngester),
    ("nq", NQIngester),
    ("hotpotqa", HotpotQAIngester),
]


def ingest_all_datasets(index_path: str = "./search_index", datasets: list[str] | None = None):
    """
    Ingest all supported datasets into a single unified Tantivy index.

    Args:
        index_path: Path where the Tantivy index will be stored.
        datasets: Optional list of dataset names to ingest. Defaults to all.
                  Example: ["nfcorpus", "scifact"] to skip the large corpora.
    """
    print("=" * 60)
    print("Starting ingestion pipeline")
    print("=" * 60)

    targets = [
        (name, cls) for name, cls in _INGESTERS
        if datasets is None or name in datasets
    ]

    for i, (name, ingester_cls) in enumerate(targets):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        ingester = ingester_cls(index_path=index_path)
        ingester.ingest()

    print("\n" + "=" * 60)
    print("Ingestion pipeline completed successfully!")
    print(f"Index location: {index_path}")
    print("=" * 60)
