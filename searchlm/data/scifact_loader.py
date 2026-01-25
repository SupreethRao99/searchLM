from typing import Dict, Optional

from datasets import load_dataset
from tqdm import tqdm

from searchlm.data.base_loader import DatasetLoader
from searchlm.data.models import Document, Query


class SciFactLoader(DatasetLoader):
    """Loader for SciFact dataset from HuggingFace."""

    DATASET_NAME = "scifact"
    DATASET_SOURCE = "mteb/scifact"

    def load_corpus(self) -> Dict[str, Document]:
        """Load SciFact documents."""
        print("Loading SciFact corpus...")

        corpus_dataset = load_dataset(
            self.DATASET_SOURCE, "corpus", cache_dir=self.cache_dir
        )
        corpus = corpus_dataset[list(corpus_dataset.keys())[0]]

        documents = {}
        for item in tqdm(corpus, desc="Processing documents"):
            # Try multiple possible field name variations
            doc_id = str(
                item.get("id")
                or item.get("_id")
                or item.get("corpus-id")
                or item.get("doc_id")
                or ""
            )
            if not doc_id:
                continue

            # Try different field names for title
            title = item.get("title") or item.get("Title") or ""

            # Try abstract, text, or content fields
            text = (
                item.get("abstract")
                or item.get("text")
                or item.get("content")
                or item.get("Abstract")
                or ""
            )

            doc = Document(
                doc_id=doc_id,
                title=title,
                text=text,
                dataset_name=self.DATASET_NAME,
                metadata={"source_id": doc_id},
            )
            documents[doc_id] = doc

        return documents

    def load_queries(
        self, split: str = "test", qrels: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Query]:
        """
        Load SciFact queries for a specific split.

        Note: The 'queries' subset contains ALL queries in a single split.
        We load all queries, then filter by which ones have qrels in the target split.

        Args:
            split: Dataset split name (used for filtering and metadata)
            qrels: Optional pre-loaded qrels to filter queries. If None, will load qrels.

        Returns:
            Dictionary of Query objects for this split
        """
        print(f"Loading SciFact queries ({split} split)...")

        # Load ALL queries from the "queries" split
        queries_dataset = load_dataset(
            self.DATASET_SOURCE, "queries", cache_dir=self.cache_dir
        )
        # The queries subset has a single split (usually called "queries")
        queries_split_name = list(queries_dataset.keys())[0]
        all_queries_data = queries_dataset[queries_split_name]

        # Build dict of all queries
        all_queries = {}
        for item in tqdm(all_queries_data, desc="Loading all queries"):
            query_id = str(item.get("_id") or item.get("id", ""))
            query_text = item.get("text", item.get("query", ""))

            if not query_id or not query_text:
                continue

            all_queries[query_id] = query_text

        # Load qrels if not provided
        if qrels is None:
            qrels = self.load_qrels(split=split)

        # Filter to only queries that have qrels in this split
        queries = {}
        for query_id in qrels.keys():
            if query_id in all_queries:
                queries[query_id] = Query(
                    query_id=query_id,
                    text=all_queries[query_id],
                    dataset_name=self.DATASET_NAME,
                    metadata={"split": split},
                )

        print(f"Filtered to {len(queries)} queries for {split} split")
        return queries

    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, float]]:
        """Load SciFact relevance judgments for a specific split."""
        print(f"Loading SciFact qrels ({split} split)...")

        qrels_dataset = load_dataset(
            self.DATASET_SOURCE, "default", split=split, cache_dir=self.cache_dir
        )

        qrels = {}
        for item in tqdm(qrels_dataset, desc="Processing qrels"):
            query_id = str(item.get("query-id", item.get("query_id", "")))
            doc_id = str(
                item.get("corpus-id", item.get("corpus_id", item.get("doc-id", "")))
            )
            score = float(item.get("score", item.get("relevance", 0.0)))

            if not query_id or not doc_id:
                continue

            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score

        return qrels
