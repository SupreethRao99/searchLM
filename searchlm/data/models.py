from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Document:
    """A document from a corpus."""

    doc_id: str
    title: str
    text: str
    dataset_name: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Query:
    """A query with optional relevance judgments."""

    query_id: str
    text: str
    dataset_name: str
    qrels: Dict[str, float] = field(default_factory=dict)  # doc_id -> relevance score
    metadata: Dict = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """A dataset split containing queries, documents, and qrels."""

    name: str  # "train", "dev", or "test"
    dataset_name: str  # "nfcorpus" or "scifact"
    queries: Dict[str, Query]  # query_id -> Query
    documents: Dict[str, Document]  # doc_id -> Document
    qrels: Dict[str, Dict[str, float]]  # query_id -> {doc_id -> relevance}

    @property
    def num_queries(self) -> int:
        """Number of queries in this split."""
        return len(self.queries)

    @property
    def num_documents(self) -> int:
        """Number of documents in this split."""
        return len(self.documents)

    def get_query(self, query_id: str) -> Optional[Query]:
        """Get a query by ID."""
        return self.queries.get(query_id)

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def get_qrels_for_query(self, query_id: str) -> Dict[str, float]:
        """Get relevance judgments for a query."""
        return self.qrels.get(query_id, {})

    def get_relevant_docs(self, query_id: str, min_relevance: float = 0.5) -> List[str]:
        """Get list of relevant document IDs for a query."""
        qrels = self.get_qrels_for_query(query_id)
        return [doc_id for doc_id, score in qrels.items() if score >= min_relevance]
