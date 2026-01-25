# SearchLM

**SearchLM** - A search system for post-training LLMs with Reinforcement Learning with Verifiable Rewards (RLVR) to generate better boolean search queries.

SearchLM provides a complete toolkit for information retrieval research, featuring dataset loading, search index management, and comprehensive evaluation metrics for PubMed NFCorpus and SciFact datasets.

## Features

- 📚 **Dataset Loading**: Load NFCorpus and SciFact datasets with train/dev/test splits and relevance judgments (qrels)
- 🔍 **Full-Text Search**: Powered by Tantivy, a fast full-text search engine written in Rust
- 📊 **Comprehensive Evaluation**: Built-in IR metrics including NDCG, MRR, Precision@K, Recall@K, and MAP
- 🏗️ **Unified Index**: Single search index supporting multiple datasets
- 🎯 **Easy-to-Use API**: Simple, intuitive interface for search and evaluation workflows

## Installation

### Requirements

- Python >= 3.12

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd searchLM

# Install using uv
uv sync

# Installing the dev dependencies
uv sync --all-groups
```

## Quick Start

### 1. Ingest Datasets

First, ingest the datasets into a search index:

```python
from searchlm import ingest_all_datasets

# Ingest both NFCorpus and SciFact datasets
ingest_all_datasets(index_path="./search_index")
```

Or ingest datasets individually:

```python
from searchlm import NFCorpusIngester, SciFactIngester

# Ingest NFCorpus
nfcorpus_ingester = NFCorpusIngester(index_path="./search_index")
nfcorpus_ingester.ingest()

# Ingest SciFact
scifact_ingester = SciFactIngester(index_path="./search_index")
scifact_ingester.ingest()
```

### 2. Load Datasets

```python
from searchlm import load_dataset_split

# Load test split of NFCorpus
nfcorpus_test = load_dataset_split("nfcorpus", split="test")
print(f"Loaded {nfcorpus_test.num_queries} queries and {nfcorpus_test.num_documents} documents")

# Load test split of SciFact
scifact_test = load_dataset_split("scifact", split="test")
```

### 3. Search

```python
from searchlm import SearchEngine

# Initialize search engine
engine = SearchEngine(index_path="./search_index")

# Search across all datasets
results = engine.search("cancer treatment", limit=10)

# Search within a specific dataset
nfcorpus_results = engine.search_by_dataset("cancer treatment", dataset="nfcorpus", limit=10)

# Search with custom fields
results = engine.search(
    "cancer treatment",
    limit=10,
    fields=["title", "text"],
    dataset_filter="scifact"
)
```

### 4. Evaluate Search Quality

```python
from searchlm import SearchEvaluator

# Initialize evaluator
evaluator = SearchEvaluator(index_path="./search_index")

# Evaluate a single query
metrics = evaluator.evaluate_single_query(
    query_id="nfcorpus_1",
    query_text="cancer treatment",
    dataset_name="nfcorpus",
    split="test",
    limit=10
)
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")

# Evaluate entire dataset
batch_metrics = evaluator.evaluate_batch(
    dataset_name="nfcorpus",
    split="test",
    limit=10
)
print(f"Average NDCG@10: {batch_metrics['ndcg@10']:.4f}")
```

## Project Structure

```
searchlm/
├── searchlm/
│   ├── __init__.py          # Main package exports
│   ├── data/                # Dataset loading
│   │   ├── base_loader.py   # Base loader interface
│   │   ├── nfcorpus_loader.py
│   │   ├── scifact_loader.py
│   │   ├── factory.py       # Loader factory
│   │   └── models.py        # Data models (Document, Query, DatasetSplit)
│   ├── ingestion/           # Index ingestion
│   │   ├── base_ingester.py
│   │   ├── nfcorpus_ingester.py
│   │   ├── scifact_ingester.py
│   │   └── pipeline.py      # Batch ingestion utilities
│   ├── search_engine/       # Search functionality
│   │   ├── engine.py        # SearchEngine class
│   │   └── utils.py
│   ├── evaluation/          # Evaluation metrics
│   │   ├── evaluator.py     # SearchEvaluator class
│   │   ├── metrics.py       # Metric calculations
│   │   └── models.py        # Evaluation result models
│   ├── schema/              # Index schema
│   │   ├── index_schema.py
│   │   └── constants.py
│   └── utils/               # Utility functions
│       ├── logging_utils.py
│       └── validation.py
├── pyproject.toml
└── README.md
```

## API Overview

### Data Loading

- `load_dataset_split(dataset_name, split)`: Load a dataset split
- `create_loader(dataset_name)`: Create a dataset loader
- `DatasetLoader`: Base class for dataset loaders
- `NFCorpusLoader`, `SciFactLoader`: Dataset-specific loaders

### Search

- `SearchEngine`: Main search engine class
  - `search(query, limit, fields, dataset_filter)`: Perform search
  - `search_by_dataset(query, dataset, limit)`: Search within a dataset
  - `reload_index()`: Reload the search index

### Evaluation

- `SearchEvaluator`: Unified search evaluator
  - `evaluate_single_query()`: Evaluate a single query
  - `evaluate_single_query_with_results()`: Evaluate with search results
  - `evaluate_batch()`: Batch evaluation
  - `load_qrels()`: Load relevance judgments
  - `load_queries()`: Load queries

### Metrics

- `calculate_ndcg()`: Normalized Discounted Cumulative Gain
- `calculate_mrr()`: Mean Reciprocal Rank
- `calculate_precision_at_k()`: Precision at K
- `calculate_recall_at_k()`: Recall at K
- `calculate_map()`: Mean Average Precision

### Ingestion

- `ingest_all_datasets()`: Ingest all datasets
- `NFCorpusIngester`, `SciFactIngester`: Dataset-specific ingesters

## Supported Datasets

- **NFCorpus**: A full-text retrieval dataset for medical domain
- **SciFact**: A scientific fact-checking dataset

Both datasets are loaded from the Hugging Face `mteb` library and include train/dev/test splits with relevance judgments.

## Evaluation Metrics

SearchLM supports standard information retrieval metrics:

- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **MRR**: Mean Reciprocal Rank
- **Precision@K**: Precision at K
- **Recall@K**: Recall at K
- **MAP**: Mean Average Precision

## License

MIT License - see LICENSE file for details.

## Author

**Supreeth Rao** - raosupreeth00@gmail.com

## Keywords

search, information retrieval, reinforcement learning, verifiable rewards, RLVR, tantivy, full-text search
