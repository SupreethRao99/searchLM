# SearchLM Usage Guide

This guide provides detailed documentation for using SearchLM, a search system for post-training LLMs with Reinforcement Learning with Verifiable Rewards (RLVR) to generate better boolean search queries.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Dataset Loading](#dataset-loading)
4. [Dataset Ingestion](#dataset-ingestion)
5. [Search Operations](#search-operations)
6. [Evaluation](#evaluation)
7. [Baseline Query Generation](#baseline-query-generation)
8. [RLHF Training Workflow](#rlhf-training-workflow)
9. [Advanced Usage](#advanced-usage)

## Overview

SearchLM provides a complete toolkit for information retrieval research with the following components:

- **Dataset Loading**: Load NFCorpus and SciFact datasets from HuggingFace MTEB with train/dev/test splits and relevance judgments
- **Search Index**: Tantivy-based full-text search engine for indexing and querying documents
- **Evaluation**: Comprehensive IR metrics including NDCG, MRR, Precision@K, Recall@K, and MAP
- **Query Generation**: Baseline query generation using instruction-tuned LLMs
- **RLHF Training**: Reinforcement learning workflow for training models to generate better search queries

## Configuration

SearchLM uses a YAML-based configuration system. The default configuration is located at `config/default.yaml`. You can load and customize configuration using the config module:

```python
from searchlm import load_config, get_config, merge_config

# Load default configuration
config = load_config()

# Get the current configuration
config = get_config()

# Merge custom overrides
config = merge_config({
    "model": {"name": "custom-model-name"},
    "training": {"learning_rate": 2e-6}
})
```

### Configuration Structure

The configuration file includes:

- **model**: Model name, max tokens, temperature, and max model length
- **training**: Learning rate, epochs, batch sizes, gradient accumulation, precision, logging settings
- **reward**: NDCG and MRR weights for reward function, evaluation split and k
- **evaluation**: Temperature, max tokens, default k, and datasets to evaluate
- **baseline**: Batch sizes and max tokens for baseline query generation
- **infrastructure**: Python version and GPU requirements
- **paths**: Directories for models, data, indices, and output
- **datasets**: Supported dataset names and splits

### Environment Variables

Create a `.env` file from `.env.example` and configure:

- `HF_TOKEN`: HuggingFace token (required for downloading models and datasets)
- `WANDB_API_KEY`: Weights & Biases API key (optional, for training metrics logging)
- Optional custom paths: `HF_HOME`, `DATA_DIR`, `MODELS_DIR`, `INDEX_DIR`, `OUTPUT_DIR`

## Dataset Loading

SearchLM supports loading datasets from HuggingFace MTEB. Currently supported datasets are NFCorpus and SciFact.

### Loading a Dataset Split

The simplest way to load a dataset is using the convenience function:

```python
from searchlm import load_dataset_split

# Load test split of NFCorpus
nfcorpus_test = load_dataset_split("nfcorpus", split="test")
print(f"Loaded {nfcorpus_test.num_queries} queries")
print(f"Loaded {nfcorpus_test.num_documents} documents")

# Load dev split of SciFact
scifact_dev = load_dataset_split("scifact", split="dev")
```

### Using Dataset Loaders Directly

For more control, use the loader classes directly:

```python
from searchlm import create_loader, NFCorpusLoader, SciFactLoader

# Using factory function
loader = create_loader("nfcorpus")
dataset_split = loader.load_split(split="test")

# Using loader class directly
nfcorpus_loader = NFCorpusLoader()
scifact_loader = SciFactLoader()

# Load specific components
documents = nfcorpus_loader.load_corpus()
queries = nfcorpus_loader.load_queries(split="test")
qrels = nfcorpus_loader.load_qrels(split="test")
```

### DatasetSplit Object

The `DatasetSplit` object contains:

- `queries`: Dictionary mapping query_id -> Query
- `documents`: Dictionary mapping doc_id -> Document
- `qrels`: Dictionary mapping query_id -> {doc_id -> relevance_score}
- `name`: Split name ("train", "dev", or "test")
- `dataset_name`: Dataset name ("nfcorpus" or "scifact")

### Accessing Data

```python
# Get a specific query
query = dataset_split.get_query("query_id_123")
print(f"Query text: {query.text}")
print(f"Relevance judgments: {query.qrels}")

# Get a specific document
document = dataset_split.get_document("doc_id_456")
print(f"Title: {document.title}")
print(f"Text: {document.text}")

# Get relevant documents for a query
relevant_docs = dataset_split.get_relevant_docs("query_id_123", min_relevance=0.5)
```

## Dataset Ingestion

Before you can search, you need to ingest datasets into a Tantivy search index. The index stores documents with fields for title, text, dataset name, and document identifiers.

### Ingesting All Datasets

The simplest approach is to ingest all supported datasets:

```python
from searchlm import ingest_all_datasets

# Ingest both NFCorpus and SciFact into a unified index
ingest_all_datasets(index_path="./search_index")
```

### Ingesting Individual Datasets

You can also ingest datasets individually:

```python
from searchlm import NFCorpusIngester, SciFactIngester

# Ingest NFCorpus
nfcorpus_ingester = NFCorpusIngester(index_path="./search_index")
nfcorpus_ingester.ingest()

# Ingest SciFact (will add to existing index)
scifact_ingester = SciFactIngester(index_path="./search_index")
scifact_ingester.ingest()
```

### Index Schema

The search index uses the following schema:

- `doc_id`: Document identifier (stored, raw tokenizer)
- `title`: Document title (stored, English stemmer)
- `text`: Document content (stored, English stemmer)
- `dataset`: Dataset name (stored, raw tokenizer)
- `source_id`: Original source ID (stored, raw tokenizer)

The title and text fields use English stemming for better search quality. The default searchable fields are `["title", "text"]`.

## Search Operations

Once datasets are ingested, you can perform searches using the `SearchEngine` class.

### Basic Search

```python
from searchlm import SearchEngine

# Initialize search engine
engine = SearchEngine(index_path="./search_index")

# Search across all datasets
results = engine.search("cancer treatment", limit=10)

# Each result contains:
# - score: Relevance score
# - doc_id: Document identifier
# - title: Document title
# - text: Document text
# - dataset: Dataset name
# - source_id: Original source ID

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Title: {result['title']}")
    print(f"Dataset: {result['dataset']}")
    print()
```

### Search Within a Dataset

```python
# Search only within NFCorpus
nfcorpus_results = engine.search_by_dataset(
    "cancer treatment", 
    dataset="nfcorpus", 
    limit=10
)

# Or use dataset_filter parameter
results = engine.search(
    "cancer treatment",
    limit=10,
    dataset_filter="scifact"
)
```

### Custom Search Fields

```python
# Search only in title field
results = engine.search(
    "cancer treatment",
    limit=10,
    fields=["title"]
)

# Search in both title and text (default)
results = engine.search(
    "cancer treatment",
    limit=10,
    fields=["title", "text"]
)
```

### Reloading the Index

If the index is updated, reload it:

```python
engine.reload_index()
```

## Evaluation

SearchLM provides comprehensive evaluation capabilities using standard IR metrics.

### Single Query Evaluation

Evaluate a single query against ground truth relevance judgments:

```python
from searchlm import SearchEvaluator

evaluator = SearchEvaluator(index_path="./search_index")

# Evaluate a query by ID
metrics, error = evaluator.evaluate_single_query(
    query_text="cancer treatment",
    query_id="nfcorpus_1",
    dataset_name="nfcorpus",
    split="test",
    k=100
)

if error:
    print(f"Error: {error}")
else:
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"NDCG@100: {metrics['ndcg@100']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Precision@10: {metrics['precision@10']:.4f}")
    print(f"Recall@10: {metrics['recall@10']:.4f}")
    print(f"MAP: {metrics['map']:.4f}")
```

### Evaluation with Results

Get both metrics and search results:

```python
query_result, error = evaluator.evaluate_single_query_with_results(
    query_text="cancer treatment",
    query_id="nfcorpus_1",
    dataset_name="nfcorpus",
    split="test",
    k=100
)

if error:
    print(f"Error: {error}")
else:
    print(f"Metrics: {query_result.metrics}")
    print(f"Retrieved: {query_result.num_retrieved} documents")
    print(f"Relevant: {query_result.num_relevant_total} documents")
    
    # Access individual results
    for result in query_result.results:
        print(f"Rank {result.rank}: {result.title}")
        print(f"  Relevance: {result.relevance}")
        print(f"  Score: {result.score}")
```

### Batch Evaluation

Evaluate multiple queries efficiently:

```python
# Evaluate a list of (query_text, query_id) tuples
queries = [
    ("cancer treatment", "nfcorpus_1"),
    ("diabetes research", "nfcorpus_2"),
    ("heart disease", "nfcorpus_3"),
]

results = evaluator.evaluate_batch(
    queries=queries,
    dataset_name="nfcorpus",
    split="test",
    k=100,
    show_progress=True
)

print(f"Average NDCG@10: {results['ndcg@10']:.4f}")
print(f"Average MRR: {results['mrr']:.4f}")
print(f"Number of queries: {results['num_queries']}")
print(f"Failed queries: {results['num_failed']}")
```

### Full Dataset Evaluation

Evaluate an entire dataset split:

```python
results = evaluator.evaluate(
    dataset_name="nfcorpus",
    split="test",
    k=100,
    max_queries=None  # None for all queries
)

# Print formatted results
evaluator.print_metrics(results)
```

### Loading Queries and Qrels

You can also load queries and relevance judgments separately:

```python
# Load all queries for a split
queries = evaluator.load_queries("nfcorpus", split="test")
# Returns: Dict[str, str] mapping query_id -> query_text

# Load all qrels for a split
qrels = evaluator.load_qrels("nfcorpus", split="test")
# Returns: Dict[str, Dict[str, float]] mapping query_id -> {doc_id -> relevance}
```

### Evaluation Metrics

SearchLM supports the following IR metrics:

- **NDCG@K**: Normalized Discounted Cumulative Gain at K - measures ranking quality considering position
- **MRR**: Mean Reciprocal Rank - measures the rank of the first relevant document
- **Precision@K**: Proportion of relevant documents in top K results
- **Recall@K**: Proportion of relevant documents retrieved out of all relevant documents
- **MAP**: Mean Average Precision - considers both precision and position of relevant documents

You can also use the metric functions directly:

```python
from searchlm import (
    calculate_ndcg,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_map
)

# Example: Calculate NDCG for a ranked list
relevance_scores = [1.0, 0.5, 0.0, 1.0, 0.0]  # Relevance for each result
ndcg_10 = calculate_ndcg(relevance_scores, k=10)
mrr = calculate_mrr(relevance_scores)
```

## Baseline Query Generation

The baseline workflow generates search queries using instruction-tuned LLMs without reinforcement learning. This provides a baseline for comparison with RLHF-trained models.

### Running Baseline Generation

```bash
# Generate queries for SciFact (default)
uv run python -m searchlm.workflows.baseline.cli

# Generate queries for NFCorpus
uv run python -m searchlm.workflows.baseline.cli \
  --dataset-name mteb/nfcorpus \
  --output-filename nfcorpus_generated_queries.tsv

# Use custom batch size
uv run python -m searchlm.workflows.baseline.cli \
  --batch-size 50
```

### Baseline Generation Process

The baseline workflow:

1. Loads queries from the specified dataset
2. Uses vLLM to generate boolean search queries using the prompt template
3. Extracts queries from the model output (between `<query>` tags)
4. Saves results to a TSV file with columns: `id`, `query`, `original_query`

### Evaluating Baseline Queries

After generating baseline queries, you can evaluate them:

```python
import pandas as pd
import re
from searchlm import SearchEvaluator

# Load generated queries
df = pd.read_csv("./data/scifact_generated_queries.tsv", sep="\t")

# Extract queries from model output
df["cleaned_query"] = df["query"].str.extract(
    r"<query>\s*(.*?)\s*</query>", flags=re.DOTALL
)
df = df.dropna(subset=["cleaned_query"])
df["cleaned_query"] = df["cleaned_query"].str.strip()

# Evaluate
evaluator = SearchEvaluator()
results = evaluator.evaluate_batch(
    queries=zip(df["cleaned_query"], df["id"]),
    dataset_name="scifact",
    split="test",
    k=100
)

print(results)
```

## RLHF Training Workflow

The RLHF workflow trains models using Group Relative Policy Optimization (GRPO) with verifiable rewards based on search evaluation metrics.

### Workflow Overview

The RLHF workflow consists of three main steps:

1. **Data Preparation**: Prepare training data from datasets
2. **Training**: Train the model using GRPO
3. **Evaluation**: Evaluate the trained model

### Data Preparation

Prepare training data from the datasets:

```bash
uv run python -m searchlm.workflows.rlhf.cli prep
```

This step prepares the training data according to the configuration settings.

### Training

Train the model using GRPO. There are two modes:

**Colocate Mode (1 GPU)** - Recommended for single GPU setups:

```bash
uv run python -m searchlm.workflows.rlhf.cli train
```

**Server Mode (2 GPUs)** - For multi-GPU setups with separate vLLM server:

```bash
uv run python -m searchlm.workflows.rlhf.cli train --use-vllm-server
```

Training configuration is controlled by `config/default.yaml`:

- Model settings: model name, max tokens, temperature
- Training hyperparameters: learning rate, epochs, batch sizes, gradient accumulation
- Reward function: NDCG and MRR weights, evaluation split and k
- Logging: Weights & Biases integration, logging steps, save steps

### Evaluation

Evaluate a trained model:

```bash
# Evaluate latest checkpoint
uv run python -m searchlm.workflows.rlhf.cli eval

# Evaluate specific checkpoint
uv run python -m searchlm.workflows.rlhf.cli eval \
  --checkpoint-path ./models/checkpoint-500

# Compare with baseline
uv run python -m searchlm.workflows.rlhf.cli eval --compare-baseline
```

The evaluation runs on both NFCorpus and SciFact test sets and reports comprehensive metrics.

## Advanced Usage

### Custom Dataset Loaders

You can create custom dataset loaders by extending `DatasetLoader`:

```python
from searchlm.data.loaders.base import DatasetLoader
from searchlm.models import Document, Query

class CustomLoader(DatasetLoader):
    DATASET_NAME = "custom"
    DATASET_SOURCE = "huggingface/dataset"
    
    def load_corpus(self) -> Dict[str, Document]:
        # Implement corpus loading
        pass
    
    def load_queries(self, split: str = "test", qrels: Optional[Dict] = None) -> Dict[str, Query]:
        # Implement query loading
        pass
    
    def load_qrels(self, split: str = "test") -> Dict[str, Dict[str, float]]:
        # Implement qrels loading
        pass
```

### Custom Dataset Ingesters

Create custom ingesters by extending `DatasetIngester`:

```python
from searchlm.data.ingesters.base import DatasetIngester

class CustomIngester(DatasetIngester):
    DATASET_NAME = "custom"
    
    def ingest(self):
        # Initialize index
        self.initialize_index()
        
        # Load documents
        documents = self.loader.load_corpus()
        
        # Add documents to index
        for doc in documents.values():
            self.add_document({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "dataset": doc.dataset_name,
                "source_id": doc.metadata.get("source_id", doc.doc_id)
            })
        
        # Commit index
        self.commit()
```

### Using vLLM Engine Directly

For custom inference needs, use the `VllmEngine` class:

```python
from searchlm.inference import VllmEngine
from searchlm.prompts import SYSTEM_PROMPT, format_user_prompt

# Use as context manager
with VllmEngine() as engine:
    prompts = [
        format_user_prompt("What is cancer treatment?")
    ]
    
    # Generate queries
    outputs = engine.generate(
        prompts=prompts,
        temperature=0.7,
        max_tokens=1024
    )
    
    print(outputs[0])
```

### Custom Search Queries

Tantivy supports boolean query syntax. You can use:

- **AND/OR operators**: `(term1 AND term2) OR term3`
- **+/- operators**: `+required -excluded`
- **Phrase search**: `"exact phrase"`
- **Field-specific search**: Use the `fields` parameter to search specific fields

Example:

```python
# Boolean query
results = engine.search("(cancer AND treatment) OR therapy", limit=10)

# Phrase search
results = engine.search('"cancer treatment"', limit=10)

# Required and excluded terms
results = engine.search("+cancer +treatment -surgery", limit=10)
```

### Programmatic Configuration

You can programmatically modify configuration:

```python
from searchlm import merge_config

# Override specific settings
config = merge_config({
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0.5
    },
    "evaluation": {
        "default_k": 50
    }
})

# Use in your code
from searchlm.config import get_config
config = get_config()
print(config.model.name)
```

### Error Handling

The evaluation methods return error messages for failed queries:

```python
metrics, error = evaluator.evaluate_single_query(
    query_text="invalid query syntax",
    query_id="test_1",
    dataset_name="nfcorpus",
    split="test"
)

if error:
    print(f"Query failed: {error}")
    # Handle error appropriately
else:
    # Use metrics
    print(metrics)
```

### Batch Processing

For large-scale evaluation, use batch processing:

```python
from searchlm import SearchEvaluator

evaluator = SearchEvaluator()

# Load all queries
queries = evaluator.load_queries("nfcorpus", split="test")

# Convert to list of tuples
query_list = [(text, qid) for qid, text in queries.items()]

# Evaluate in batches
results = evaluator.evaluate_batch(
    queries=query_list,
    dataset_name="nfcorpus",
    split="test",
    k=100,
    show_progress=True
)
```

## Troubleshooting

### Index Not Found

If you get an error about the index not being found:

```python
# Make sure you've ingested datasets first
from searchlm import ingest_all_datasets
ingest_all_datasets(index_path="./search_index")
```

### Query Syntax Errors

Tantivy may fail on malformed queries. The evaluator catches these errors and returns them in the error message. Check the error message for details about the syntax issue.

### Memory Issues

For large datasets, you may encounter memory issues. Consider:

- Processing queries in smaller batches
- Using `max_queries` parameter in evaluation
- Reducing the `k` parameter for retrieval

### GPU Requirements

- **Training**: Requires H100 or A100 (1-2 GPUs)
- **Evaluation**: Requires H100 or A100 (1 GPU)
- **Baseline inference**: Requires L4, T4, or similar (1 GPU)

Make sure you have the appropriate GPU resources for your workflow.

## Best Practices

1. **Index Management**: Ingest all datasets into a single unified index for cross-dataset search capabilities
2. **Evaluation**: Always evaluate on test splits to avoid data leakage
3. **Query Generation**: Use the baseline workflow to establish a performance baseline before RLHF training
4. **Configuration**: Keep configuration in `config/default.yaml` and use environment variables for sensitive data
5. **Error Handling**: Always check for errors when evaluating queries, especially for generated queries
6. **Progress Tracking**: Use `show_progress=True` for batch operations to monitor progress
7. **Checkpoint Management**: Save and evaluate checkpoints regularly during training to track progress
