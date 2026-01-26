# SearchLM

SearchLM is a search system for post-training LLMs with Reinforcement Learning with Verifiable Rewards (RLVR) to generate better boolean search queries. It provides a complete toolkit for information retrieval research, featuring dataset loading, search index management, and comprehensive evaluation metrics.

## Philosophy

SearchLM is designed to be **simple, hackable, and easy to use**:

- **Clear package boundaries**: Separate packages for data loading, evaluation, inference, and RL workflows
- **Minimal abstraction**: Code is organized to be readable and modifiable without deep framework knowledge
- **Shared utilities**: Common patterns extracted into reusable helpers to reduce duplication
- **Direct access**: All functionality accessible through straightforward Python APIs

## Overview

SearchLM enables research and development of search query generation systems through:

- **Dataset Management**: Load and ingest NFCorpus and SciFact datasets from HuggingFace MTEB
- **Full-Text Search**: Tantivy-based search engine for indexing and querying documents
- **Evaluation Framework**: Comprehensive IR metrics (NDCG, MRR, Precision@K, Recall@K, MAP)
- **Baseline Generation**: Generate search queries using instruction-tuned LLMs
- **RLHF Training**: Train models using Group Relative Policy Optimization (GRPO) with verifiable rewards

## Features

- Dataset loading with train/dev/test splits and relevance judgments (qrels)
- Unified search index supporting multiple datasets
- Full-text search powered by Tantivy
- Comprehensive evaluation with standard IR metrics
- Baseline query generation workflow
- RLHF training workflow with GRPO
- Configuration management via YAML
- vLLM integration for efficient inference

## Installation

### Requirements

- Python >= 3.12
- CUDA 12.1+ (for GPU acceleration)
- Linux recommended (Ubuntu 22.04+)

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd searchLM

# Install using uv (recommended)
# Install core package only (data processing, no GPU dependencies)
uv sync

# Install with vLLM for inference
uv sync --group vllm

# Install with training dependencies
uv sync --group training

# Install everything including dev tools
uv sync --all-groups

# Alternative: Using pip
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your tokens
# - HF_TOKEN: HuggingFace token (required)
# - WANDB_API_KEY: Weights & Biases key (optional, for training)

# Create data directories
mkdir -p models data search_index output
```

## Quick Start

### 1. Ingest Datasets

First, ingest the datasets into a search index:

```python
from searchlm import ingest_all_datasets

# Ingest both NFCorpus and SciFact datasets
ingest_all_datasets(index_path="./search_index")
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
```

### 4. Evaluate Search Quality

```python
from searchlm import SearchEvaluator

# Initialize evaluator
evaluator = SearchEvaluator(index_path="./search_index")

# Evaluate a single query
metrics, error = evaluator.evaluate_single_query(
    query_text="cancer treatment",
    query_id="nfcorpus_1",
    dataset_name="nfcorpus",
    split="test",
    k=100
)

if not error:
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")

# Evaluate entire dataset
results = evaluator.evaluate(
    dataset_name="nfcorpus",
    split="test",
    k=100
)
evaluator.print_metrics(results)
```

## Workflows

### Baseline Query Generation

Generate baseline queries using instruction-tuned LLMs:

```bash
# Generate queries for SciFact (default)
uv run python -m searchlm.workflows.baseline.baseline

# Or use the class directly in Python:
from searchlm.workflows.baseline.baseline import BaselineGenerator

# SciFact
generator = BaselineGenerator(
    dataset_name="mteb/scifact",
    output_filename="scifact_generated_queries.tsv"
)
generator.generate()

# NFCorpus
generator = BaselineGenerator(
    dataset_name="mteb/nfcorpus",
    output_filename="nfcorpus_generated_queries.tsv"
)
generator.generate()
```

### RLHF Training

Train models using Group Relative Policy Optimization (GRPO):

```python
from searchlm.workflows.rlhf.data_prep import prepare_training_data
from searchlm.workflows.rlhf.training import train
from searchlm.workflows.rlhf.evaluation import evaluate

# Step 1: Prepare training data
prepare_training_data()

# Step 2: Train (choose mode)
train(use_vllm_server=False)  # 1 GPU mode (colocate)
# train(use_vllm_server=True)  # 2 GPU mode (server)

# Step 3: Evaluate
evaluate(compare_baseline=True)
```

## Documentation

For detailed usage instructions, see the [Usage Guide](docs/USAGE.md). The usage guide covers:

- Configuration management
- Dataset loading and ingestion
- Search operations
- Evaluation methods and metrics
- Baseline query generation
- RLHF training workflow
- Advanced usage patterns

## Project Structure

```
searchlm/
├── searchlm/
│   ├── __init__.py              # Main package exports
│   ├── config.py                # Configuration management (fixed cache bug)
│   ├── prompts.py               # LLM prompts + shared utilities
│   ├── inference.py             # vLLM inference engine
│   ├── data/                    # Dataset loading and ingestion
│   │   ├── loaders/
│   │   │   ├── base.py          # Base loader class
│   │   │   ├── nfcorpus.py      # NFCorpus loader
│   │   │   ├── scifact.py       # SciFact loader
│   │   │   ├── factory.py       # Dict-based loader factory
│   │   │   └── helpers.py       # Shared loading utilities (NEW)
│   │   ├── ingesters/
│   │   │   ├── base.py          # Base ingester + shared helpers
│   │   │   ├── nfcorpus.py      # NFCorpus ingester
│   │   │   ├── scifact.py       # SciFact ingester
│   │   │   └── pipeline.py      # Ingestion pipeline
│   │   └── schemas/
│   │       ├── constants.py     # Field constants
│   │       └── index.py         # Index schema builder
│   ├── services/
│   │   ├── search.py            # Search engine wrapper
│   │   ├── evaluator.py         # Search evaluator (617 → 539 lines)
│   │   └── metrics.py           # IR metrics calculations
│   ├── models/
│   │   ├── domain.py            # Core domain models
│   │   └── evaluation.py        # Evaluation models
│   └── workflows/
│       ├── baseline/
│       │   └── baseline.py      # BaselineGenerator class (merged CLI + sampling)
│       └── rlhf/
│           ├── data_prep.py     # Training data preparation
│           ├── reward.py        # Reward function for GRPO
│           ├── training.py      # GRPO training
│           └── evaluation.py    # Model evaluation
├── config/
│   └── default.yaml             # Configuration file
├── scripts/
│   └── base_evaluation.py       # Standalone evaluation script
├── docs/
│   └── USAGE.md                 # Usage guide
└── README.md
```

**Key Simplifications:**
- ✅ Removed 6 duplicate/unused files (4 loaders + 2 utils)
- ✅ Added shared helpers to reduce code duplication (~250 lines saved)
- ✅ Simplified SearchEvaluator (78 lines saved)
- ✅ Consolidated prompt utilities across workflows
- ✅ Merged baseline: CLI + sampling → single `BaselineGenerator` class (2 files → 1)
- ✅ Simplified RLHF: Removed CLI argparse, clean separation of concerns (4 files)
- ✅ All imports at the top of files (no lazy imports)
- ✅ Fixed config.py cache bug

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

## Configuration

Configuration is managed through `config/default.yaml`. Key settings include:

- Model configuration (name, max tokens, temperature)
- Training hyperparameters (learning rate, batch sizes, epochs)
- Reward function weights (NDCG, MRR)
- Paths for models, data, indices, and output

See the [Usage Guide](docs/USAGE.md) for detailed configuration documentation.

## License

MIT License - see LICENSE file for details.

## Author

**Supreeth Rao** - raosupreeth00@gmail.com

## Keywords

search, information retrieval, reinforcement learning, verifiable rewards, RLVR, tantivy, full-text search
