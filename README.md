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
- **Evaluation Framework**: Comprehensive IR metrics (NDCG, MRR, Precision@K, Recall@K, MAP) with multiple runs and aggregate statistics
- **RLHF Training**: Train models using Group Relative Policy Optimization (GRPO) with verifiable rewards
- **Unified Evaluation**: Compare base and RLHF models with JSON audit logs and aggregate metrics

## Features

- Dataset loading with train/dev/test splits and relevance judgments (qrels)
- Unified search index supporting multiple datasets
- Full-text search powered by Tantivy
- Comprehensive evaluation with standard IR metrics, multiple runs, and aggregate statistics
- RLHF training workflow with GRPO
- Unified evaluation for base and fine-tuned models with JSON audit logs
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

### RLHF Training and Evaluation

Train models using Group Relative Policy Optimization (GRPO) and evaluate with comprehensive metrics:

```python
from searchlm.workflows.rlhf.data_prep import prepare_training_data
from searchlm.workflows.rlhf.training import train
from searchlm.workflows.rlhf.evaluation import evaluate_multiple_runs

# Step 1: Prepare training data
prepare_training_data()

# Step 2: Train (choose mode)
train(use_vllm_server=False)  # 1 GPU mode (colocate)
# train(use_vllm_server=True)  # 2 GPU mode (server)

# Step 3: Comprehensive Evaluation
# Run multiple evaluations for both base and RLHF models
# Results saved as JSON to volume mounts for auditing
evaluate_multiple_runs(
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    checkpoint_path=None,  # Uses latest checkpoint
    num_runs=3,  # Number of evaluation runs per model
    evaluate_base=True,  # Evaluate base model
    evaluate_rlhf=True,  # Evaluate RLHF model
)

# For single evaluation (backwards compatible):
from searchlm.workflows.rlhf.evaluation import evaluate
evaluate()
```

## Cloud Development with Modal

SearchLM provides Modal infrastructure for cloud-based GPU development with two powerful workflows:

### Hot Reload Mode (Recommended)

Rapid iteration with automatic code syncing - edit locally, changes reflect on remote GPU instantly:

```bash
# Terminal 1: Start serve mode (watches for file changes)
# Run from the repo root directory
modal serve modal_dev.py

# Terminal 2: Run workflows (edit code, save, and re-run)
modal run modal_dev.py::run_baseline
modal run modal_dev.py::run_training
modal run modal_dev.py::run_evaluation
modal run modal_dev.py::run_data_prep

# Run any Python module with hot reload
modal run modal_dev.py::run_python --module searchlm.workflows.baseline.baseline
```

**Iteration speed:** Edit → Save → Run (2-3 seconds, no restart needed)

### Interactive Shell Mode

VM-like SSH access for deep debugging and exploration:

```bash
# Terminal 1: Start container
modal run modal_infra.py::dev_shell

# Terminal 2: Attach and explore
modal container list
modal shell ta-XXXXXXXXXXXXXXXXXXXXX
cd /root/searchlm
python -m searchlm.workflows.baseline.baseline
```

**For complete documentation**, see [docs/MODAL_DEVELOPMENT.md](docs/MODAL_DEVELOPMENT.md).

## Model Sharing

### Upload to Hugging Face Hub

After training your model, you can easily upload it to Hugging Face Hub with comprehensive documentation:

```bash
# Upload latest checkpoint (private by default)
python scripts/upload_to_hf.py --repo-name searchlm-qwen2.5-3b-rlhf

# Upload specific checkpoint
python scripts/upload_to_hf.py \
  --repo-name searchlm-qwen2.5-3b-rlhf \
  --checkpoint-path ./modal_data/models/checkpoint-500

# Upload as public repository
python scripts/upload_to_hf.py \
  --repo-name searchlm-qwen2.5-3b-rlhf \
  --public
```

The upload utility automatically:
- Finds the latest checkpoint
- Generates a comprehensive model card with training details
- Uploads all model files and documentation
- Creates a private repository by default

For complete documentation, see [HuggingFace Upload Guide](docs/HUGGINGFACE_UPLOAD.md).

## Documentation

For detailed usage instructions, see the [Usage Guide](docs/USAGE.md). The usage guide covers:

- Configuration management
- Dataset loading and ingestion
- Search operations
- Evaluation methods and metrics
- Baseline query generation
- RLHF training workflow
- Advanced usage patterns

Additional documentation:
- [Complete Workflow Guide](docs/COMPLETE_WORKFLOW.md) - End-to-end workflow from training to HuggingFace
- [HuggingFace Upload Guide](docs/HUGGINGFACE_UPLOAD.md) - Upload models to HuggingFace Hub  
- [Modal Development Guide](docs/MODAL_DEVELOPMENT.md) - Cloud development with Modal
- [HuggingFace Quick Start](HUGGINGFACE_UPLOAD_QUICK_START.md) - Quick reference for model upload

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
│       └── rlhf/
│           ├── data_prep.py     # Training data preparation
│           ├── reward.py        # Reward function for GRPO
│           ├── training.py      # GRPO training
│           └── evaluation.py    # Unified evaluation (base + RLHF, multiple runs)
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
- ✅ Unified evaluation: Removed baseline directory, using RLHF evaluation for both base and fine-tuned models
- ✅ Enhanced evaluation with multiple runs, aggregate metrics, and JSON audit logs
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
