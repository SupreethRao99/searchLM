# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

Use `uv` exclusively for all dependency management. Never use `pip` or `pip-tools` directly.

```bash
uv sync                    # Install/reinstall all dependencies
uv add <package>           # Add a dependency
uv remove <package>        # Remove a dependency
uv run <script.py>         # Run a script with proper env
```

## Development Commands

```bash
make format          # Format code with ruff + organize imports with isort
make lint            # Lint with ruff (no modifications)
make check           # Check imports and lint (dry run)
make clean           # Remove all build/cache artifacts
```

Individual tools (via `uv run`):
```bash
uv run ruff format searchlm/ scripts/
uv run ruff check searchlm/ scripts/
uv run isort searchlm/ scripts/
```

There are no automated tests in this project.

## Architecture Overview

SearchLM trains a language model (Qwen2.5-3B-Instruct) to generate better boolean search queries using RLVR (Reinforcement Learning with Verifiable Rewards). The reward signal comes from actual search quality metrics (NDCG@10 and MRR) computed against a Tantivy full-text index.

### Package Structure (`searchlm/`)

- **`config.py`** - OmegaConf-based configuration loaded from `config/default.yaml`. `get_data_path(subdir)` handles path resolution for both local and Modal environments (detects via `MODAL_IMAGE_ID` env var).
- **`data/`** - Dataset loading and index ingestion
  - `loaders/` - Load NFCorpus/SciFact from HuggingFace MTEB into `DatasetSplit` objects (queries, documents, qrels)
  - `ingesters/` - Index documents into Tantivy; `ingest_all_datasets()` ingests both datasets into a unified index
  - `schemas/` - Tantivy index schema (fields: doc_id, title, text, dataset, source_id)
- **`services/`** - Core runtime services
  - `search.py` - `SearchEngine`: Tantivy wrapper supporting boolean queries, field-specific search, and dataset filtering
  - `evaluator.py` - `SearchEvaluator`: Orchestrates IR metric computation against ground-truth qrels
  - `metrics.py` - Pure metric implementations: NDCG, MRR, MAP, Precision@K, Recall@K
- **`models/`** - Pydantic-style data structures
  - `domain.py` - `DatasetSplit`, `Document`, `Query`
  - `evaluation.py` - `SearchResult`, `QuerySearchResult`
- **`rlhf/`** - Training workflow
  - `data_prep.py` - Generates formatted prompts from train split queries; saves HuggingFace Dataset to `modal_data/datasets/train`
  - `reward.py` - `RewardFunction` class: called by TRL's GRPOTrainer; parses `<query>...</query>` tags from model output, executes search, returns `0.6*NDCG@10 + 0.4*MRR`
  - `training.py` - GRPO training loop; two modes: `use_vllm_server=False` (colocate, single GPU) or `True` (server, dual GPU)
  - `evaluation.py` - Multi-run evaluation for statistical reporting

### Key Design Decisions

**Reward signal**: The model must output queries wrapped in `<query>...</query>` XML tags. Missing or empty tags yield 0.0 reward.

**Two-environment path resolution**: `get_data_path()` resolves `./modal_data` relative to project root locally, or `/root/searchlm/modal_data` on Modal (detected via `MODAL_IMAGE_ID`).

**Training data split**: Reward is computed against the `train` split qrels during GRPO training (configured in `config.reward.split`). Evaluation uses `test` split.

**Unified search index**: Both NFCorpus and SciFact documents are ingested into a single Tantivy index, with a `dataset` field for filtering.

## Cloud Deployment (Modal)

Training requires GPU (H100/A100 recommended). Modal is used for cloud GPU access.

```bash
# Start interactive GPU container for development/debugging
modal run modal_infra.py::dev_shell

# Connect to running container
modal shell <container-id>   # get ID from: modal container list
```

Data persists in Modal volumes: `searchlm` (datasets, models, indices), `huggingface-cache`, `vllm-cache`. Always press Ctrl+C in Terminal 1 after `dev_shell` to commit volume changes.

## Configuration

`config/default.yaml` controls all hyperparameters. Key sections:
- `model.name`: Base model (default: `Qwen/Qwen2.5-3B-Instruct`)
- `training.batch_size.colocate/server`: Separate configs for 1-GPU vs 2-GPU training
- `reward.ndcg_weight`/`mrr_weight`: Reward weighting (default: 0.6/0.4)
- `paths.data_dir`: Root data directory (default: `./modal_data`)

## Environment Variables

Copy `.env.example` to `.env`. Required: `HF_TOKEN`. Optional: `WANDB_API_KEY`.
