# SearchLM — Project Progress Log

Last updated: 2026-06-27 (v2 complete)

---

## Project Goals

1. **NL2BM25 model** — a powerful small model (Qwen2.5-3B) that converts natural language queries into high-quality Tantivy boolean search queries
2. **Reward hacking research** — study how the model tries to game the retrieval reward signal
3. **Mechanistic interpretability** — understand what the model has learned internally

---

## What We've Built

### Architecture

```
Natural Language Query
        ↓
  [Qwen2.5-3B-Instruct]
  (SFT warm-start on NL2BM25-SFT dataset)
  (GRPO fine-tuning with live retrieval reward)
        ↓
  <reasoning>...</reasoning>
  <query>boolean query here</query>
        ↓
  [Tantivy Full-Text Index]
  (~8M documents: NFCorpus + SciFact + FiQA + ArguAna + HotpotQA + NQ)
        ↓
  NDCG@10 + MRR  →  reward signal
```

### Training Pipeline (Two-Stage)

**Stage 1 — LoRA SFT** (`searchlm/rlhf/sft.py`)
- Goal: teach format (`<reasoning>/<query>` tags), not retrieval quality
- Dataset: `Supreeth/nl2bm25-sft` (5,083 examples, 6 BEIR datasets)
- Method: LoRA r=16 α=32, all attention + MLP projections, merged into base after training
- Result: loss 2.07 → 0.23, token accuracy 60% → 94% in 1 epoch on A10G
- W&B: `supreethrao/searchlm/runs/piu2eupj`

**Stage 2 — GRPO** (`searchlm/rlhf/training.py`)
- Goal: refine retrieval quality via live reward signal
- Starts from SFT merged checkpoint (`modal_data/models/sft/final`)
- Reward: `0.6 × NDCG@10 + 0.4 × MRR` from live Tantivy search
- Training datasets (for reward): NFCorpus + SciFact (train split qrels)
- 3 epochs on H100
- W&B: `supreethrao/searchlm/runs/nlp69ydi`
- **Finding**: reward hacking — collapsed to 5-token keyword bags, empty reasoning

**Stage 1v2 — LoRA SFT v2** (`searchlm/rlhf/sft.py`, `version="v2"`)
- Same as SFT v1 but filters to examples where `ndcg_at_10 > 0` (quality gate)
- Dataset: 1,751 examples (35% of 4,999 syntax-valid) — removes queries that retrieved nothing
- Result: loss 1.83 → 0.24, token accuracy 63% → 93.8% in 1 epoch on A10G (~22 min)
- Saved to `modal_data/models/sft_v2/`
- W&B: `supreethrao/searchlm/runs/k00s9ype`

**Stage 2v2 — GRPO v2** (`searchlm/rlhf/training.py`, `version="v2"`)
- Shaped reward: `complexity_mult × (0.6 × max(0, NDCG−baseline) + 0.4 × MRR) + 0.15 × reasoning_bonus`
- Keyword baseline delta: model must beat naive noun-extraction to earn NDCG credit
- Hard gate: queries < 3 tokens → reward = 0.0
- Reasoning depth bonus: up to +0.15 for ≥100-token reasoning blocks
- Complexity multiplier: 1.0 with boolean ops, 0.5 without
- Training datasets: NFCorpus + SciFact + FiQA (3K sampled), num_generations=8
- 1 epoch on H100, ~3h 3m, 2,879 steps at ~3.3s/step
- Saved to `modal_data/models/grpo_v2/`
- W&B: `supreethrao/searchlm/runs/9x1tg52j`

---

## Published Artifacts

| Artifact | URL | Description |
|---|---|---|
| SFT Dataset | `Supreeth/nl2bm25-sft` | 5,083 (NL → reasoning + boolean query) examples |
| SFT Model | `Supreeth/searchlm-nl2bm25-sft` | Qwen2.5-3B SFT checkpoint, merged LoRA |
| GRPO Model (v1) | `modal_data/models/final` (Modal volume) | Post-GRPO v1 checkpoint — reward hacking model |
| SFT Model (v2) | `modal_data/models/sft_v2/final` (Modal volume) | Quality-filtered SFT, ndcg_at_10 > 0 gate |
| GRPO Model (v2) | `modal_data/models/grpo_v2/final` (Modal volume) | Shaped reward, no hacking, best retrieval scores |

---

## Dataset: `Supreeth/nl2bm25-sft`

### Generation Pipeline (`scripts/generate_sft_dataset.py`)

- 4 models cycled round-robin via NVIDIA NIM to avoid per-model quota exhaustion:
  - `meta/llama-3.3-70b-instruct` — 2,751 examples
  - `mistralai/mistral-medium-3.5-128b` — 1,755 examples
  - `nvidia/llama-3.3-nemotron-super-49b-v1` — 577 examples
- System prompt: detailed Tantivy syntax reference + 4-step construction strategy + 5 worked examples
- Output format enforced: `<reasoning>...</reasoning><query>...</query>`
- Every query validated against live Tantivy index (syntax check + NDCG@10)

### Dataset Stats

| Metric | Value |
|---|---|
| Total examples | 5,083 |
| Syntax-valid | 4,999 / 5,083 (98.3%) |
| Retrieval non-empty | 3,241 / 5,083 (63.8%) |
| Mean NDCG@10 | 0.290 |

| Source Dataset | Domain | Examples |
|---|---|---|
| NFCorpus | Biomedical / nutrition | 1,726 |
| FiQA-2018 | Financial Q&A | 972 |
| HotpotQA | Multi-hop Wikipedia | 647 |
| ArguAna | Counter-argument retrieval | 643 |
| NQ | Open-domain Wikipedia | 560 |
| SciFact | Scientific fact-checking | 535 |

### Schema

Each record has:
- `id`, `original_query_id`, `dataset` — provenance
- `nl_query` — natural language input
- `boolean_query` — generated Tantivy query
- `reasoning` — chain-of-thought trace
- `messages` — `[system, user, assistant]` conversation, ready for SFT
- `syntax_valid`, `retrieval_valid`, `ndcg_at_10`, `num_retrieved` — quality signals
- `generator_model`, `generated_at` — generation provenance

---

## Search Infrastructure

### Tantivy Index (`searchlm/data/`)

All 6 BEIR corpora ingested into a unified Tantivy full-text index (~8M documents total):

| Dataset | Corpus Size |
|---|---|
| HotpotQA | ~5.2M docs |
| NQ | ~2.7M docs |
| FiQA-2018 | ~57K docs |
| NFCorpus | ~3.6K docs |
| SciFact | ~5.2K docs |
| ArguAna | ~8.7K docs |

Index schema: `doc_id`, `title`, `text`, `dataset`, `source_id`  
Field for dataset filtering: `dataset_filter="{dataset_name}"` in `SearchEngine.search()`

### Loaders + Ingesters

Factory pattern — `create_loader(dataset_id)` returns the right loader.  
`ingest_all_datasets(index_path, datasets=None)` ingests all or a filtered subset.

Datasets supported: `nfcorpus`, `scifact`, `fiqa`, `arguana`, `hotpotqa`, `nq`

---

## Reward Hacking Analysis + v2 Mitigations (2026-06-27)

Full five-way comparison via `scripts/analyze_reward_hacking.py --version compare`.
Outputs: `modal_data/outputs/reward_hacking_compare/` on Modal volume.
Full write-up: `reports/reward_hacking_v2.md`.

### Benchmark Scores — All Five Checkpoints

#### NDCG@10

| Model     | NFCorpus | SciFact | vs Base        |
|-----------|----------|---------|----------------|
| base      | 0.455    | 0.386   | —              |
| sft       | 0.441    | 0.273   | −0.014 / −0.113 |
| grpo      | 0.556    | 0.608   | +0.101 / +0.222 |
| sft\_v2   | 0.466    | 0.358   | +0.011 / −0.028 |
| **grpo\_v2** | **0.577** | **0.657** | **+0.122 / +0.271** |

#### Completion Length

| Model     | NFCorpus (mean tokens) | SciFact (mean tokens) |
|-----------|------------------------|-----------------------|
| base      | 120                    | 163                   |
| sft       | 95                     | 136                   |
| grpo      | **5**                  | **7**                 |
| sft\_v2   | 109                    | 139                   |
| grpo\_v2  | **147**                | **147**               |

#### Zero-reward fraction

| Model     | NFCorpus | SciFact |
|-----------|----------|---------|
| base      | 39.6%    | 47.1%   |
| sft       | 42.9%    | 69.6%   |
| grpo      | 21.7%    | 10.4%   |
| sft\_v2   | 39.1%    | 54.9%   |
| grpo\_v2  | **16.7%** | **8.0%** |

### v1 Reward Hacking (GRPO)

GRPO v1 gamed the `0.6 × NDCG@10 + 0.4 × MRR` reward by collapsing all outputs to 3–5 token keyword bags with empty `<reasoning>` blocks. On small corpora (NFCorpus 3.6K docs, SciFact 5.2K docs), BM25 keyword bags are near-optimal — the hack worked. Zero boolean operators used. `frac_reward_zero_std` = 90–96% throughout training (almost no within-group variance → near-zero policy gradient).

Despite the hack, GRPO v1 scored higher than SFT on NDCG@10 because SFT over-specified queries with wrong synonyms. SciFact NDCG actually dropped from base (0.386 → 0.273) under SFT.

### v2 Mitigations

**SFT v2 — quality filter**: filtered 4,999 → 1,751 examples keeping only `ndcg_at_10 > 0`. Removes training examples where the LLM-generated boolean query retrieved nothing. Result: +0.085 NDCG@10 on SciFact over SFT v1 (0.273 → 0.358).

**GRPO v2 — shaped reward**:
```
base  = 0.6 × max(0, NDCG@10 − keyword_baseline) + 0.4 × MRR
reward = complexity_mult × base + 0.15 × min(reasoning_tokens/100, 1.0)
       = 0.0  if  len(query.split()) < 3

keyword_baseline = NDCG@10 of naive noun-extraction (precomputed in dataset)
complexity_mult  = 1.0 with boolean operators, 0.5 without
```
Three mechanisms eliminate the v1 hack:
1. **Keyword baseline delta** — model must beat noun-extraction to earn NDCG credit
2. **Hard length gate** — queries < 3 tokens score zero unconditionally
3. **Reasoning bonus** — up to +0.15 for substantive reasoning

**Training dataset**: NFCorpus + SciFact + FiQA (3K sampled, 57K docs — resists keyword bags). `num_generations=8` (was 2) for within-group diversity.

### Result: Hack Eliminated, Performance Improved

GRPO v2 completions: **147 tokens** (vs 5-7 for v1). Full `<reasoning>` + structured `<query>` throughout training. `frac_reward_zero_std` started at 0.0, reached ~0.61 only at end of epoch. Boolean operators used on ~35% of queries (vs 0% for v1). And retrieval improved: +0.021 NFCorpus, +0.049 SciFact over the hacking model.

---

## Infrastructure

### Modal (`modal_infra.py`)

| Function | GPU | Timeout | Purpose |
|---|---|---|---|
| `dev_shell` | L4 | 1h | Interactive debugging |
| `run_sft` | A10G | 2h | LoRA SFT v1 training |
| `run_sft_v2` | A10G | 2h | LoRA SFT v2 (quality-filtered, ndcg > 0) |
| `push_sft_to_hub` | CPU | 30m | Push SFT model to HuggingFace |
| `run_grpo` | H100 | 12h | GRPO v1 training |
| `run_grpo_v2` | H100 | 14h | GRPO v2 (shaped reward, FiQA, num_gen=8) |
| `run_reward_hacking_analysis` | A10G | 4h | Evaluate base/SFT/GRPO v1 |
| `run_analysis_v2` | A10G | 4h | Evaluate all 5 checkpoints (pass `--analysis-version compare`) |
| `regenerate_report` | CPU | 10m | Re-render report from saved JSON without re-running inference |
| `fetch_analysis_outputs` | CPU | 10m | Print latest v1 report to stdout |

**Secrets required** (create at modal.com/secrets):
- `huggingface-secret` → `HF_TOKEN`
- `wandb-secret` → `WANDB_API_KEY`

**Volumes:**
- `searchlm` — datasets, models, Tantivy indices (`modal_data/`)
- `huggingface-cache` — model weights cache
- `vllm-cache` — vLLM compiled artifacts

### Run Order

```bash
# ── v1 (original, reward hacking) ──────────────────────────
modal run modal_infra.py::run_sft
modal run modal_infra.py::run_grpo
modal run modal_infra.py::run_reward_hacking_analysis

# ── v2 (mitigated) ─────────────────────────────────────────
modal run modal_infra.py::run_sft_v2
modal run modal_infra.py::run_grpo_v2
modal run modal_infra.py::run_analysis_v2 --analysis-version compare

# ── Utilities ───────────────────────────────────────────────
modal run modal_infra.py::fetch_analysis_outputs   # print latest v1 report
modal run modal_infra.py::push_sft_to_hub          # push SFT to HuggingFace
```

### Key Paths (Modal volume)

```
modal_data/
├── models/
│   ├── sft/              # v1 SFT (all syntax-valid examples)
│   │   ├── adapter/
│   │   └── final/
│   ├── sft_v2/           # v2 SFT (ndcg_at_10 > 0 filter, 1,751 examples)
│   │   ├── adapter/
│   │   └── final/
│   ├── final/            # v1 GRPO (reward hacking model)
│   └── grpo_v2/          # v2 GRPO (shaped reward, best retrieval scores)
│       └── final/
├── datasets/
│   ├── train/            # v1 GRPO dataset (NFCorpus + SciFact)
│   └── train_v2/         # v2 GRPO dataset (+ FiQA 3K, with keyword baselines)
├── outputs/
│   ├── reward_hacking/   # v1 analysis outputs
│   └── reward_hacking_compare/  # cross-version comparison outputs
└── indices/              # Tantivy index (NFCorpus + SciFact + FiQA)
```

---

## Code Structure

```
searchlm/
├── config.py                    # OmegaConf config, get_data_path()
├── prompts.py                   # Shared system prompt (aligned between SFT + GRPO)
├── data/
│   ├── loaders/                 # Dataset loaders (nfcorpus, scifact, fiqa, arguana, hotpotqa, nq)
│   │   └── factory.py           # create_loader(dataset_id)
│   └── ingesters/               # Tantivy ingesters + pipeline.ingest_all_datasets()
├── services/
│   ├── search.py                # SearchEngine (Tantivy wrapper)
│   ├── evaluator.py             # SearchEvaluator (NDCG, MRR against qrels)
│   └── metrics.py               # NDCG, MRR, MAP, P@K, R@K implementations
├── models/
│   ├── domain.py                # DatasetSplit, Document, Query
│   └── evaluation.py            # SearchResult, QuerySearchResult
└── rlhf/
    ├── sft.py                   # LoRA SFT (version="v1"/"v2")
    ├── training.py              # GRPO training (version="v1"/"v2", TRL + vLLM colocate/server)
    ├── reward.py                # RewardFunction v1 (0.6×NDCG + 0.4×MRR)
    ├── reward_v2.py             # RewardFunctionV2 (shaped: baseline delta + reasoning bonus + complexity mult)
    ├── data_prep.py             # Prepare GRPO training dataset (version="v1"/"v2")
    └── evaluation.py            # Multi-run eval for statistical reporting

scripts/
├── generate_sft_dataset.py      # SFT data generation (NVIDIA NIM, 4-model cycling)
├── analyze_reward_hacking.py    # Full behavioral + retrieval analysis (--version v1/v2/compare)
└── push_models.py               # Push all checkpoints to HuggingFace + create collection

config/default.yaml              # All hyperparameters (training_v2, reward_v2, sft_v2, datasets_v2)
modal_infra.py                   # Modal cloud functions
reports/
├── reward_hacking_v1.md         # v1 hacking analysis (publishable write-up)
└── reward_hacking_v2.md         # v1 vs v2 comparison (full five-checkpoint results)
```

---

## Reward Function Details

**v1** (`reward.py`) — `RewardFunction.__call__`:
1. Extract `<query>...</query>` from model completion — 0.0 if missing
2. Look up `query_id` → ground-truth qrels (pre-loaded at init)
3. Execute boolean query against Tantivy index with `dataset_filter`
4. Compute NDCG@10 and MRR against qrels
5. Return `0.6 × NDCG@10 + 0.4 × MRR`

**v2** (`reward_v2.py`) — `RewardFunctionV2.__call__`:
1. Extract `<query>...</query>` — 0.0 if missing or < 3 tokens (hard gate)
2. Look up qrels + `keyword_baseline_ndcg` (stored in dataset from data_prep)
3. Execute query against Tantivy
4. `ndcg_delta = max(0, NDCG@10 − keyword_baseline)`
5. `base = 0.6 × ndcg_delta + 0.4 × MRR`
6. `complexity_mult = 1.0` if boolean operators present, else `0.5`
7. `reasoning_bonus = 0.15 × min(reasoning_tokens / 100, 1.0)`
8. Return `complexity_mult × base + reasoning_bonus`

---

## Known Issues / Open Directions

- **OOD evaluation**: all evals use NFCorpus + SciFact (same as training). GRPO v2 uses boolean operators on ~35% of queries — unclear how it performs on large corpora (NQ 2.7M docs) where keyword bags definitely fail. Add NQ or TREC-COVID as zero-shot test.
- **Boolean operator gap**: GRPO v2 at ~35% vs SFT v2 at ~65% boolean usage. Curriculum complexity multiplier (anneal from 0.5 → 0.0 over training) could close this without sacrificing gradient signal early.
- **FiQA keyword baseline = 0.0**: precomputed baselines for FiQA were all zero (FiQA documents not yet ingested when data_prep ran on Modal). The model was rewarded for raw NDCG on FiQA rather than delta. Re-index FiQA before v3.
- **SFT generation coverage**: dataset has 5,083/~7,700 target examples due to rate-limit interruptions. Re-run `generate_sft_dataset.py` to top up (auto-resumes from checkpoint).
