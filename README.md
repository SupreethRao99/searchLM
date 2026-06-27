# SearchLM

Training a small language model to generate Tantivy boolean search queries via **LoRA SFT + GRPO with live retrieval rewards** (NDCG@10 + MRR). Along the way, we discovered and mitigated a clear case of specification gaming.

**[HuggingFace Collection](https://huggingface.co/collections/Supreeth/searchlm-6a3fc9272a7f2e5aa7773cf0) · [Reports](reports/)**

---

## Results

Five checkpoints trained and evaluated on BEIR test splits:

| Model | NFCorpus NDCG@10 | SciFact NDCG@10 | Completion length | Boolean ops |
|-------|-----------------|----------------|-------------------|-------------|
| Base (Qwen2.5-3B-Instruct) | 0.455 | 0.386 | 120 tok | ~20% |
| [SFT v1](https://huggingface.co/Supreeth/searchlm-nl2bm25-sft) | 0.441 | 0.273 | 95 tok | ~80% |
| [GRPO v1](https://huggingface.co/Supreeth/searchlm-nl2bm25-grpo) ⚠️ | 0.556 | 0.608 | **5–7 tok** | **0%** |
| [SFT v2](https://huggingface.co/Supreeth/searchlm-nl2bm25-sft-v2) | 0.466 | 0.358 | 109 tok | ~65% |
| [**GRPO v2**](https://huggingface.co/Supreeth/searchlm-nl2bm25-grpo-v2) ✅ | **0.577** | **0.657** | **147 tok** | ~35% |

GRPO v1 gamed the reward — it ignored all boolean structure and output 3–7 token keyword bags. Despite the hack, it scored higher than SFT (small corpora make keyword bags near-optimal for BM25). GRPO v2's shaped reward eliminated the gaming while improving retrieval further. Full analysis in [`reports/reward_hacking_v2.md`](reports/reward_hacking_v2.md).

---

## What it does

The model converts natural language queries into [Tantivy](https://github.com/quickwit-oss/tantivy) boolean search queries with explicit chain-of-thought reasoning:

```
Input: Do statins cause breast cancer?

<reasoning>
Key concepts:
1. Statin drugs — statin, HMG-CoA reductase inhibitor, simvastatin, atorvastatin
2. Causal relationship — cause, risk, association, induce
3. Breast cancer — "breast cancer", "breast carcinoma", "breast neoplasm"

Strategy: AND the three groups; OR synonyms within each.
</reasoning>
<query>(statin OR "HMG-CoA reductase inhibitor" OR simvastatin OR atorvastatin)
AND (cause OR risk OR association OR induce)
AND ("breast cancer" OR "breast carcinoma" OR "breast neoplasm")</query>
```

---

## Architecture

```
NL query
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Training pipeline                                              │
│                                                                 │
│  Stage 1 — LoRA SFT                                            │
│  LLM-generated (query, boolean) pairs → format + syntax warm-  │
│  start. Filtered to ndcg_at_10 > 0 in v2.                      │
│                                                                 │
│  Stage 2 — GRPO                                                 │
│  Live Tantivy search → NDCG@10 + MRR as reward.               │
│  v2 adds: keyword baseline delta · complexity multiplier ·     │
│  reasoning depth bonus · hard min-token gate.                   │
└─────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  SearchEngine│────▶│ SearchEvaluator │────▶│  RewardFunction  │
│  (Tantivy)   │     │ (NDCG, MRR, MAP)│     │  (v1 or v2)      │
└──────────────┘     └─────────────────┘     └──────────────────┘
```

### Reward functions

**v1** (simple, gameable):
```
reward = 0.6 × NDCG@10 + 0.4 × MRR
```

**v2** (shaped, anti-gaming):
```python
base   = 0.6 * max(0, ndcg_at_10 - keyword_baseline_ndcg)  # must beat noun-extraction
       + 0.4 * mrr
shaped = complexity_mult * base                              # 1.0 with boolean ops, 0.5 without
       + 0.15 * min(reasoning_tokens / 100, 1.0)            # up to +0.15 reasoning bonus
reward = 0.0 if len(query.split()) < 3 else shaped          # hard gate: ≥3 tokens required
```

---

## Reward hacking findings

GRPO v1 discovered that on small corpora (NFCorpus: 3,633 docs, SciFact: 5,183 docs), 2–4 content nouns yield near-optimal BM25 recall. It collapsed to outputs like:

```
<reasoning>
</reasoning>
<query>Cholesterol Statin Breast Cancer</query>
```

Training signal collapsed from step 1: `frac_reward_zero_std` = 90–96% (fraction of GRPO groups with zero within-group reward variance, meaning near-zero policy gradient throughout).

Three mechanisms in v2's shaped reward closed the loophole:
1. **Keyword baseline delta** — model must beat naive noun-extraction to earn NDCG credit
2. **Hard length gate** — queries < 3 tokens → 0.0
3. **Reasoning depth bonus** — up to +0.15 for ≥100 reasoning tokens

Result: `frac_reward_zero_std` started at 0.0 in v2, completion length 147 tokens vs 5–7, NDCG improved further despite eliminating the hack.

See [`reports/reward_hacking_v1.md`](reports/reward_hacking_v1.md) and [`reports/reward_hacking_v2.md`](reports/reward_hacking_v2.md) for full analysis.

---

## Quick start

```bash
git clone https://github.com/SupreethRao99/searchLM.git
cd searchLM
uv sync
cp .env.example .env  # add HF_TOKEN (required), WANDB_API_KEY (optional)
```

**Search with an existing index:**
```python
from searchlm.services.search import SearchEngine
from searchlm.services.evaluator import SearchEvaluator

engine = SearchEngine(index_path="./modal_data/indices")
results = engine.search('("breast cancer" OR "breast carcinoma") AND statin', limit=10)

evaluator = SearchEvaluator(index_path="./modal_data/indices")
metrics, _ = evaluator.evaluate_single_query(
    query_text="do statins cause breast cancer",
    query_id="nfcorpus_1",
    dataset_name="nfcorpus",
    split="test",
    k=100,
)
print(f"NDCG@10: {metrics['ndcg@10']:.4f}  MRR: {metrics['mrr']:.4f}")
```

**Run inference with the best model:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Supreeth/searchlm-nl2bm25-grpo-v2",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Supreeth/searchlm-nl2bm25-grpo-v2")

SYSTEM_PROMPT = """You are an expert information retrieval specialist. Convert the \
natural language query into a Tantivy boolean search query.

Output format (strictly follow this):
<reasoning>
Step-by-step concept extraction and synonym expansion.
</reasoning>
<query>your boolean query here</query>"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Convert to a Tantivy boolean search query:\n\nDo statins cause breast cancer?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## Training on Modal (cloud GPU)

All training runs on Modal with persistent volumes. An H100 is required for GRPO.

```bash
# ── v1 pipeline ────────────────────────────────────────
modal run modal_infra.py::run_sft          # LoRA SFT v1 (~30 min, A10G)
modal run modal_infra.py::run_grpo         # GRPO v1 (~6h, H100)

# ── v2 pipeline (recommended) ──────────────────────────
modal run modal_infra.py::run_sft_v2       # LoRA SFT v2 (~22 min, A10G)
modal run modal_infra.py::run_grpo_v2      # GRPO v2 (~3h, H100)

# ── Analysis ───────────────────────────────────────────
modal run modal_infra.py::run_analysis_v2 --analysis-version compare  # all 5 checkpoints

# ── Publish ────────────────────────────────────────────
modal run modal_infra.py::push_all_models_to_hub   # push all + create HF collection
modal run modal_infra.py::upload_cards_to_hub       # update README only (no weight re-upload)

# ── Dev shell ──────────────────────────────────────────
modal run modal_infra.py::dev_shell   # interactive GPU container (L4)
```

Data persists in Modal volume `searchlm` at `modal_data/`. Always Ctrl+C `dev_shell` to commit.

---

## Project structure

```
searchlm/                        # Core package
├── config.py                    # OmegaConf config + get_data_path()
├── prompts.py                   # Shared system prompt (SFT + GRPO aligned)
├── inference.py                 # vLLM engine wrapper
├── data/
│   ├── loaders/                 # NFCorpus, SciFact, FiQA, ArguAna, HotpotQA, NQ
│   └── ingesters/               # Tantivy index ingestion pipeline
├── services/
│   ├── search.py                # SearchEngine (Tantivy wrapper)
│   ├── evaluator.py             # SearchEvaluator (NDCG/MRR against qrels)
│   └── metrics.py               # NDCG, MRR, MAP, P@K, R@K
├── models/
│   ├── domain.py                # DatasetSplit, Document, Query
│   └── evaluation.py            # SearchResult, QuerySearchResult
└── rlhf/
    ├── sft.py                   # LoRA SFT (version="v1"/"v2")
    ├── training.py              # GRPO (version="v1"/"v2", colocate/server)
    ├── reward.py                # RewardFunction v1
    ├── reward_v2.py             # RewardFunctionV2 (shaped reward)
    ├── data_prep.py             # Prepare GRPO training dataset
    └── evaluation.py            # Multi-run eval for statistical reporting

scripts/
├── generate_sft_dataset.py      # SFT data generation via NVIDIA NIM (multi-model)
├── analyze_reward_hacking.py    # Behavioral + retrieval analysis (--version compare)
└── push_models.py               # Push all checkpoints to HuggingFace + collection

config/default.yaml              # All hyperparameters (training/reward/sft/datasets v1+v2)
modal_infra.py                   # Modal cloud function definitions
reports/
├── reward_hacking_v1.md         # v1 hacking analysis
└── reward_hacking_v2.md         # v1 vs v2 five-checkpoint comparison
```

### Key paths on Modal volume

```
modal_data/
├── models/
│   ├── sft/final          # SFT v1 (4,999 examples)
│   ├── sft_v2/final       # SFT v2 (1,751 quality-filtered)
│   ├── final              # GRPO v1 (reward hacking)
│   └── grpo_v2/final      # GRPO v2 (shaped reward) ← best
├── datasets/
│   ├── train/             # GRPO v1 dataset
│   └── train_v2/          # GRPO v2 dataset (+ FiQA, keyword baselines)
├── outputs/
│   ├── reward_hacking/           # v1 analysis outputs
│   └── reward_hacking_compare/   # cross-version comparison outputs
└── indices/               # Tantivy index (NFCorpus + SciFact + FiQA)
```

---

## Configuration

`config/default.yaml` — annotated key sections:

```yaml
# v2 training (recommended)
training_v2:
  num_generations: 8             # was 2 in v1; critical for within-group variance
  vllm_gpu_memory_utilization: 0.30  # H100 budget: 24GB vLLM, 56GB training
  torch_compile: false           # compiled backward OOMs on fp32 FFN buffers

reward_v2:
  use_keyword_baseline: true     # NDCG delta over noun-extraction
  complexity_soft_penalty: 0.5   # queries without boolean ops get 0.5× reward
  min_query_tokens: 3            # hard gate
  reasoning_bonus_weight: 0.15   # reasoning depth bonus

datasets_v2:
  names: ["nfcorpus", "scifact", "fiqa"]  # arguana excluded: no train split
  fiqa_max_train_queries: 3000
```

---

## Development

```bash
make format   # ruff format + isort
make lint     # ruff check (read-only)
make check    # imports + lint dry run
make clean    # remove build/cache artifacts
```

No automated test suite. Evaluation runs on BEIR test splits via `scripts/analyze_reward_hacking.py`.

---

## Datasets

| Dataset | Train queries | Test queries | Documents | Domain |
|---------|--------------|-------------|-----------|--------|
| NFCorpus | 2,590 | 323 | 3,633 | Medical |
| SciFact | 809 | 300 | 5,183 | Scientific |
| FiQA-2018 | 5,500 (cap 3K) | — | 57,638 | Finance |

All loaded from [MTEB](https://github.com/embeddings-benchmark/mteb) via HuggingFace.

---

## Citation

```bibtex
@misc{searchlm2026,
  title  = {SearchLM: Training Small Language Models for Boolean Query Generation via RLVR},
  author = {Rao, Supreeth},
  year   = {2026},
  url    = {https://github.com/SupreethRao99/searchLM},
}
```

## License

MIT — see [LICENSE](LICENSE).
