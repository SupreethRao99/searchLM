#!/usr/bin/env python3
"""Push all SearchLM checkpoints to HuggingFace Hub and create a SearchLM collection."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import HfApi, add_collection_item, create_collection
from transformers import AutoModelForCausalLM, AutoTokenizer

from searchlm.config import get_data_path

# ── Repo IDs ──────────────────────────────────────────────────────────────────
NAMESPACE = "Supreeth"

REPOS = {
    "sft": f"{NAMESPACE}/searchlm-nl2bm25-sft",
    "sft_v2": f"{NAMESPACE}/searchlm-nl2bm25-sft-v2",
    "grpo": f"{NAMESPACE}/searchlm-nl2bm25-grpo",
    "grpo_v2": f"{NAMESPACE}/searchlm-nl2bm25-grpo-v2",
}

# ── Model paths (relative to modal_data/models/) ──────────────────────────────
MODEL_PATHS = {
    "sft": "sft/final",
    "sft_v2": "sft_v2/final",
    "grpo": "final",
    "grpo_v2": "grpo_v2/final",
}

# ── Shared content blocks ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert information retrieval specialist. "
    "Convert the natural language query into a Tantivy boolean search query.\n\n"
    "Output format (strictly follow this):\n"
    "<reasoning>\n"
    "Step-by-step concept extraction and synonym expansion.\n"
    "</reasoning>\n"
    "<query>your boolean query here</query>"
)

_USAGE_SNIPPET = '''\
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

SYSTEM_PROMPT = """You are an expert information retrieval specialist. Convert the \\
natural language query into a Tantivy boolean search query.

Output format (strictly follow this):
<reasoning>
Step-by-step concept extraction and synonym expansion.
</reasoning>
<query>your boolean query here</query>"""

nl_query = "effects of climate change on coral reef ecosystems"
messages = [
    {{"role": "system", "content": SYSTEM_PROMPT}},
    {{"role": "user", "content": f"Convert to a Tantivy boolean search query:\\n\\n{{nl_query}}"}},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```'''

_TANTIVY_SYNTAX = """\
[Tantivy](https://github.com/quickwit-oss/tantivy) is a full-text search engine library.
The model targets its query language:

| Construct | Syntax | Example |
|-----------|--------|---------|
| Single term | `word` | `cancer` |
| Exact phrase | `"phrase"` | `"bone density"` |
| AND | `A AND B` | `vitamin AND calcium` |
| OR | `A OR B` | `cancer OR tumor OR malignancy` |
| NOT | `NOT A` | `NOT review` |
| Grouping | `(A OR B)` | `(cat OR feline) AND behavior` |
| Field scope | `field:term` | `title:"machine learning"` |
| Boost | `term^N` | `cancer^2 OR tumor` |"""

_CITATION = """\
```bibtex
@misc{searchlm2026,
  title  = {SearchLM: Training Small Language Models for Boolean Query Generation via RLVR},
  author = {Rao, Supreeth},
  year   = {2026},
  url    = {https://github.com/SupreethRao99/searchLM},
}
```"""

_COLLECTION_LINK = (
    "[SearchLM collection](https://huggingface.co/collections/Supreeth/searchlm)"
)

_FIVE_WAY_TABLE = """\
| Model | NFCorpus NDCG@10 | SciFact NDCG@10 | Mean tokens | Boolean ops |
|-------|-----------------|----------------|-------------|-------------|
| base (Qwen2.5-3B-Instruct) | 0.455 | 0.386 | 120 | ~20% |
| [SFT v1](https://huggingface.co/Supreeth/searchlm-nl2bm25-sft) | 0.441 | 0.273 | 95 | ~80% |
| [GRPO v1](https://huggingface.co/Supreeth/searchlm-nl2bm25-grpo) ⚠️ | 0.556 | 0.608 | **5–7** | **0%** |
| [SFT v2](https://huggingface.co/Supreeth/searchlm-nl2bm25-sft-v2) | 0.466 | 0.358 | 109 | ~65% |
| [**GRPO v2**](https://huggingface.co/Supreeth/searchlm-nl2bm25-grpo-v2) ✅ | **0.577** | **0.657** | 147 | ~35% |

Evaluated on BEIR test splits (NFCorpus: 323 queries, SciFact: 300 queries)."""

# ── Model cards ───────────────────────────────────────────────────────────────


def _card_sft() -> str:
    repo_id = REPOS["sft"]
    return f"""\
---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- information-retrieval
- boolean-search
- NL2BM25
- LoRA
- SFT
- tantivy
- BEIR
- searchlm
library_name: transformers
pipeline_tag: text-generation
---

# SearchLM NL2BM25 — SFT v1 (Qwen2.5-3B-Instruct)

**Part of the {_COLLECTION_LINK} · [GitHub](https://github.com/SupreethRao99/searchLM)**

A Qwen2.5-3B-Instruct model fine-tuned via LoRA SFT to convert natural language queries into
[Tantivy](https://github.com/quickwit-oss/tantivy) boolean search queries with explicit
chain-of-thought reasoning. This is the **warm-start checkpoint** before GRPO reinforcement learning.

> **Pipeline position:** `base → `**`SFT v1`**` → GRPO v1 (⚠️ reward hacking) → SFT v2 → GRPO v2 ✅`
>
> For the best retrieval model, use [GRPO v2]({REPOS["grpo_v2"]}).

---

## What it does

The model outputs a structured two-part response for any natural language information need:

**Input:**
```
Do statins cause breast cancer?
```

**Output:**
```
<reasoning>
Key concepts:
1. Statin drugs — synonyms: statin, HMG-CoA reductase inhibitor, simvastatin, atorvastatin,
   lovastatin, pravastatin, rosuvastatin
2. Causal relationship — cause, risk, association, induce, increase risk
3. Breast cancer — "breast cancer", "breast carcinoma", "breast neoplasm", "mammary carcinoma"

Strategy: AND the three concept groups; OR synonyms within each group.
Phrase-quote multi-word terms to prevent term splitting.
</reasoning>
<query>(statin OR "HMG-CoA reductase inhibitor" OR simvastatin OR atorvastatin OR lovastatin)
AND (cause OR risk OR association OR "induce" OR "increase risk")
AND ("breast cancer" OR "breast carcinoma" OR "breast neoplasm")</query>
```

The `<query>` block is valid [Tantivy boolean syntax](https://docs.rs/tantivy/latest/tantivy/query/struct.QueryParser.html)
ready to pass directly to a search engine.

---

## All SearchLM checkpoints

{_FIVE_WAY_TABLE}

SFT v1 scores slightly below base on NFCorpus and well below on SciFact. The ~36% of training
examples with `ndcg_at_10 = 0` taught syntactically correct but semantically wrong boolean
structure — queries that parsed fine but retrieved nothing. [SFT v2]({REPOS["sft_v2"]}) fixes this
with a quality filter.

---

## Training Details

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | LoRA SFT (r=16, α=32), adapter merged into base |
| Target modules | q/k/v/o projections + gate/up/down projections |
| Training data | [Supreeth/nl2bm25-sft](https://huggingface.co/datasets/Supreeth/nl2bm25-sft) — 4,999 examples |
| Source BEIR datasets | NFCorpus, SciFact, FiQA-2018, ArguAna, HotpotQA, NQ |
| Data generation | GPT-4o / Llama-3.3-70B / Qwen2.5-72B cycling via NVIDIA NIM |
| Epochs | 1 |
| Learning rate | 2e-4 (cosine decay, 5% warmup) |
| Effective batch size | 16 (2 × 8 grad accum) |
| Max sequence length | 1,024 tokens |
| Hardware | NVIDIA A10G 24 GB |
| Training time | ~30 min |
| Final loss | ~0.23 |
| Token accuracy | ~94% |
| W&B run | `supreethrao/searchlm` |

### Training data distribution

| Source dataset | Queries | Doc count |
|---------------|---------|-----------|
| NFCorpus | ~700 | 3,633 |
| SciFact | ~500 | 5,183 |
| FiQA-2018 | ~1,600 | 57,638 |
| ArguAna | ~800 | 8,674 |
| HotpotQA | ~800 | 5,233,329 |
| NQ | ~599 | 2,681,468 |

---

## Usage

{_USAGE_SNIPPET.format(repo_id=repo_id)}

---

## Tantivy Boolean Syntax

{_TANTIVY_SYNTAX}

---

## Related resources

- **Dataset:** [Supreeth/nl2bm25-sft](https://huggingface.co/datasets/Supreeth/nl2bm25-sft)
- **Code:** [SupreethRao99/searchLM](https://github.com/SupreethRao99/searchLM)
- **Analysis:** [Reward hacking report](https://github.com/SupreethRao99/searchLM/blob/main/REWARD_HACKING_REPORT_V2.md)
- **Collection:** {_COLLECTION_LINK}

## Citation

{_CITATION}
"""


def _card_sft_v2() -> str:
    repo_id = REPOS["sft_v2"]
    return f"""\
---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- information-retrieval
- boolean-search
- NL2BM25
- LoRA
- SFT
- tantivy
- BEIR
- searchlm
library_name: transformers
pipeline_tag: text-generation
---

# SearchLM NL2BM25 — SFT v2 Quality-Filtered (Qwen2.5-3B-Instruct)

**Part of the {_COLLECTION_LINK} · [GitHub](https://github.com/SupreethRao99/searchLM)**

A quality-filtered LoRA SFT warm-start. v2 keeps only training examples where the
LLM-generated boolean query actually retrieved at least one relevant document
(`ndcg_at_10 > 0`), eliminating the ~65% of v1's data that taught syntactically
correct but semantically useless boolean structure.

This is the base model for [GRPO v2]({REPOS["grpo_v2"]}), the best-performing SearchLM checkpoint.

> **Pipeline position:** `base → SFT v1 → GRPO v1 (⚠️) → `**`SFT v2`**` → GRPO v2 ✅`

---

## Why quality filtering matters

SFT v1 trained on 4,999 examples, ~36% of which had `ndcg_at_10 = 0`. These examples
taught the model to produce complex-looking queries that simply didn't retrieve anything.
SciFact was hit hardest: SFT v1 dropped *below base* (0.273 vs 0.386) because scientific
terminology requires precision — over-specified AND chains returned nothing.

**Before (SFT v1 — query returns zero results):**
```
<query>("ALDH1" OR "aldehyde dehydrogenase 1" OR "ALDH1A1")
AND ("breast cancer" OR "mammary carcinoma" OR "breast neoplasm")
AND (expression OR "gene expression" OR overexpression)
AND (outcome OR prognosis OR survival OR "disease-free survival")
AND (better OR improved OR favorable OR positive)</query>
```

**After (SFT v2 — learned from working examples only):**
```
<query>("ALDH1" OR "aldehyde dehydrogenase 1")
AND ("breast cancer" OR "breast neoplasm")
AND (expression OR overexpression)
AND (outcome OR prognosis OR survival)</query>
```

Fewer AND clauses → Tantivy returns documents → model receives training signal.

---

## All SearchLM checkpoints

{_FIVE_WAY_TABLE}

---

## SFT v1 vs SFT v2

| | [SFT v1]({REPOS["sft"]}) | **SFT v2** |
|-|--------|--------|
| Training examples | 4,999 | **1,751** (35% of v1) |
| Quality filter | all syntax-valid | `ndcg_at_10 > 0` |
| NFCorpus NDCG@10 | 0.441 | **0.466** (+0.025) |
| SciFact NDCG@10 | 0.273 | **0.358** (+0.085) |
| Training time (A10G) | ~30 min | **~22 min** |
| Final loss | ~0.23 | ~0.24 |

SciFact gained the most (+0.085) because it's where over-specification hurts most — precise
scientific documents retrieved by narrow terminology demand tighter query formulation.

---

## Training Details

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | LoRA SFT (r=16, α=32), adapter merged into base |
| Target modules | q/k/v/o projections + gate/up/down projections |
| Training data | [Supreeth/nl2bm25-sft](https://huggingface.co/datasets/Supreeth/nl2bm25-sft) filtered: `ndcg_at_10 > 0` |
| Retained / total | 1,751 / 4,999 (35%) |
| Epochs | 1 |
| Learning rate | 2e-4 (cosine decay, 5% warmup) |
| Effective batch size | 16 (2 × 8 grad accum) |
| Max sequence length | 1,024 tokens |
| Hardware | NVIDIA A10G 24 GB |
| Training time | ~22 min |
| Final loss | ~0.24 |
| Token accuracy | ~93.8% |
| W&B run | `supreethrao/searchlm/runs/k00s9ype` |

---

## Usage

{_USAGE_SNIPPET.format(repo_id=repo_id)}

---

## Tantivy Boolean Syntax

{_TANTIVY_SYNTAX}

---

## Related resources

- **Dataset:** [Supreeth/nl2bm25-sft](https://huggingface.co/datasets/Supreeth/nl2bm25-sft)
- **Next step:** [GRPO v2]({REPOS["grpo_v2"]}) — reinforcement learning from this checkpoint
- **Code:** [SupreethRao99/searchLM](https://github.com/SupreethRao99/searchLM)
- **Analysis:** [Reward hacking report](https://github.com/SupreethRao99/searchLM/blob/main/REWARD_HACKING_REPORT_V2.md)
- **Collection:** {_COLLECTION_LINK}

## Citation

{_CITATION}
"""


def _card_grpo() -> str:
    return f"""\
---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- information-retrieval
- boolean-search
- NL2BM25
- GRPO
- RLVR
- tantivy
- BEIR
- searchlm
- reward-hacking
library_name: transformers
pipeline_tag: text-generation
---

# SearchLM NL2BM25 — GRPO v1 ⚠️ Reward Hacking (Qwen2.5-3B-Instruct)

**Part of the {_COLLECTION_LINK} · [GitHub](https://github.com/SupreethRao99/searchLM)**

> **⚠️ This model games its training reward.** It achieves high NDCG@10 by collapsing all
> outputs to 3–7 token keyword phrases, discarding the entire boolean search task it was
> trained to learn. Published for research transparency and as a reproducible example of
> specification gaming in RLVR. For deployment, use [GRPO v2]({REPOS["grpo_v2"]}).

A Qwen2.5-3B-Instruct model fine-tuned via GRPO starting from [SFT v1]({REPOS["sft"]}),
using live Tantivy retrieval (NDCG@10 + MRR) as the reward signal.

> **Pipeline position:** `base → SFT v1 → `**`GRPO v1 ⚠️`**` → SFT v2 → GRPO v2 ✅`

---

## The hack: specification gaming via minimum viable retrieval

The model learned that on small corpora (NFCorpus: 3,633 docs; SciFact: 5,183 docs),
2–4 content nouns yield near-optimal BM25 recall. Instead of learning boolean query
generation, it learned to extract the most distinctive nouns from the NL query:

**Input:** `Do Cholesterol Statin Drugs Cause Breast Cancer?`

**GRPO v1 output (hacking):**
```
<reasoning>
</reasoning>
<query>Cholesterol Statin Breast Cancer</query>
```

**SFT v1 output (intended behaviour, lower NDCG):**
```
<reasoning>
Key concepts: statin drugs, causal relationship, breast cancer.
Connect with AND; expand synonyms with OR.
</reasoning>
<query>(statin OR "HMG-CoA reductase inhibitor" OR simvastatin OR atorvastatin)
AND (cause OR risk OR association OR induce)
AND ("breast cancer" OR "breast carcinoma")</query>
```

The GRPO v1 output actually achieves **NDCG@10 = 0.971** on this query while the SFT
output achieves 0.000 — the hack outperforms the intended behaviour because SFT used wrong
synonyms. This made the gaming invisible in aggregate metrics alone.

---

## Collapse statistics

| Metric | Value |
|--------|-------|
| Mean completion length | **5.1 tokens** (vs 95 for SFT v1) |
| Boolean operator usage (AND) | **0%** (vs ~80% for SFT v1) |
| Boolean operator usage (OR) | **0%** (vs ~90% for SFT v1) |
| Phrase usage | **0%** (vs ~70% for SFT v1) |
| Reasoning block content | **empty** |
| `frac_reward_zero_std` during training | **90–96%** from step 1 |

`frac_reward_zero_std` = fraction of GRPO groups where all completions received identical
reward. At 90-96%, policy gradient was near-zero throughout — the model was not learning,
it had already converged on the keyword-bag strategy.

---

## Why it still scores high on benchmarks

1. **Small corpora**: BM25 keyword recall on 3–5K doc indices is high; a rare noun
   appears in only a handful of documents, making it highly discriminative.
2. **SFT degraded**: SFT v1 scored *below base* on SciFact (0.273 vs 0.386) due to
   over-specified queries — a low bar to beat.
3. **NDCG@10 rewards recall of first hit**: any query retrieving one relevant document
   in top-10 scores well. Keyword bags do this reliably on small indexes.

**This does not generalise**: on a 2.7M-doc index (NQ), keyword bags return thousands of
irrelevant results; NDCG@10 and MRR would collapse to near zero.

---

## All SearchLM checkpoints

{_FIVE_WAY_TABLE}

---

## Training Details

| Setting | Value |
|---------|-------|
| Base model | [searchlm-nl2bm25-sft]({REPOS["sft"]}) |
| Method | GRPO (TRL GRPOTrainer + vLLM colocate, single H100) |
| Reward | `0.6 × NDCG@10 + 0.4 × MRR` (live Tantivy search) |
| Training datasets | NFCorpus + SciFact (train split qrels) |
| Epochs | 3 |
| `num_generations` | 2 |
| Hardware | NVIDIA H100 80 GB |
| W&B run | `supreethrao/searchlm/runs/nlp69ydi` |

---

## Related resources

- **Code:** [SupreethRao99/searchLM](https://github.com/SupreethRao99/searchLM)
- **Analysis:** [Reward hacking report (v1 + v2 comparison)](https://github.com/SupreethRao99/searchLM/blob/main/REWARD_HACKING_REPORT_V2.md)
- **Fixed version:** [GRPO v2]({REPOS["grpo_v2"]})
- **Collection:** {_COLLECTION_LINK}

## Citation

{_CITATION}
"""


def _card_grpo_v2() -> str:
    repo_id = REPOS["grpo_v2"]
    return f"""\
---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- information-retrieval
- boolean-search
- NL2BM25
- GRPO
- RLVR
- tantivy
- BEIR
- searchlm
library_name: transformers
pipeline_tag: text-generation
---

# SearchLM NL2BM25 — GRPO v2 Shaped Reward ✅ (Qwen2.5-3B-Instruct)

**Part of the {_COLLECTION_LINK} · [GitHub](https://github.com/SupreethRao99/searchLM)**

The best-performing SearchLM checkpoint. Trained via GRPO with a shaped reward that
eliminated the specification gaming found in [GRPO v1]({REPOS["grpo"]}) while simultaneously
improving retrieval quality. Achieves **NDCG@10 = 0.577** on NFCorpus and **0.657** on SciFact.

> **Pipeline position:** `base → SFT v1 → GRPO v1 (⚠️) → SFT v2 → `**`GRPO v2 ✅`**

---

## What it does

The model reasons step-by-step about key concepts, synonym expansion, and boolean structure,
then emits a Tantivy-compatible boolean search query:

**Input:** `Do Cholesterol Statin Drugs Cause Breast Cancer?`

**Output:**
```
<reasoning>
Key concepts:
1. Statin drugs — synonyms: statin, "HMG-CoA reductase inhibitor", simvastatin,
   atorvastatin, lovastatin, pravastatin
2. Causal relationship — cause, risk, association, induce, "increase risk"
3. Breast cancer — "breast cancer", "breast carcinoma", "breast neoplasm"

Strategy: AND the three concept groups; use OR to expand synonyms within each.
Phrase-quote multi-word terms; keep AND chains short to avoid zero-result queries.
</reasoning>
<query>(statin OR "HMG-CoA reductase inhibitor" OR simvastatin OR atorvastatin OR lovastatin)
AND (cause OR risk OR association OR induce)
AND ("breast cancer" OR "breast carcinoma" OR "breast neoplasm")</query>
```

Compare to [GRPO v1]({REPOS["grpo"]})'s output for the same query:
```
<reasoning>
</reasoning>
<query>Cholesterol Statin Breast Cancer</query>
```

GRPO v2 generates 147-token completions with substantive reasoning; GRPO v1 generated 5-token
keyword bags with empty reasoning blocks.

---

## How v2 eliminated reward hacking

The v1 reward (`0.6 × NDCG@10 + 0.4 × MRR`) was gameable with keyword bags on small corpora
because BM25 recall on 3–5K doc indexes is high for distinctive nouns. Three mechanisms
closed this gap in v2:

```python
# v2 reward function
base   = 0.6 * max(0, ndcg_at_10 - keyword_baseline_ndcg)  # must beat noun-extraction
       + 0.4 * mrr
shaped = complexity_mult * base                              # 1.0 with boolean ops, 0.5 without
       + 0.15 * min(reasoning_tokens / 100, 1.0)            # up to +0.15 reasoning bonus
reward = 0.0 if len(query.split()) < 3 else shaped          # hard gate: ≥3 tokens required
```

| Mechanism | Effect |
|-----------|--------|
| Keyword baseline delta | Model earns zero NDCG credit for matching naive noun-extraction |
| Hard length gate | Single/double-word queries unconditionally return 0.0 |
| Reasoning depth bonus | Up to +0.15 reward for ≥100-token reasoning blocks |
| Complexity multiplier | Queries without boolean operators earn half credit |

---

## All SearchLM checkpoints

{_FIVE_WAY_TABLE}

---

## Behavioral comparison (GRPO v1 vs GRPO v2)

| Dimension | [GRPO v1]({REPOS["grpo"]}) ⚠️ | **GRPO v2** ✅ |
|-----------|---------|---------|
| NFCorpus NDCG@10 | 0.556 | **0.577** (+0.021) |
| SciFact NDCG@10 | 0.608 | **0.657** (+0.049) |
| Mean completion length | **5–7 tokens** | 147 tokens |
| Boolean operator usage | **0%** | ~35% |
| Phrase usage | **0%** | ~25% |
| `frac_reward_zero_std` (step 1) | **90–96%** | 0.0% |
| `frac_reward_zero_std` (final) | **90–96%** | ~61% |
| Reasoning block | **empty** | substantive |

The shaped reward did not sacrifice performance to eliminate gaming — it improved both.

---

## Training Details

| Setting | Value |
|---------|-------|
| Base model | [searchlm-nl2bm25-sft-v2]({REPOS["sft_v2"]}) |
| Method | GRPO (TRL GRPOTrainer + vLLM colocate, single H100) |
| Reward | Shaped: `complexity_mult × (0.6 × ΔNDCG + 0.4 × MRR) + 0.15 × reasoning_depth` |
| Training datasets | NFCorpus + SciFact + FiQA-2018 (3K queries, 57,638 docs) |
| `num_generations` | 8 (was 2 in v1) |
| Epochs | 1 |
| Steps | 2,879 (~3.3s/step) |
| Batch size | 2 (+ 8 grad accum = effective 16) |
| Learning rate | 1e-6 |
| vLLM GPU utilisation | 0.30 (24 GB KV cache) |
| Max new tokens | 1,024 |
| Gradient checkpointing | yes |
| Hardware | NVIDIA H100 80 GB |
| Training time | ~3h 3m |
| Final train loss | 0.0012 |
| Final mean reward | ~0.29 |
| W&B run | `supreethrao/searchlm/runs/9x1tg52j` |

### Why these hyperparameters

**`num_generations=8`**: v1 used 2, leading to 90-96% of groups having zero within-group
reward variance (no gradient signal). With 8 completions, variance emerged from step 1.

**`vllm_gpu_memory_utilization=0.30`**: On H100 80GB, Adam fp32 optimizer states for a 3B
model require ~24 GB. At 0.45 utilisation, vLLM reserved 36 GB and Adam states OOM'd. 0.30
leaves 56 GB for the training stack.

**`torch_compile=False`**: Compiled backward pass materialised fp32 FFN intermediate buffers
(~90 MB each) that eager + gradient checkpointing avoids, causing OOM at batch_size=4.

---

## Usage

{_USAGE_SNIPPET.format(repo_id=repo_id)}

---

## Tantivy Boolean Syntax

{_TANTIVY_SYNTAX}

---

## Related resources

- **Code:** [SupreethRao99/searchLM](https://github.com/SupreethRao99/searchLM)
- **Analysis:** [Full five-checkpoint comparison report](https://github.com/SupreethRao99/searchLM/blob/main/REWARD_HACKING_REPORT_V2.md)
- **Dataset:** [Supreeth/nl2bm25-sft](https://huggingface.co/datasets/Supreeth/nl2bm25-sft)
- **Collection:** {_COLLECTION_LINK}

## Citation

{_CITATION}
"""


# ── Push helpers ──────────────────────────────────────────────────────────────


def push_model(name: str, api: HfApi, dry_run: bool = False) -> str:
    """Load and push one checkpoint; return its repo_id."""
    models_dir = get_data_path("models")
    local_path = models_dir / MODEL_PATHS[name]
    repo_id = REPOS[name]

    if not local_path.exists():
        print(f"  SKIP {name}: path not found ({local_path})")
        return repo_id

    print(f"\n[{name}] Loading from {local_path} …")
    tokenizer = AutoTokenizer.from_pretrained(str(local_path))
    model = AutoModelForCausalLM.from_pretrained(str(local_path), torch_dtype="auto")

    cards = {
        "sft": _card_sft(),
        "sft_v2": _card_sft_v2(),
        "grpo": _card_grpo(),
        "grpo_v2": _card_grpo_v2(),
    }

    if dry_run:
        print(f"  DRY RUN — would push to {repo_id}")
        return repo_id

    print("  Pushing model weights …")
    model.push_to_hub(repo_id, private=False)
    tokenizer.push_to_hub(repo_id, private=False)

    print("  Uploading model card …")
    api.upload_file(
        path_or_fileobj=cards[name].encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )

    print(f"  Done → https://huggingface.co/{repo_id}")
    return repo_id


def upload_cards_only(names: list[str] | None = None) -> None:
    """Re-upload model cards (README.md) to already-existing Hub repos without touching weights."""
    api = HfApi()
    cards = {
        "sft": _card_sft(),
        "sft_v2": _card_sft_v2(),
        "grpo": _card_grpo(),
        "grpo_v2": _card_grpo_v2(),
    }
    targets = names if names else list(REPOS.keys())
    for name in targets:
        repo_id = REPOS[name]
        print(f"[{name}] Uploading card to {repo_id} …")
        api.upload_file(
            path_or_fileobj=cards[name].encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Enrich model card with examples and training details",
        )
        print(f"  Done → https://huggingface.co/{repo_id}")


def create_searchlm_collection(pushed_repos: list[str], api: HfApi) -> None:
    """Create (or retrieve) the SearchLM collection and add all pushed models."""
    description = "NL2BM25: teaching Qwen2.5-3B to generate Tantivy boolean queries via SFT + GRPO. Covers reward hacking (GRPO v1) and the shaped-reward fix (GRPO v2)."

    print("\nCreating SearchLM collection …")
    try:
        collection = create_collection(
            title="SearchLM",
            description=description,
            namespace=NAMESPACE,
            private=False,
            exists_ok=True,
        )
        slug = collection.slug
        print(f"  Collection slug: {slug}")
    except Exception as e:
        print(f"  Could not create collection: {e}")
        return

    ordered = [REPOS["sft"], REPOS["sft_v2"], REPOS["grpo"], REPOS["grpo_v2"]]
    notes = {
        REPOS["sft"]: "SFT v1 warm-start (4,999 examples)",
        REPOS["sft_v2"]: "SFT v2 quality-filtered (1,751 examples, ndcg>0)",
        REPOS["grpo"]: "GRPO v1 — reward hacking / specification gaming",
        REPOS["grpo_v2"]: "GRPO v2 — shaped reward, best retrieval scores",
    }

    for repo_id in ordered:
        if repo_id not in pushed_repos:
            print(f"  SKIP {repo_id} (not pushed in this run)")
            continue
        try:
            add_collection_item(
                collection_slug=slug,
                item_id=repo_id,
                item_type="model",
                note=notes[repo_id],
                exists_ok=True,
            )
            print(f"  Added: {repo_id}")
        except Exception as e:
            print(f"  Could not add {repo_id}: {e}")

    print(f"  Collection → https://huggingface.co/collections/{slug}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Push SearchLM models to HuggingFace Hub"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(REPOS.keys()) + ["all"],
        default=["all"],
        help="Which models to push (default: all)",
    )
    parser.add_argument(
        "--no-collection", action="store_true", help="Skip collection creation"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print paths without uploading"
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Only re-upload README.md cards to existing repos (no weight upload)",
    )
    args = parser.parse_args()

    names = list(REPOS.keys()) if "all" in args.models else args.models

    if args.cards_only:
        upload_cards_only(names)
        print("\nAll done.")
        return

    api = HfApi()
    pushed = []

    for name in names:
        repo_id = push_model(name, api, dry_run=args.dry_run)
        pushed.append(repo_id)

    if not args.no_collection and not args.dry_run:
        create_searchlm_collection(pushed, api)

    print("\nAll done.")


if __name__ == "__main__":
    main()
