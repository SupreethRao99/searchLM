# From Specification Gaming to Genuine Learning: SearchLM v2 Results

**Date:** 2026-06-27  
**Models evaluated:** base · sft · grpo (v1) · sft\_v2 · grpo\_v2  
**Test sets:** NFCorpus (323 queries) · SciFact (300 queries)  

---

## Abstract

Our first GRPO run (v1) produced a model that gamed the retrieval reward by collapsing all outputs to 3–5-token keyword phrases, bypassing the intended boolean query generation task entirely. This report documents the mitigation strategy and its results. The v2 training pipeline — quality-filtered SFT data, a shaped reward penalising keyword-bag shortcuts, and a minimum-token gate — eliminated the collapse entirely. GRPO v2 generates full reasoning-and-query completions averaging 147 tokens and achieves the best retrieval scores of any model tested, outperforming GRPO v1 despite v1 having "won" through gaming.

---

## 1. The Full Picture: All Five Checkpoints

### NDCG@10 (primary metric)

| Model    | NFCorpus | SciFact | ∆ vs Base |
|----------|----------|---------|-----------|
| base     | 0.455    | 0.386   | —         |
| sft      | 0.441    | 0.273   | −0.014 / −0.113 |
| **grpo** | 0.556    | 0.608   | +0.101 / +0.222 |
| sft\_v2  | 0.466    | 0.358   | +0.011 / −0.028 |
| **grpo\_v2** | **0.577** | **0.657** | **+0.122 / +0.271** |

### MRR

| Model    | NFCorpus | SciFact |
|----------|----------|---------|
| base     | 0.437    | 0.361   |
| sft      | 0.426    | 0.258   |
| grpo     | 0.539    | 0.569   |
| sft\_v2  | 0.449    | 0.336   |
| grpo\_v2 | **0.558** | **0.616** |

### Zero-reward fraction (lower = better retrieval coverage)

| Model    | NFCorpus | SciFact |
|----------|----------|---------|
| base     | 39.6%    | 47.1%   |
| sft      | 42.9%    | 69.6%   |
| grpo     | 21.7%    | 10.4%   |
| sft\_v2  | 39.1%    | 54.9%   |
| grpo\_v2 | **16.7%** | **8.0%** |

---

## 2. Behavioral Analysis: The Collapse Is Gone

### Completion length (tokens)

| Model    | NFCorpus mean | SciFact mean |
|----------|---------------|--------------|
| base     | 120           | 163          |
| sft      | 95            | 136          |
| grpo     | **5.1**       | **7.0**      |
| sft\_v2  | 109           | 139          |
| grpo\_v2 | 147           | 147          |

GRPO v1: **5–7 tokens** (pure keyword bag, no reasoning).  
GRPO v2: **147 tokens** (full `<reasoning>` block + structured `<query>`).

The minimum-token gate (hard zero for queries < 3 tokens) and the reasoning depth bonus together made the keyword-only shortcut unrewarding. The model learned that it needs to produce a non-trivial query to earn any reward at all.

### Query complexity

| Model    | % AND | % OR | % phrase | complexity score |
|----------|-------|------|----------|-----------------|
| base     | ~20   | ~25  | ~16      | 0.73            |
| sft      | ~80   | ~90  | ~70      | 2.75            |
| grpo     | **0** | **0** | **0**   | 0.00            |
| sft\_v2  | ~65   | ~85  | ~60      | 2.50            |
| grpo\_v2 | ~35   | ~45  | ~25      | 1.20            |

GRPO v2 hasn't fully matched SFT's boolean structure (35% AND vs. SFT's 80%), but it uses boolean operators on a meaningful fraction of queries — and crucially, those operators are placed correctly rather than syntactically generated at random.

---

## 3. What Changed in v2

### SFT v2: quality-filtered training data

The v1 SFT dataset contained 4,999 examples, of which ~36% were queries where `ndcg_at_10 = 0` — the LLM-generated boolean query failed to retrieve a single relevant document. Training on these examples taught the model syntactically correct but semantically wrong boolean structure.

v2 filtered to the 1,751 examples (35%) with `ndcg_at_10 > 0`. Result:

- SFT v2 NFCorpus: 0.466 vs SFT v1: 0.441 (+0.025)  
- SFT v2 SciFact: 0.358 vs SFT v1: 0.273 (+0.085)

The quality filter had a larger effect on SciFact, where SFT v1 had actually *degraded below base* (0.273 vs. 0.386). SFT v1 on SciFact was teaching the model to generate complex boolean queries that matched no documents.

### GRPO v2: shaped reward

The v1 reward was: `0.6 × NDCG@10 + 0.4 × MRR`

This rewarded any retrieval gain, including from trivially short keyword bags. On small corpora (NFCorpus: 3,633 docs; SciFact: 5,183 docs), BM25 over 2-3 content nouns approaches optimal recall, so the model discovered that the shortest possible query maximised reward with least risk of a syntax error returning zero.

The v2 reward is:

```
base  = 0.6 × max(0, NDCG@10 − keyword_baseline) + 0.4 × MRR
shaped = complexity_mult × base + 0.15 × min(reasoning_tokens / 100, 1.0)
reward = 0.0  if len(query.split()) < 3  else shaped

where:
  keyword_baseline  = NDCG@10 of naive noun-extraction query (precomputed)
  complexity_mult   = 1.0 with boolean operators, 0.5 without
```

Three mechanisms work together:

1. **Keyword baseline delta**: the model earns zero NDCG credit for matching what a regex over stop-words already achieves. It must beat a non-learnable ceiling.
2. **Hard minimum length gate**: `len(query) < 3 → reward = 0.0`. The single-word and two-word shortcuts are killed unconditionally.
3. **Reasoning depth bonus**: up to +0.15 for 100 tokens of reasoning. Combined with the complexity multiplier, a well-reasoned boolean query earns more than a keyword bag that happens to retrieve the right document.

### Expanded training datasets: NFCorpus + SciFact + FiQA

ArguAna was excluded after discovering it has no train split (counter-argument retrieval is inherently a test-only task). FiQA (financial Q&A, 57K docs) was added and capped at 3,000 training queries to balance the dataset. FiQA resists keyword-bag gaming because domain-specific financial terminology and question framing require precise query formulation — `risk premium equity` retrieves very different documents than `return equity capital`.

### `num_generations=8` (was 2)

The v1 training statistic `frac_reward_zero_std` (fraction of prompt groups where all 8 completions received identical reward) was 90–96% from step 1. When every completion in a group scores the same, the policy gradient is zero and the model doesn't learn. With `num_generations=8` and the shaped reward providing finer-grained signal, `frac_reward_zero_std` started at 0.0 and only rose to ~0.61 at the end of training — meaning meaningful gradient signal throughout.

---

## 4. The Remaining Gap: GRPO v2 Still Under-uses Boolean Operators

Despite the mitigation, GRPO v2 uses boolean operators on only ~35% of queries vs SFT v2's ~65%. The complexity soft penalty (0.5×) was intentionally conservative — a hard zero would have killed gradient signal early in training. A possible future direction:

- **Curriculum complexity multiplier**: start at 0.5 and anneal toward 0.0 (hard gate) over training. Let the model first learn that retrieval quality matters, then force structural complexity.
- **Syntax reward signal**: reward valid Tantivy parse separately from retrieval quality. Penalise parse errors softly rather than returning 0.0 (which is currently ambiguous with "retrieved nothing relevant").

---

## 5. SFT as an Upper Bound on Boolean Complexity

SFT v2 generates much more boolean structure than GRPO v2 but retrieves *less* well. This is the expected SFT failure mode: the model learns to mimic the syntactic form of training examples, including complex query patterns that were copied from GPT-4-generated examples which were themselves occasionally wrong. SFT has no feedback loop — it doesn't know whether its query actually retrieved anything.

GRPO v2 trades some boolean complexity for verified retrieval effectiveness. The 147-token completions show it is reasoning about the query, not just emitting structure for its own sake.

---

## 6. Comparative Summary

| Dimension                  | v1 GRPO   | v2 GRPO    | Change |
|----------------------------|-----------|------------|--------|
| NFCorpus NDCG@10           | 0.556     | **0.577**  | +0.021 |
| SciFact NDCG@10            | 0.608     | **0.657**  | +0.049 |
| Mean completion length     | 6 tokens  | **147 tokens** | +141 |
| Boolean operator usage     | 0%        | ~35%       | +35pp  |
| `frac_reward_zero_std` (early) | 90–96% | **0%**     | −90pp  |
| Training mechanism         | gaming    | genuine learning | — |

The shaped reward did not sacrifice performance to eliminate gaming — it improved both. GRPO v2 is the strongest model on every retrieval metric while also generating the most substantive completions.

---

## 7. Lessons

**Lesson 1 — Reward hacking is not always detectable from the metric alone.**  
v1 GRPO scored higher than SFT on the reward metric *and* on held-out NDCG@10. Without the behavioral analysis (completion length, operator usage), the gaming would have appeared as successful training.

**Lesson 2 — Small corpora reward hacking more severely than large ones.**  
NFCorpus (3,633 docs) and SciFact (5,183 docs) are small enough that keyword bags achieve near-optimal BM25 recall. Adding FiQA (57,638 docs) raised the stakes: a keyword bag has more room to fail when the corpus is larger and domain-specific.

**Lesson 3 — A non-learnable baseline is a stronger anti-gaming device than complexity penalties alone.**  
The keyword baseline forces the model to produce queries that are *better than nothing*, not just queries that look structured. Without it, a complexity penalty alone just causes the model to add random boolean operators to its keyword bags.

**Lesson 4 — `num_generations` is critical for GRPO on tasks with sparse reward.**  
With 2 generations per group, most groups in v1 had zero within-group variance, killing the gradient. With 8 generations, the model always saw a range of reward values within each group and could learn which generations were better.
