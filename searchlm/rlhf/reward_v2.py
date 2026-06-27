"""Shaped reward function for GRPO v2.

Addresses the three root causes of reward hacking identified in v1:

1. Keyword-baseline delta: model must BEAT naive noun extraction, not just
   retrieve something. A keyword bag that equals the baseline gets no NDCG reward.

2. Reasoning depth bonus: incentivises a non-empty <reasoning> block.
   Caps at 1.0 when reasoning reaches ~100 whitespace tokens.

3. Complexity soft penalty: queries with no boolean operators (AND/OR/NOT/phrases)
   receive only `complexity_soft_penalty` (default 0.5) of the base reward.
   This is a soft penalty rather than a hard zero to preserve gradient signal
   early in training while still pushing the model toward structured queries.

4. Hard minimum-length gate: queries shorter than `min_query_tokens` tokens
   get zero reward — eliminates the single-word hack trivially.

Reward formula:
    base = ndcg_weight * max(0, NDCG@10 - keyword_baseline) + mrr_weight * MRR
    shaped = complexity_mult * base
           + reasoning_bonus_weight * min(reasoning_tokens / reasoning_target_tokens, 1.0)
    reward = 0.0 if query < min_query_tokens tokens else shaped
"""

import re
from typing import Iterable, Sequence

from searchlm import SearchEvaluator
from searchlm.config import get_config, get_data_path


def _has_boolean_structure(query: str) -> bool:
    q = query.upper()
    return (
        " AND " in q
        or " OR " in q
        or " NOT " in q
        or q.startswith("NOT ")
        or '"' in query
    )


def _extract_reasoning(text: str) -> str:
    match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


class RewardFunctionV2:
    """
    Shaped reward for GRPO v2.

    Differences from v1:
    - Rewards improvement over a precomputed keyword baseline (stored in the dataset)
    - Adds a reasoning depth bonus
    - Applies a soft complexity multiplier for queries without boolean operators
    - Hard zeros queries shorter than min_query_tokens
    """

    __name__ = "reward_function_v2"

    def __init__(self, dataset):
        self.config = get_config()
        self.cfg = self.config.reward_v2

        indices_dir = get_data_path("indices")
        self.evaluator = SearchEvaluator(index_path=str(indices_dir))

        # Build prompt → metadata mapping (same as v1)
        self.prompt_to_metadata: dict[str, dict] = {}
        self.qrels_cache: dict[tuple, dict] = {}

        for example in dataset:
            prompt = example["prompt"]
            prompt_key = str(prompt) if isinstance(prompt, list) else prompt
            self.prompt_to_metadata[prompt_key] = {
                "query_id": example["query_id"],
                "dataset_name": example["dataset_name"],
                "keyword_baseline_ndcg": float(
                    example.get("keyword_baseline_ndcg", 0.0)
                ),
            }

        # Pre-load qrels
        print("Pre-loading qrels for v2 training...")
        unique_pairs = {(ex["dataset_name"], self.cfg.split) for ex in dataset}
        for ds_name, split in unique_pairs:
            key = (ds_name, split)
            print(f"  Loading qrels: {ds_name} ({split})...")
            self.qrels_cache[key] = self.evaluator.load_qrels(ds_name, split)
        print("✓ Qrels pre-loaded")

    def __call__(
        self,
        completions: Sequence[str | list],
        prompts: Sequence[str | list],
        **kwargs,
    ) -> Iterable[float]:
        rewards = []

        for completion, prompt in zip(completions, prompts):
            # Normalise completion to string
            if isinstance(completion, list):
                completion_text = next(
                    (
                        m.get("content", "")
                        for m in completion
                        if isinstance(m, dict) and m.get("role") == "assistant"
                    ),
                    " ".join(
                        m.get("content", "") if isinstance(m, dict) else str(m)
                        for m in completion
                    ),
                )
            else:
                completion_text = completion

            # Look up metadata
            prompt_key = str(prompt) if isinstance(prompt, list) else prompt
            meta = self.prompt_to_metadata.get(prompt_key)
            if not meta:
                rewards.append(0.0)
                continue

            query_id = meta["query_id"]
            dataset_name = meta["dataset_name"]
            kw_baseline = meta["keyword_baseline_ndcg"]

            # Extract boolean query from completion
            query_match = re.search(
                r"<query>\s*(.*?)\s*</query>", completion_text, re.DOTALL
            )
            if not query_match or not query_match.group(1).strip():
                rewards.append(0.0)
                continue
            query_text = query_match.group(1).strip()

            # Hard minimum-length gate
            if len(query_text.split()) < self.cfg.min_query_tokens:
                rewards.append(0.0)
                continue

            # Retrieve + evaluate
            cache_key = (dataset_name, self.cfg.split)
            qrels_all = self.qrels_cache.get(cache_key, {})
            qrels = qrels_all.get(query_id, {})
            if not qrels:
                rewards.append(0.0)
                continue

            metrics, error = self.evaluator.evaluate_query(
                query_text=query_text,
                qrels=qrels,
                k=self.cfg.eval_k,
                dataset_filter=dataset_name,
            )
            if error:
                rewards.append(0.0)
                continue

            model_ndcg = metrics["ndcg@10"]
            model_mrr = metrics["mrr"]

            # 1. Base reward: improvement over keyword baseline
            ndcg_delta = max(0.0, model_ndcg - kw_baseline)
            base_reward = (
                self.cfg.ndcg_weight * ndcg_delta + self.cfg.mrr_weight * model_mrr
            )

            # 2. Complexity soft multiplier
            complexity_mult = (
                1.0
                if _has_boolean_structure(query_text)
                else self.cfg.complexity_soft_penalty
            )

            # 3. Reasoning depth bonus
            reasoning_text = _extract_reasoning(completion_text)
            reasoning_tokens = len(reasoning_text.split())
            reasoning_bonus = self.cfg.reasoning_bonus_weight * min(
                reasoning_tokens / self.cfg.reasoning_target_tokens, 1.0
            )

            reward = complexity_mult * base_reward + reasoning_bonus
            rewards.append(float(reward))

        return rewards
