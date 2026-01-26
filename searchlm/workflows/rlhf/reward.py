"""Reward function for GRPO training."""

import re
from pathlib import Path
from typing import Iterable, Sequence

from searchlm import SearchEvaluator
from searchlm.config import get_config


def reward_function(
    completions: Sequence[str],
    query_ids: Sequence[str],
    dataset_names: Sequence[str],
    **kwargs,
) -> Iterable[float]:
    """
    Compute rewards using SearchEvaluator.

    Called by TRL's GRPOTrainer to evaluate generated completions.
    Extracts boolean queries from model outputs and evaluates them.

    Args:
        completions: Model's generated responses (contains <query>...</query>)
        query_ids: Query IDs for looking up qrels
        dataset_names: Dataset names
        **kwargs: Additional args from trainer

    Returns:
        List of float rewards in [0, 1] range
    """
    config = get_config()
    index_dir = Path(config.paths.index_dir)
    evaluator = SearchEvaluator(index_path=str(index_dir))

    rewards = []
    for completion, query_id, dataset_name in zip(
        completions, query_ids, dataset_names
    ):
        # Parse query from model output
        query_match = re.search(r"<query>\s*(.*?)\s*</query>", completion, re.DOTALL)
        if not query_match or not query_match.group(1).strip():
            rewards.append(0.0)
            continue

        query_text = query_match.group(1).strip()

        # Evaluate using SearchEvaluator
        metrics, error = evaluator.evaluate_single_query(
            query_text=query_text,
            query_id=query_id,
            dataset_name=dataset_name,
            split=config.reward.split,
            k=config.reward.eval_k,
        )

        if error:
            rewards.append(0.0)
            continue

        # Combined reward: configurable weights for NDCG@10 + MRR
        reward = (
            config.reward.ndcg_weight * metrics["ndcg@10"]
            + config.reward.mrr_weight * metrics["mrr"]
        )
        rewards.append(float(reward))

    return rewards
