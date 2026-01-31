"""Reward function for GRPO training."""

import re
from typing import Iterable, Sequence

from searchlm import SearchEvaluator
from searchlm.config import get_config, get_data_path


class RewardFunction:
    """
    Reward function wrapper that accesses dataset metadata.

    TRL's GRPOTrainer calls the reward function with completions and prompts,
    but doesn't automatically pass additional dataset columns. This wrapper
    maintains a mapping from prompts to metadata.
    """

    # Add __name__ attribute for GRPOTrainer logging
    __name__ = "reward_function"

    def __init__(self, dataset):
        """
        Initialize with the training dataset.

        Args:
            dataset: HuggingFace Dataset with 'prompt', 'query_id', 'dataset_name' columns
        """
        self.config = get_config()
        indices_dir = get_data_path("indices")
        self.evaluator = SearchEvaluator(index_path=str(indices_dir))

        # Create mapping from prompt to metadata
        self.prompt_to_metadata = {}

        # Cache qrels per dataset/split to avoid reloading
        # Key: (dataset_name, split), Value: qrels dict
        self.qrels_cache = {}

        for example in dataset:
            # Store as tuple: handle both dict and dataset formats
            prompt = (
                example["prompt"] if isinstance(example, dict) else example["prompt"]
            )
            query_id = (
                example["query_id"]
                if isinstance(example, dict)
                else example["query_id"]
            )
            dataset_name = (
                example["dataset_name"]
                if isinstance(example, dict)
                else example["dataset_name"]
            )

            # Use prompt as key (may need to handle list format)
            if isinstance(prompt, list):
                # Convert chat format to string key
                prompt_key = str(prompt)
            else:
                prompt_key = prompt

            self.prompt_to_metadata[prompt_key] = {
                "query_id": query_id,
                "dataset_name": dataset_name,
            }

        # Pre-load qrels for all datasets/splits used in training
        print("Pre-loading qrels for training...")
        unique_dataset_splits = set(
            (example["dataset_name"], self.config.reward.split) for example in dataset
        )
        for dataset_name, split in unique_dataset_splits:
            cache_key = (dataset_name, split)
            print(f"Loading qrels for {dataset_name} ({split} split)...")
            self.qrels_cache[cache_key] = self.evaluator.load_qrels(dataset_name, split)
        print("✓ Qrels pre-loaded and cached")

    def __call__(
        self,
        completions: Sequence[str | list],
        prompts: Sequence[str | list],
        **kwargs,
    ) -> Iterable[float]:
        """
        Compute rewards using SearchEvaluator.

        Called by TRL's GRPOTrainer to evaluate generated completions.
        Extracts boolean queries from model outputs and evaluates them.

        Args:
            completions: Model's generated responses (contains <query>...</query>)
                        Can be strings or lists (chat format)
            prompts: Original prompts (used to look up metadata)
            **kwargs: Additional args from trainer

        Returns:
            List of float rewards in [0, 1] range
        """
        rewards = []
        for completion, prompt in zip(completions, prompts):
            # Convert completion to string if it's a list (chat format)
            if isinstance(completion, list):
                # Chat format: extract the assistant's message
                completion_text = ""
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        completion_text = msg.get("content", "")
                        break
                # If no assistant message found, join all content
                if not completion_text:
                    completion_text = " ".join(
                        msg.get("content", "") if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
            else:
                completion_text = completion

            # Look up metadata for this prompt
            prompt_key = str(prompt) if isinstance(prompt, list) else prompt
            metadata = self.prompt_to_metadata.get(prompt_key)

            if not metadata:
                rewards.append(0.0)
                continue

            query_id = metadata["query_id"]
            dataset_name = metadata["dataset_name"]

            # Parse query from model output
            query_match = re.search(
                r"<query>\s*(.*?)\s*</query>", completion_text, re.DOTALL
            )
            if not query_match or not query_match.group(1).strip():
                rewards.append(0.0)
                continue

            query_text = query_match.group(1).strip()

            # Get cached qrels for this dataset/split
            cache_key = (dataset_name, self.config.reward.split)
            qrels_all = self.qrels_cache.get(cache_key, {})
            qrels = qrels_all.get(query_id, {})

            if not qrels:
                rewards.append(0.0)
                continue

            # Evaluate using cached qrels (avoid reloading dataset)
            metrics, error = self.evaluator.evaluate_query(
                query_text=query_text,
                qrels=qrels,
                k=self.config.reward.eval_k,
                dataset_filter=dataset_name,
            )

            if error:
                rewards.append(0.0)
                continue

            # Combined reward: configurable weights for NDCG@10 + MRR
            reward = (
                self.config.reward.ndcg_weight * metrics["ndcg@10"]
                + self.config.reward.mrr_weight * metrics["mrr"]
            )
            rewards.append(float(reward))

        return rewards


# Legacy function signature for backwards compatibility
def reward_function(
    completions: Sequence[str],
    query_ids: Sequence[str],
    dataset_names: Sequence[str],
    **kwargs,
) -> Iterable[float]:
    """
    Legacy reward function (deprecated).

    Use RewardFunction class instead for proper integration with GRPOTrainer.
    """
    config = get_config()
    indices_dir = get_data_path("indices")
    evaluator = SearchEvaluator(index_path=str(indices_dir))

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
