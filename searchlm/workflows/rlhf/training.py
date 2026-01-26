"""
GRPO training with TRL and vLLM for boolean query generation.

This module trains a language model using GRPO (Group Relative Policy Optimization)
with verifiable rewards from SearchEvaluator.
"""

import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from searchlm.config import get_config

config = get_config()

# Directories
DATA_DIR = Path(config.paths.data_dir)
MODELS_DIR = Path(config.paths.models_dir)
INDEX_DIR = Path(config.paths.index_dir)


# ============================================================================
# Reward Function
# ============================================================================


def reward_function(
    completions: Sequence[str],
    query_ids: Sequence[str],
    dataset_names: Sequence[str],
    **kwargs,
) -> Iterable[float]:
    """
    Compute rewards using SearchEvaluator.

    This function is called by TRL's GRPOTrainer during training to evaluate
    generated completions. It extracts boolean queries from model outputs and
    evaluates them using the search engine to get NDCG@10 and MRR metrics.

    Args:
        completions: Model's generated responses (contains <query>...</query>)
        query_ids: Query IDs for looking up qrels (from dataset)
        dataset_names: Dataset names (from dataset)
        **kwargs: Additional args from trainer

    Returns:
        List of float rewards in [0, 1] range
    """
    from searchlm import SearchEvaluator

    # Initialize evaluator (will be cached by Python)
    evaluator = SearchEvaluator(index_path=str(INDEX_DIR))

    rewards = []

    for completion, query_id, dataset_name in zip(
        completions, query_ids, dataset_names
    ):
        # 1. Parse query from model output
        query_match = re.search(r"<query>\s*(.*?)\s*</query>", completion, re.DOTALL)
        if not query_match:
            rewards.append(0.0)
            continue

        query_text = query_match.group(1).strip()
        if not query_text:
            rewards.append(0.0)
            continue

        # 2. Evaluate using SearchEvaluator
        metrics, error = evaluator.evaluate_single_query(
            query_text=query_text,
            query_id=query_id,
            dataset_name=dataset_name,
            split=config.reward.split,
            k=config.reward.eval_k,
        )

        if error:
            # Query has syntax error or other issue
            rewards.append(0.0)
            continue

        # 3. Combined reward: configurable weights for NDCG@10 + MRR
        # Both metrics are already in [0, 1] range
        reward = (
            config.reward.ndcg_weight * metrics["ndcg@10"]
            + config.reward.mrr_weight * metrics["mrr"]
        )
        rewards.append(float(reward))

    return rewards


# ============================================================================
# Training Functions
# ============================================================================


def train():
    """
    Train with vLLM colocate mode (vLLM shares GPU with training).

    This is the simpler setup requiring only 1 GPU. vLLM and the training
    model share GPU memory, which is managed by setting vllm_gpu_memory_utilization.
    """
    from datasets import load_from_disk
    from trl import GRPOConfig, GRPOTrainer

    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Set up distributed training env vars for colocate mode
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    print("Loading dataset...")
    train_dataset = load_from_disk(str(DATA_DIR / "train"))
    print(f"Loaded {len(train_dataset)} training examples")

    # Configure training
    print("Configuring GRPO training...")
    training_args = GRPOConfig(
        output_dir=str(MODELS_DIR),
        # GRPO hyperparameters
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size.colocate,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps.colocate,
        # Generation settings
        max_new_tokens=config.training.max_new_tokens,
        temperature=config.model.temperature,
        num_generations=config.training.num_generations,
        # Optimization
        gradient_checkpointing=config.training.gradient_checkpointing,
        bf16=(config.training.precision == "bf16"),
        # vLLM acceleration (colocate mode - shares GPU with training)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=config.training.vllm_gpu_memory_utilization,
        # Logging and checkpointing
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        report_to="wandb",
        run_name=config.training.wandb_run_name.colocate,
    )

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=config.model.name,
        args=training_args,
        reward_funcs=reward_function,
        train_dataset=train_dataset,
    )

    # Train!
    print("Starting training...")
    trainer.train()

    # Save final checkpoint
    print("Saving final model...")
    trainer.save_model(str(MODELS_DIR / "final"))

    print("Training complete!")


def train_with_vllm_server():
    """
    Train with vLLM server mode (vLLM on separate GPU).

    This setup uses 2 GPUs: one for vLLM generation and one for training.
    This provides better throughput but requires more resources.
    """
    import subprocess

    from datasets import load_from_disk
    from trl import GRPOConfig, GRPOTrainer

    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Start vLLM server on GPU 0 in background
    print("Starting vLLM server on GPU 0...")
    env_copy = os.environ.copy()
    env_copy["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.Popen(
        ["trl", "vllm-serve", "--model", config.model.name],
        env=env_copy,
    )

    # Run training on GPU 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Loading dataset...")
    train_dataset = load_from_disk(str(DATA_DIR / "train"))
    print(f"Loaded {len(train_dataset)} training examples")

    # Configure training
    print("Configuring GRPO training (server mode)...")
    training_args = GRPOConfig(
        output_dir=str(MODELS_DIR),
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size.server,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps.server,
        max_new_tokens=config.training.max_new_tokens,
        temperature=config.model.temperature,
        num_generations=config.training.num_generations,
        gradient_checkpointing=config.training.gradient_checkpointing,
        bf16=(config.training.precision == "bf16"),
        # vLLM server mode
        use_vllm=True,
        vllm_mode="server",  # vLLM runs on separate GPU
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        report_to="wandb",
        run_name=config.training.wandb_run_name.server,
    )

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=config.model.name,
        args=training_args,
        reward_funcs=reward_function,
        train_dataset=train_dataset,
    )

    # Train!
    print("Starting training...")
    trainer.train()

    # Save final checkpoint
    print("Saving final model...")
    trainer.save_model(str(MODELS_DIR / "final"))

    print("Training complete!")
