"""GRPO training with TRL and vLLM."""

import os
import subprocess

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from searchlm.config import get_config, get_data_path
from searchlm.workflows.rlhf.reward import RewardFunction


def train(use_vllm_server: bool = False):
    """
    Train with GRPO using vLLM.

    Args:
        use_vllm_server: If True, use server mode (2 GPUs).
                        If False, use colocate mode (1 GPU).
    """
    config = get_config()
    datasets_dir = get_data_path("datasets")
    models_dir = get_data_path("models")

    print("=" * 60)
    mode = "server" if use_vllm_server else "colocate"
    print(f"Training with vLLM {mode} mode")
    print("=" * 60)

    # Ensure directories exist
    models_dir.mkdir(parents=True, exist_ok=True)

    if use_vllm_server:
        # Start vLLM server on GPU 0
        print("Starting vLLM server on GPU 0...")
        env_copy = os.environ.copy()
        env_copy["CUDA_VISIBLE_DEVICES"] = "0"
        subprocess.Popen(
            ["trl", "vllm-serve", "--model", config.model.name],
            env=env_copy,
        )
        # Run training on GPU 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        # Set up distributed training env vars for colocate mode
        os.environ.update(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12355",
            }
        )

    print("Loading dataset...")
    train_dataset = load_from_disk(str(datasets_dir / "train"))
    print(f"Loaded {len(train_dataset)} training examples")

    # Configure training
    print(f"Configuring GRPO training ({mode} mode)...")
    batch_config = (
        config.training.batch_size.server
        if use_vllm_server
        else config.training.batch_size.colocate
    )
    grad_accum = (
        config.training.gradient_accumulation_steps.server
        if use_vllm_server
        else config.training.gradient_accumulation_steps.colocate
    )
    run_name = (
        config.training.wandb_run_name.server
        if use_vllm_server
        else config.training.wandb_run_name.colocate
    )

    training_args = GRPOConfig(
        output_dir=str(models_dir),
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=batch_config,
        gradient_accumulation_steps=grad_accum,
        temperature=config.model.temperature,
        num_generations=config.training.num_generations,
        gradient_checkpointing=config.training.gradient_checkpointing,
        bf16=(config.training.precision == "bf16"),
        use_vllm=True,
        vllm_mode="server" if use_vllm_server else "colocate",
        vllm_gpu_memory_utilization=(
            config.training.vllm_gpu_memory_utilization if not use_vllm_server else None
        ),
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        run_name=run_name,
    )

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    reward_fn = RewardFunction(train_dataset)
    trainer = GRPOTrainer(
        model=config.model.name,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
    )

    # Train!
    print("Starting training...")
    trainer.train(resume_from_checkpoint=True)

    # Save final checkpoint
    print("Saving final model...")
    trainer.save_model(str(models_dir / "final"))

    print("Training complete!")

if __name__ == "__main__":
    train()
