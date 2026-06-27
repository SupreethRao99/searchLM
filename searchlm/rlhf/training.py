"""GRPO training with TRL and vLLM.

Supports two versions:
  v1  — original run (NFCorpus+SciFact, RewardFunction, output to models/)
  v2  — improved run (expanded datasets, RewardFunctionV2 with shaped reward,
         num_generations=8, output to models/grpo_v2/)
"""

import os
import subprocess

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from searchlm.config import get_config, get_data_path
from searchlm.rlhf.reward import RewardFunction


def train(use_vllm_server: bool = False, version: str = "v1"):
    """
    Train with GRPO using vLLM.

    Args:
        use_vllm_server: If True, use server mode (2 GPUs).
                         If False, use colocate mode (1 GPU).
        version: "v1" — original training configuration.
                 "v2" — shaped reward, expanded datasets, num_generations=8,
                         saved to models/grpo_v2/.
    """
    config = get_config()
    datasets_dir = get_data_path("datasets")
    models_dir = get_data_path("models")

    # Version-specific config and paths
    if version == "v2":
        train_cfg = config.training_v2
        dataset_key = "train_v2"
        output_dir = models_dir / "grpo_v2"
        # Prefer sft_v2 checkpoint, fall back to sft, then base model
        for candidate in ["sft_v2", "sft"]:
            sft_path = models_dir / candidate / "final"
            if sft_path.exists():
                base_model = str(sft_path)
                break
        else:
            base_model = config.model.name
    else:
        train_cfg = config.training
        dataset_key = "train"
        output_dir = models_dir
        sft_checkpoint = models_dir / "sft" / "final"
        base_model = (
            str(sft_checkpoint) if sft_checkpoint.exists() else config.model.name
        )

    mode = "server" if use_vllm_server else "colocate"
    print("=" * 60)
    print(f"GRPO training [{version}] — vLLM {mode} mode")
    print(f"Base model:   {base_model}")
    print(f"Output dir:   {output_dir}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_vllm_server:
        print("Starting vLLM server on GPU 0...")
        env_copy = os.environ.copy()
        env_copy["CUDA_VISIBLE_DEVICES"] = "0"
        subprocess.Popen(
            ["trl", "vllm-serve", "--model", base_model],
            env=env_copy,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ.update(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12355",
            }
        )

    print(f"Loading dataset ({dataset_key})...")
    train_dataset = load_from_disk(str(datasets_dir / dataset_key))
    print(f"Loaded {len(train_dataset)} training examples")

    batch_size = (
        train_cfg.batch_size.server
        if use_vllm_server
        else train_cfg.batch_size.colocate
    )
    grad_accum = (
        train_cfg.gradient_accumulation_steps.server
        if use_vllm_server
        else train_cfg.gradient_accumulation_steps.colocate
    )
    run_name = (
        train_cfg.wandb_run_name.server
        if use_vllm_server
        else train_cfg.wandb_run_name.colocate
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        temperature=config.model.temperature,
        num_generations=train_cfg.num_generations,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        bf16=(train_cfg.precision == "bf16"),
        torch_compile=getattr(train_cfg, "torch_compile", False),
        use_vllm=True,
        vllm_mode="server" if use_vllm_server else "colocate",
        vllm_gpu_memory_utilization=(
            train_cfg.vllm_gpu_memory_utilization if not use_vllm_server else None
        ),
        vllm_max_model_length=getattr(train_cfg, "vllm_max_model_length", None),
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        run_name=run_name,
    )

    # Select reward function
    print("Initializing reward function...")
    if version == "v2":
        from searchlm.rlhf.reward_v2 import RewardFunctionV2

        reward_fn = RewardFunctionV2(train_dataset)
    else:
        reward_fn = RewardFunction(train_dataset)

    trainer = GRPOTrainer(
        model=base_model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
    )

    # Resume from checkpoint if one exists in the output dir
    grpo_checkpoints = (
        sorted(
            [
                d
                for d in output_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if output_dir.exists()
        else []
    )
    resume = grpo_checkpoints[-1] if grpo_checkpoints else None
    if resume:
        print(f"Resuming from {resume}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    print("Saving final model...")
    trainer.save_model(str(output_dir / "final"))
    print(f"Training complete! Model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    train()
