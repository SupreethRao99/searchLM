"""SFT warm-start training for NL2BM25.

Trains Qwen2.5-3B-Instruct to generate <reasoning>...</reasoning><query>...</query>
format using LoRA, then merges the adapter into the base model so GRPO can start
from a full checkpoint without any PEFT overhead.

One epoch is enough — the goal is format acquisition, not perfect retrieval.
GRPO handles the retrieval quality via live reward signal.
"""

from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from searchlm.config import get_config, get_data_path


def train(dataset_source: str = "hub", version: str = "v1"):
    """
    LoRA SFT on the nl2bm25-sft dataset, followed by adapter merge.

    Args:
        dataset_source: "hub" loads from HuggingFace Hub;
                        "local" loads from modal_data/datasets/sft.
        version: "v1" — syntax-valid filter, saves to models/sft/;
                 "v2" — also filters ndcg_at_10 > threshold, saves to models/sft_v2/.
    """
    config = get_config()
    models_dir = get_data_path("models")
    run_name = f"sft_{version}"
    adapter_dir = models_dir / run_name / "adapter"
    merged_dir = models_dir / run_name / "final"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Use version-specific config if available, fall back to base sft config
    sft_cfg = (
        getattr(config, f"sft_{version}", config.sft) if version != "v1" else config.sft
    )

    print("=" * 60)
    print(f"SFT warm-start (LoRA) [{version}]: NL2BM25 boolean query generation")
    print("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    hub_id = getattr(sft_cfg, "dataset_hub_id", config.sft.dataset_hub_id)
    print(f"\nLoading dataset ({dataset_source})...")
    if dataset_source == "hub":
        dataset = load_dataset(hub_id, split="train")
    else:
        datasets_dir = get_data_path("datasets")
        dataset = load_from_disk(str(datasets_dir / "sft"))

    before = len(dataset)
    dataset = dataset.filter(
        lambda x: x["syntax_valid"] is not False and x["boolean_query"] is not None
    )
    print(f"Filtered {before} → {len(dataset)} examples (syntax-valid)")

    # v2: additionally filter on retrieval quality — only keep examples where the
    # generated query actually retrieved at least one relevant document
    if version == "v2":
        ndcg_threshold = getattr(sft_cfg, "ndcg_threshold", 0.0)
        before = len(dataset)
        dataset = dataset.filter(
            lambda x: x.get("ndcg_at_10") is not None
            and x["ndcg_at_10"] > ndcg_threshold
        )
        print(
            f"Filtered {before} → {len(dataset)} examples (ndcg_at_10 > {ndcg_threshold})"
        )

    dataset = dataset.select_columns(["messages"])

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.chat_template is None:
        raise ValueError(f"{config.model.name} has no chat template")

    # ── LoRA config ───────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=sft_cfg.lora_r,
        lora_alpha=sft_cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── SFT training config ───────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=sft_cfg.num_epochs,
        per_device_train_batch_size=sft_cfg.batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        warmup_ratio=sft_cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=sft_cfg.logging_steps,
        save_steps=sft_cfg.save_steps,
        save_total_limit=1,
        max_length=sft_cfg.max_seq_length,
        packing=False,
        dataset_text_field=None,
        run_name=sft_cfg.wandb_run_name,
        report_to="wandb" if sft_cfg.get("use_wandb", True) else "none",
    )

    print(
        f"\nModel:   {config.model.name}  (LoRA r={sft_cfg.lora_r}, α={sft_cfg.lora_alpha})"
    )
    print(f"Dataset: {len(dataset)} examples")
    print(
        f"Epochs:  {sft_cfg.num_epochs}  |  LR: {sft_cfg.learning_rate}  |  BS: {sft_cfg.batch_size} × {sft_cfg.gradient_accumulation_steps} grad accum"
    )
    print(f"Adapter: {adapter_dir}")
    print(f"Merged:  {merged_dir}\n")

    trainer = SFTTrainer(
        model=config.model.name,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting LoRA SFT training...")
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nAdapter saved to {adapter_dir}")

    # ── Merge LoRA into base and save full model ──────────────────────────────
    # GRPO needs a plain HF model, not a PEFT wrapper
    print("\nMerging LoRA adapter into base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir}")
    print("\nSFT complete. GRPO will start from the merged checkpoint.")


if __name__ == "__main__":
    import sys

    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    train(version=ver)
