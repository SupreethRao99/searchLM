"""Upload latest model checkpoint to Hugging Face Hub.

This utility:
- Finds the latest checkpoint from training
- Creates a comprehensive model card with documentation
- Uploads the model as a private repository to Hugging Face
- Includes training configuration and evaluation metrics
"""

import os
import re
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo

from searchlm.config import get_config, get_data_path


def find_latest_checkpoint(models_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the models directory.

    Args:
        models_dir: Path to models directory

    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoints = []

    # Look for checkpoint directories (e.g., checkpoint-100, checkpoint-150)
    for item in models_dir.glob("checkpoint-*"):
        if item.is_dir():
            # Extract step number
            match = re.search(r"checkpoint-(\d+)", item.name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, item))

    if not checkpoints:
        # Check if "final" checkpoint exists
        final_checkpoint = models_dir / "final"
        if final_checkpoint.exists() and final_checkpoint.is_dir():
            return final_checkpoint
        return None

    # Return checkpoint with highest step number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def create_model_card(
    base_model: str,
    checkpoint_path: Path,
    training_config: dict,
) -> str:
    """Create a comprehensive model card for the uploaded model.

    Args:
        base_model: Name of the base model
        checkpoint_path: Path to the checkpoint
        training_config: Training configuration dictionary

    Returns:
        Model card content as markdown string
    """
    checkpoint_name = checkpoint_path.name

    model_card = f"""---
language:
- en
license: mit
base_model: {base_model}
tags:
- information-retrieval
- search
- reinforcement-learning
- RLHF
- GRPO
- query-generation
- boolean-search
library_name: transformers
pipeline_tag: text-generation
---

# SearchLM: RLHF-Trained Search Query Generator

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) trained using **Group Relative Policy Optimization (GRPO)** with verifiable rewards for generating better boolean search queries.

## Model Description

SearchLM uses Reinforcement Learning with Verifiable Rewards (RLVR) to train language models to generate effective boolean search queries for information retrieval tasks. The model is optimized using real search evaluation metrics (NDCG and MRR) as rewards.

- **Base Model**: {base_model}
- **Training Method**: Group Relative Policy Optimization (GRPO)
- **Checkpoint**: {checkpoint_name}
- **Reward Function**: Weighted combination of NDCG and MRR from actual search results
- **Datasets**: NFCorpus and SciFact (MTEB)
- **Task**: Boolean search query generation

## Training Details

### Training Configuration

- **Learning Rate**: {training_config.get("learning_rate", "N/A")}
- **Epochs**: {training_config.get("num_epochs", "N/A")}
- **Batch Size**: {training_config.get("batch_size", {}).get("colocate", "N/A")} (colocate) / {training_config.get("batch_size", {}).get("server", "N/A")} (server)
- **Gradient Accumulation**: {training_config.get("gradient_accumulation_steps", {}).get("colocate", "N/A")} (colocate) / {training_config.get("gradient_accumulation_steps", {}).get("server", "N/A")} (server)
- **Precision**: {training_config.get("precision", "N/A")}
- **Gradient Checkpointing**: {training_config.get("gradient_checkpointing", "N/A")}
- **Max New Tokens**: {training_config.get("max_new_tokens", "N/A")}
- **Num Generations**: {training_config.get("num_generations", "N/A")}

### Reward Function

- **NDCG Weight**: {training_config.get("reward", {}).get("ndcg_weight", "N/A")}
- **MRR Weight**: {training_config.get("reward", {}).get("mrr_weight", "N/A")}
- **Evaluation K**: {training_config.get("reward", {}).get("eval_k", "N/A")}

## Usage

### Prerequisites

```bash
uv add transformers torch vllm trl[vllm] datasets omegaconf
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "supreethrao/searchlm-qwen2.5-3b-rlhf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# System prompt for query generation
system_prompt = \"\"\"You are a search expert. Given a user's information need, generate an effective boolean search query using AND, OR, NOT operators and parentheses for grouping. The query should be precise and retrieve relevant documents.

Guidelines:
- Use AND to require multiple terms
- Use OR for synonyms or alternatives
- Use NOT to exclude irrelevant terms
- Use parentheses for grouping complex logic
- Keep queries focused and not overly complex

Format your response with the query inside <query></query> tags.\"\"\"

# User query
user_query = "What are the latest treatments for Type 2 diabetes?"

messages = [
    {{"role": "system", "content": system_prompt}},
    {{"role": "user", "content": user_query}}
]

# Generate search query
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
)
outputs = model.generate(inputs, max_new_tokens=1024, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### Using with vLLM (Recommended)

```python
from vllm import LLM, SamplingParams

model_name = "YOUR_USERNAME/YOUR_MODEL_NAME"
llm = LLM(model=model_name)

system_prompt = \"\"\"You are a search expert. Given a user's information need, generate an effective boolean search query...\"\"\"

prompts = [
    f"<|im_start|>system\\n{{system_prompt}}<|im_end|>\\n<|im_start|>user\\nWhat are the latest treatments for Type 2 diabetes?<|im_end|>\\n<|im_start|>assistant\\n"
]

sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Evaluation

The model is evaluated on standard information retrieval datasets (NFCorpus and SciFact) using the following metrics:

- **NDCG@10, NDCG@100**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Precision@10**: Precision at top 10 results
- **Recall@10**: Recall at top 10 results
- **MAP**: Mean Average Precision

## Training Data

The model was trained on:
- **NFCorpus**: Medical information retrieval dataset
- **SciFact**: Scientific fact-checking dataset

Both datasets are from the MTEB (Massive Text Embedding Benchmark) collection.

## Limitations and Bias

- The model is specifically trained for scientific and medical domains (NFCorpus and SciFact)
- Performance may vary on other domains
- Boolean query syntax is optimized for full-text search engines (e.g., Tantivy)
- Generated queries may need domain-specific tuning for production use

## Citation

If you use this model, please cite:

```bibtex
@misc{{searchlm2025,
  author = {{Supreeth Rao}},
  title = {{SearchLM: Reinforcement Learning with Verifiable Rewards for Search Query Generation}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_NAME}}}}
}}
```

## License

MIT License

## Contact

For questions or issues:
- Email: raosupreeth00@gmail.com
- GitHub: [SearchLM Repository](https://github.com/YOUR_USERNAME/searchLM)

## Acknowledgments

- Base model: [{base_model}](https://huggingface.co/{base_model})
- Training framework: [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- Inference engine: [vLLM](https://github.com/vllm-project/vllm)
- Search engine: [Tantivy](https://github.com/quickwit-oss/tantivy)
"""

    return model_card


def upload_to_huggingface(
    repo_name: str,
    checkpoint_path: Path,
    private: bool = True,
    organization: Optional[str] = None,
) -> str:
    """Upload model checkpoint to Hugging Face Hub.

    Args:
        repo_name: Name for the repository on HuggingFace
        checkpoint_path: Path to the model checkpoint
        private: Whether to make the repository private (default: True)
        organization: Optional organization name (if None, uses personal account)

    Returns:
        URL of the created repository
    """
    config = get_config()

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Check for required files in checkpoint
    required_files = ["config.json", "model.safetensors"]
    missing_files = []

    for file in required_files:
        if not (checkpoint_path / file).exists():
            # Check for alternative formats
            if file == "model.safetensors":
                # Check for pytorch_model.bin as alternative
                if not (checkpoint_path / "pytorch_model.bin").exists():
                    missing_files.append(file)
            else:
                missing_files.append(file)

    if missing_files:
        print(f"Warning: Missing files in checkpoint: {missing_files}")
        print("Continuing anyway, but upload may fail if files are required.")

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment. Please set it in your .env file or export it."
        )

    # Initialize HF API
    api = HfApi(token=hf_token)

    # Create full repo ID
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        # Get username from API
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{repo_name}"

    print(f"\n{'=' * 60}")
    print("Uploading model to Hugging Face")
    print(f"{'=' * 60}")
    print(f"Repository: {repo_id}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Private: {private}")
    print(f"{'=' * 60}\n")

    # Create repository
    print("Creating repository...")
    repo_url = create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
        token=hf_token,
    )
    print(f"✓ Repository created: {repo_url}")

    # Create model card
    print("\nCreating model card...")
    training_config = {
        "learning_rate": config.training.learning_rate,
        "num_epochs": config.training.num_epochs,
        "batch_size": {
            "colocate": config.training.batch_size.colocate,
            "server": config.training.batch_size.server,
        },
        "gradient_accumulation_steps": {
            "colocate": config.training.gradient_accumulation_steps.colocate,
            "server": config.training.gradient_accumulation_steps.server,
        },
        "precision": config.training.precision,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "max_new_tokens": config.training.max_new_tokens,
        "num_generations": config.training.num_generations,
        "reward": {
            "ndcg_weight": config.reward.ndcg_weight,
            "mrr_weight": config.reward.mrr_weight,
            "eval_k": config.reward.eval_k,
        },
    }

    model_card = create_model_card(
        base_model=config.model.name,
        checkpoint_path=checkpoint_path,
        training_config=training_config,
    )

    # Write model card to checkpoint directory temporarily
    model_card_path = checkpoint_path / "README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)
    print("✓ Model card created")

    # Upload the folder
    print("\nUploading model files...")
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )
    print("✓ Model files uploaded")

    # Clean up temporary model card
    model_card_path.unlink()

    print(f"\n{'=' * 60}")
    print("✓ Upload complete!")
    print(f"{'=' * 60}")
    print(f"Repository URL: {repo_url}")
    print("\nYou can view your model at:")
    print(f"https://huggingface.co/{repo_id}")
    print(f"{'=' * 60}\n")

    return repo_url


def main():
    """Main function to upload latest checkpoint."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload latest model checkpoint to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Name for the repository on HuggingFace (e.g., 'searchlm-qwen2.5-3b-rlhf')",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to specific checkpoint (default: auto-detect latest)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public (default: private)",
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Upload to an organization instead of personal account",
    )

    args = parser.parse_args()

    # Get models directory from config
    models_dir = get_data_path("models")

    # Find checkpoint
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = models_dir / checkpoint_path
    else:
        print("Searching for latest checkpoint...")
        checkpoint_path = find_latest_checkpoint(models_dir)

        if checkpoint_path is None:
            print(f"\n❌ No checkpoints found in {models_dir}")
            print("\nPlease ensure you have:")
            print("1. Completed model training")
            print("2. Checkpoints saved in the models directory")
            print("3. Or specify a checkpoint path with --checkpoint-path")
            return

        print(f"✓ Found latest checkpoint: {checkpoint_path.name}")

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return

    # Upload to HuggingFace
    try:
        upload_to_huggingface(
            repo_name=args.repo_name,
            checkpoint_path=checkpoint_path,
            private=not args.public,
            organization=args.organization,
        )
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
