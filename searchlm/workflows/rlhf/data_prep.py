"""
Data preparation for GRPO training.

This module prepares training data from SciFact and NFCorpus datasets,
formatting them as HuggingFace Datasets with prompts and metadata for
reward computation.
"""

from pathlib import Path

from searchlm.config import get_config
from searchlm.prompts import SYSTEM_PROMPT

config = get_config()

# Data directory
DATA_DIR = Path(config.paths.data_dir)


def prep_dataset():
    """Prepare training data as HuggingFace Dataset."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    from searchlm import create_loader

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer for chat template
    print(f"Loading tokenizer for {config.model.name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    all_data = []

    # Load train + dev splits from both datasets
    for dataset_name in config.datasets.names:
        print(f"Loading {dataset_name}...")
        loader = create_loader(dataset_name)

        for split in ["train", "dev"]:
            dataset_split = loader.load_split(split=split)
            print(f"  {split}: {len(dataset_split.queries)} queries")

            for query_id, query in dataset_split.queries.items():
                # Format prompt with chat template
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Translate the following question into a "
                            f"boolean search query:\n{query.text}"
                        ),
                    },
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                all_data.append(
                    {
                        "prompt": prompt,
                        "query_id": query_id,  # For reward function
                        "dataset_name": dataset_name,  # For reward function
                    }
                )

    print(f"\nTotal examples: {len(all_data)}")

    # Create HF Dataset and shuffle
    dataset = Dataset.from_list(all_data)
    dataset = dataset.shuffle(seed=42)

    # Split train/test (90/10)
    split_idx = int(len(dataset) * 0.9)
    train_dataset = dataset.select(range(split_idx))
    test_dataset = dataset.select(range(split_idx, len(dataset)))

    # Save to disk
    print(f"\nSaving to {DATA_DIR}...")
    train_dataset.save_to_disk(str(DATA_DIR / "train"))
    test_dataset.save_to_disk(str(DATA_DIR / "test"))

    print(f"✓ Saved {len(train_dataset)} training examples")
    print(f"✓ Saved {len(test_dataset)} validation examples")

    # Print dataset breakdown
    train_df = train_dataset.to_pandas()
    print("\nDataset breakdown:")
    print(train_df["dataset_name"].value_counts())
