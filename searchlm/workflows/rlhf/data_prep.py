"""Data preparation for GRPO training."""

from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from searchlm import create_loader
from searchlm.config import get_config
from searchlm.prompts import create_chat_prompt


def prepare_training_data():
    """Prepare training data from SciFact and NFCorpus datasets."""
    config = get_config()
    data_dir = Path(config.paths.data_dir)

    print("=" * 60)
    print("Preparing training data")
    print("=" * 60)

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
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
                prompt = create_chat_prompt(query.text, tokenizer)
                all_data.append(
                    {
                        "prompt": prompt,
                        "query_id": query_id,
                        "dataset_name": dataset_name,
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
    print(f"\nSaving to {data_dir}...")
    train_dataset.save_to_disk(str(data_dir / "train"))
    test_dataset.save_to_disk(str(data_dir / "test"))

    print(f"✓ Saved {len(train_dataset)} training examples")
    print(f"✓ Saved {len(test_dataset)} validation examples")

    # Print dataset breakdown
    train_df = train_dataset.to_pandas()
    print("\nDataset breakdown:")
    print(train_df["dataset_name"].value_counts())
