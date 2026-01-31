"""Data preparation for GRPO training."""

from datasets import Dataset

from searchlm import create_loader
from searchlm.config import get_config, get_data_path
from searchlm.prompts import create_chat_prompt


def prepare_training_data():
    """Prepare training data from SciFact and NFCorpus datasets."""
    config = get_config()
    datasets_dir = get_data_path("datasets")

    print("=" * 60)
    print("Preparing training data")
    print("=" * 60)

    # Ensure data directory exists
    datasets_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    # Load train split from both datasets
    for dataset_name in config.datasets.names:
        print(f"Loading {dataset_name}...")
        loader = create_loader(dataset_name)

        for split in ["train"]:
            dataset_split = loader.load_split(split=split)
            print(f"  {split}: {len(dataset_split.queries)} queries")

            for query_id, query in dataset_split.queries.items():
                prompt = create_chat_prompt(query.text)
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
    print(f"\nSaving to {datasets_dir}...")
    train_dataset.save_to_disk(str(datasets_dir / "train"))
    test_dataset.save_to_disk(str(datasets_dir / "test"))

    print(f"✓ Saved {len(train_dataset)} training examples")
    print(f"✓ Saved {len(test_dataset)} validation examples")

    # Print dataset breakdown
    train_df = train_dataset.to_pandas()
    print("\nDataset breakdown:")
    print(train_df["dataset_name"].value_counts())


if __name__ == "__main__":
    prepare_training_data()
