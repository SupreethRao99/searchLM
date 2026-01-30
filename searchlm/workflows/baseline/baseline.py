"""Baseline query generation using instruction-tuned LLMs."""

import csv
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from searchlm.config import get_config
from searchlm.inference import VllmEngine
from searchlm.prompts import create_chat_prompt


@dataclass
class SearchQuery:
    """Search query with original text and generated query."""

    id: str
    text: str
    query: str | None = None


class BaselineGenerator:
    """Generate baseline queries using instruction-tuned LLMs."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        dataset_name: str = "mteb/scifact",
        subset: str = "queries",
        split: str = "queries",
        output_filepath: str = "./data/scifact_generated_queries.tsv",
    ):
        """
        Initialize the baseline generator.

        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset
            split: Dataset split
            output_filepath: Output TSV filepath
        """
        self.config = get_config()
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.output_filepath = output_filepath
        self.engine = VllmEngine(model_name=model_name)

    def load_dataset(self) -> list[SearchQuery]:
        """Load queries from HuggingFace dataset."""
        print(
            f"Loading dataset {self.dataset_name}, subset={self.subset}, "
            f"split={self.split}"
        )
        dataset = load_dataset(self.dataset_name, self.subset, split=self.split)

        queries = [SearchQuery(id=item["_id"], text=item["text"]) for item in dataset]

        print(f"Loaded {len(queries)} queries from dataset")
        return queries

    def save_results(self, results: list[SearchQuery]) -> str:
        """Save results to TSV file."""
        print(f"Saving {len(results)} results to {self.output_filepath}")

        Path(self.output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "text", "query"])
            for result in results:
                writer.writerow([result.id, result.text, result.query])

        print(f"Results saved to {self.output_filepath}")
        return str(self.output_filepath)

    def generate(self) -> str:
        """
        Generate baseline queries for the configured dataset.

        Returns:
            Path to the saved results file
        """
        # Load dataset
        queries = self.load_dataset()

        # Create prompts and send to LLM (engine handles batching internally)
        prompts = [create_chat_prompt(q.text) for q in queries]
        responses = self.engine.generate(
            prompts, max_tokens=self.config.baseline.max_tokens
        )
        for response, query in zip(responses, queries):
            query.query = response

        # Save results
        print(f"\nSaving {len(queries)} results...")
        output_filepath = self.save_results(queries)

        print(f"\n{'=' * 60}")
        print("✓ Processing complete!")
        print(f"✓ Results saved to: {output_filepath}")
        print(f"✓ Total queries processed: {len(queries)}")
        print(f"{'=' * 60}\n")

        return output_filepath


def main():
    """
    Run baseline query generation.

    Examples:
        # Generate queries for SciFact (default):
        python -m searchlm.workflows.baseline.baseline

        # For NFCorpus, modify the class instantiation in this function
    """
    config = get_config()
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = BaselineGenerator(
        dataset_name="mteb/scifact",
        subset="queries",
        split="queries",
        output_filepath=str(output_dir / "scifact_generated_queries.tsv"),
    )
    generator.generate()

    # # Uncomment for NFCorpus:
    # generator = BaselineGenerator(
    #     dataset_name="mteb/nfcorpus",
    #     subset="queries",
    #     split="queries",
    #     output_filepath=str(output_dir / "nfcorpus_generated_queries.tsv"),
    # )
    # generator.generate()


if __name__ == "__main__":
    main()
