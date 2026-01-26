"""Baseline query generation using instruction-tuned LLMs."""

import csv
import time
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
        dataset_name: str = "mteb/scifact",
        subset: str = "queries",
        split: str = "queries",
        batch_size: int | None = None,
        output_filename: str = "scifact_generated_queries.tsv",
    ):
        """
        Initialize the baseline generator.

        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset
            split: Dataset split
            batch_size: Number of queries per batch (uses config default if None)
            output_filename: Output TSV filename
        """
        self.config = get_config()
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.batch_size = batch_size or self.config.baseline.batch_size
        self.output_filename = output_filename
        self.output_root = Path(self.config.paths.output_dir)

    def load_dataset(self) -> list[SearchQuery]:
        """Load queries from HuggingFace dataset."""
        print(
            f"Loading dataset {self.dataset_name}, subset={self.subset}, split={self.split}"
        )
        dataset = load_dataset(self.dataset_name, self.subset, split=self.split)

        queries = [SearchQuery(id=item["_id"], text=item["text"]) for item in dataset]

        print(f"Loaded {len(queries)} queries from dataset")
        return queries

    def create_batches(self, queries: list[SearchQuery]) -> list[list[SearchQuery]]:
        """Split queries into batches for processing."""
        batches = [
            queries[i : i + self.batch_size]
            for i in range(0, len(queries), self.batch_size)
        ]
        print(f"Created {len(batches)} batches from {len(queries)} queries")
        return batches

    def process_batch(
        self, engine: VllmEngine, queries: list[SearchQuery]
    ) -> list[SearchQuery]:
        """Process a batch of queries through the LLM."""
        # Build prompts
        prompts = [
            create_chat_prompt(query.text, engine.tokenizer) for query in queries
        ]

        # Generate responses
        start = time.time()
        responses = engine.generate(prompts, max_tokens=self.config.baseline.max_tokens)
        duration_s = time.time() - start

        print(f"Generated {len(responses)} responses in {int(duration_s)} seconds")

        # Attach responses to queries
        for response, query in zip(responses, queries):
            query.query = response

        return queries

    def save_results(self, results: list[SearchQuery]) -> str:
        """Save results to TSV file."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        output_path = self.output_root / self.output_filename

        print(f"Saving {len(results)} results to {output_path}")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "text", "query"])
            for result in results:
                writer.writerow([result.id, result.text, result.query])

        print(f"Results saved to {self.output_filename}")
        return str(output_path)

    def generate(self) -> str:
        """
        Generate baseline queries for the configured dataset.

        Returns:
            Path to the saved results file
        """
        # Load dataset
        print(f"Extracting data from {self.dataset_name}")
        queries = self.load_dataset()

        # Create batches
        print("Transforming data into batches")
        query_batches = self.create_batches(queries)

        n_queries = sum(len(batch) for batch in query_batches)
        print(f"Processing {n_queries} queries in {len(query_batches)} batches")

        # Process batches with LLM
        all_results = []
        with VllmEngine() as engine:
            for i, batch in enumerate(query_batches):
                print(f"Processing batch {i + 1}/{len(query_batches)}")
                results = self.process_batch(engine, batch)
                all_results.extend(results)

        # Save results
        print(f"\nSaving {len(all_results)} results...")
        output_path = self.save_results(all_results)

        print(f"\n{'=' * 60}")
        print("✓ Processing complete!")
        print(f"✓ Results saved to: {output_path}")
        print(f"✓ Total queries processed: {len(all_results)}")
        print(f"{'=' * 60}\n")

        return output_path


def main():
    """
    Run baseline query generation.

    Examples:
        # Generate queries for SciFact (default):
        python -m searchlm.workflows.baseline.baseline

        # For NFCorpus, modify the class instantiation in this function
    """
    # SciFact (default)
    generator = BaselineGenerator(
        dataset_name="mteb/scifact",
        subset="queries",
        split="queries",
        output_filename="scifact_generated_queries.tsv",
    )
    generator.generate()

    # # Uncomment for NFCorpus:
    # generator = BaselineGenerator(
    #     dataset_name="mteb/nfcorpus",
    #     subset="queries",
    #     split="queries",
    #     output_filename="nfcorpus_generated_queries.tsv",
    # )
    # generator.generate()


if __name__ == "__main__":
    main()
