"""
Generation of baseline queries using instruction tuned LLMs.

This module generates baseline queries to set a benchmark for the query generation task.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from searchlm.config import get_config
from searchlm.inference import VllmEngine
from searchlm.prompts import create_chat_prompt

config = get_config()


@dataclass
class SearchQuery:
    id: str
    text: str
    query: str | None = None


# Path configuration
output_root = Path(config.paths.output_dir)


def process_queries(engine: VllmEngine, queries: list[SearchQuery]) -> list[SearchQuery]:
    """Process a batch of queries through the LLM"""
    # Build prompts using the shared utility
    prompts = [create_chat_prompt(query.text, engine.tokenizer) for query in queries]

    # Generate responses
    start = time.time()
    responses = engine.generate(prompts, max_tokens=config.baseline.max_tokens)
    duration_s = time.time() - start

    print(f"Generated {len(responses)} responses in {int(duration_s)} seconds")

    # Attach responses to queries
    for response, query in zip(responses, queries):
        query.query = response

    return queries


def extract(dataset_name: str, subset: str, split: str) -> list[dict]:
    """Download dataset from HuggingFace and return raw data."""
    from datasets import load_dataset

    print(f"Loading dataset {dataset_name}, subset={subset}, split={split}")
    dataset = load_dataset(dataset_name, subset, split=split)

    # Convert to list of dicts
    data = []
    for item in dataset:
        data.append({"_id": item["_id"], "text": item["text"]})

    print(f"Loaded {len(data)} items from dataset")
    return data


def transform(raw_data: list[dict], batch_size: int = 100) -> list[list[SearchQuery]]:
    """Transform raw data into SearchQuery batches."""
    queries = [SearchQuery(id=item["_id"], text=item["text"]) for item in raw_data]

    # Split into batches for parallel processing
    batches = [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]

    print(f"Created {len(batches)} batches from {len(queries)} queries")
    return batches


def save_results(results: list[SearchQuery], output_filename: str) -> str:
    """Save results to a TSV file."""
    # Ensure output directory exists
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / output_filename

    print(f"Saving {len(results)} results to {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        # Write header
        writer.writerow(["id", "text", "query"])
        # Write data
        for result in results:
            writer.writerow([result.id, result.text, result.query])

    print(f"Results saved to {output_filename}")
    return str(output_path)


def orchestrate(
    dataset_name: str = "mteb/nfcorpus",
    subset: str = "queries",
    split: str = "queries",
    batch_size: int = 100,
    output_filename: str = "generated_queries.tsv",
) -> str:
    """
    Orchestrate the full pipeline: extract, transform, process, save.

    Returns:
        Path to the saved results file
    """
    # Extract data from HuggingFace
    print(f"Extracting data from {dataset_name}")
    raw_data = extract(dataset_name, subset, split)

    # Transform into batches
    print("Transforming data into batches")
    query_batches = transform(raw_data, batch_size=batch_size)

    n_queries = sum(len(batch) for batch in query_batches)
    print(f"Processing {n_queries} queries in {len(query_batches)} batches")

    # Process batches with LLM
    all_results = []
    with VllmEngine() as engine:
        for i, batch in enumerate(query_batches):
            print(f"Processing batch {i + 1}/{len(query_batches)}")
            results = process_queries(engine, batch)
            all_results.extend(results)

    # Save results
    print(f"\nSaving {len(all_results)} results...")
    output_path = save_results(all_results, output_filename)

    print(f"\n{'=' * 60}")
    print("✓ Processing complete!")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Total queries processed: {len(all_results)}")
    print(f"{'=' * 60}\n")

    return output_path
