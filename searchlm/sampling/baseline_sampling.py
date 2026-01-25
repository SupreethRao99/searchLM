"""
Generation of baseline queries using instruction tuned LLMs

This aims to set a baseline for the query generation task.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import modal

SYSTEM_PROMPT = """
You are an expert at generating boolean search queries for a search engine.
You will be given a question in natural language and you need to generate a boolean search query for it.
The boolean search query will be used in conjunction with a search engine to retrieve relevant documents.
You should generate a query that is as specific as possible to the question, and that will return the most relevant documents.
You should use the following operators: AND, OR, NOT.

Below a few basic query formats are shown:

AND and OR conjunctions.
query = '(Old AND Man) OR Stream'

+(includes) and -(excludes) operators.
query = '+Old +Man chef -fished'

phrase search.
query = '"eighty-four days"'

Think step by step and generate the query.

The output format should be as follows:
<think>
your reasoning here
</think>
<query>
generated query here
</query>
"""

USER_PROMPT = """
Translate the following question into a boolean search query:
{question}
"""

app = modal.App(
    name="searchlm-baseline-sampling",
    tags={
        "system": "searchlm",
        "task": "baseline-sampling",
        "model": "Qwen2.5-3B-Instruct",
    },
)

GPU = "l4"


@dataclass
class SearchQuery:
    id: str
    text: str
    query: str | None = None


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install("vllm==0.13.0", "huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)
vllm_throughput_kwargs = {
    "max_model_len": 4096 * 4,  # based on data
}

# Image for data loading and processing
data_proc_image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "datasets==3.2.0"
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
output_vol = modal.Volume.from_name(
    "searchlm-output", create_if_missing=True, version=2
)
output_root = Path("/output")


@app.cls(
    image=vllm_image,
    gpu=GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=3600,  # 1 hour timeout for large batches
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Vllm:
    @modal.enter()
    def start(self):
        import vllm

        self.llm = vllm.LLM(model="Qwen/Qwen2.5-3B-Instruct", **vllm_throughput_kwargs)
        self.sampling_params = self.llm.get_default_sampling_params()
        self.sampling_params.max_tokens = 1000

        # Test the LLM
        self.llm.chat([{"role": "user", "content": "Is this thing on?"}])

    @modal.method()
    def process(self, queries: list[SearchQuery]) -> list[SearchQuery]:
        messages = []
        for query in queries:
            messages.append(
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(question=query.text),
                    },
                ]
            )

        start = time.time()
        responses = self.llm.chat(messages, sampling_params=self.sampling_params)
        duration_s = time.time() - start

        in_token_count = sum(len(response.prompt_token_ids) for response in responses)
        out_token_count = sum(
            len(response.outputs[0].token_ids) for response in responses
        )

        print(f"processed {in_token_count} prompt tokens in {int(duration_s)} seconds")
        print(f"generated {out_token_count} output tokens in {int(duration_s)} seconds")

        for response, query in zip(responses, queries):
            query.query = response.outputs[0].text

        return queries

    @modal.exit()
    def stop(self):
        del self.llm


@app.function(
    image=data_proc_image,
    scaledown_window=5,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
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


@app.function(scaledown_window=5)
def transform(raw_data: list[dict], batch_size: int = 100) -> list[list[SearchQuery]]:
    """Transform raw data into SearchQuery batches."""
    queries = [SearchQuery(id=item["_id"], text=item["text"]) for item in raw_data]

    # Split into batches for parallel processing
    batches = [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]

    print(f"Created {len(batches)} batches from {len(queries)} queries")
    return batches


@app.function(volumes={output_root: output_vol})
def save_results(results: list[SearchQuery], output_filename: str):
    """Save results to a TSV file on Modal Volume."""
    output_path = output_root / output_filename

    print(f"Saving {len(results)} results to {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        # Write header
        writer.writerow(["id", "text", "query"])
        # Write data
        for result in results:
            writer.writerow([result.id, result.text, result.query])

    output_vol.commit()
    print(f"Results saved to {output_filename}")
    return output_filename


@app.function(
    timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")]
)  # 1 hour timeout for orchestration
def orchestrate(
    dataset_name: str = "mteb/nfcorpus",
    subset: str = "queries",
    split: str = "queries",
    batch_size: int = 100,
) -> list[modal.FunctionCall]:
    """Orchestrate the full pipeline: extract, transform, process, save."""
    llm = Vllm()

    # Extract data from HuggingFace
    print(f"Extracting data from {dataset_name}")
    raw_data = extract.remote(dataset_name, subset, split)

    # Transform into batches
    print("Transforming data into batches")
    query_batches = transform.remote(raw_data, batch_size=batch_size)

    n_queries = sum(len(batch) for batch in query_batches)
    print(f"Submitting {n_queries} queries to LLM in {len(query_batches)} batches")

    # Process batches with LLM
    jobs = [llm.process.spawn(batch) for batch in query_batches]

    if jobs:
        print("FunctionCall IDs:", *[job.object_id for job in jobs], sep="\n\t")

    return jobs


@app.local_entrypoint()
def main(
    dataset_name: str = "mteb/nfcorpus",
    subset: str = "queries",
    split: str = "queries",
    batch_size: int = 4096,
    wait_for_results: bool = True,
    output_filename: str = "generated_queries.tsv",
):
    """
    Main entrypoint for baseline query generation.

    Args:
        dataset_name: HuggingFace dataset name (default: mteb/nfcorpus)
        subset: Dataset subset (default: queries)
        split: Dataset split (default: queries)
        batch_size: Number of queries per batch (default: 100)
        wait_for_results: Whether to wait for results or return immediately (default: True)
        output_filename: Name of the output TSV file (default: generated_queries.tsv)
    """
    # Trigger remote job orchestration
    jobs = orchestrate.remote(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        batch_size=batch_size,
    )

    if wait_for_results:
        print("Waiting for LLM processing to complete...")
        batches = modal.FunctionCall.gather(*jobs)

        # Flatten all batches into a single list
        all_results = []
        for batch in batches:
            all_results.extend(batch)

        print(f"Collected {len(all_results)} results")

        # Save results to TSV file
        print("Saving results to Modal Volume...")
        output_file = save_results.remote(all_results, output_filename)

        print(f"\n{'=' * 60}")
        print("✓ Processing complete!")
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ Total queries processed: {len(all_results)}")
        print("\nTo download the file, use:")
        print(f"  modal volume get searchlm-output {output_filename} {output_filename}")
        print(f"{'=' * 60}\n")
    else:
        print(
            "Job submitted. Collect results asynchronously with modal.FunctionCall.from_id"
        )
        print("FunctionCall IDs:", *[job.object_id for job in jobs], sep="\n\t")
