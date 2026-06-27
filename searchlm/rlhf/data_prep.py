"""Data preparation for GRPO training.

v1: NFCorpus + SciFact train queries, no extra fields.
v2: Expanded datasets (+ FiQA + ArguAna), includes `nl_query` and a precomputed
    `keyword_baseline_ndcg` field so the shaped reward can measure improvement
    over naive keyword extraction without extra Tantivy queries at training time.
"""

import re

from datasets import Dataset

from searchlm import SearchEvaluator, create_loader
from searchlm.config import get_config, get_data_path
from searchlm.prompts import create_chat_prompt

# ── Keyword baseline helpers ───────────────────────────────────────────────────

_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "cannot",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "and",
    "or",
    "but",
    "not",
    "with",
    "as",
    "by",
    "from",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "each",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "then",
    "there",
    "they",
    "their",
    "them",
    "we",
    "our",
    "you",
    "your",
    "i",
    "my",
}


def extract_keyword_query(nl_query: str, max_words: int = 5) -> str:
    """Extract a naive keyword query: top-N content words from the NL query."""
    words = re.sub(r"[^\w\s]", " ", nl_query).split()
    content = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2]
    return " ".join(content[:max_words]) if content else nl_query.strip()


def compute_keyword_baselines(
    queries: dict,  # {query_id: Query}
    qrels: dict,  # {query_id: {doc_id: float}}
    evaluator: "SearchEvaluator",
    dataset_name: str,
    split: str = "train",
) -> dict:
    """Return {query_id: keyword_baseline_ndcg} for every query in the split."""
    baselines: dict[str, float] = {}
    for query_id, query in queries.items():
        kw_query = extract_keyword_query(query.text)
        query_qrels = qrels.get(query_id, {})
        if not kw_query or not query_qrels:
            baselines[query_id] = 0.0
            continue
        metrics, error = evaluator.evaluate_query(
            query_text=kw_query,
            qrels=query_qrels,
            k=100,
            dataset_filter=dataset_name,
        )
        baselines[query_id] = metrics["ndcg@10"] if not error else 0.0
    return baselines


# ── Core preparation ───────────────────────────────────────────────────────────


def prepare_training_data(version: str = "v1"):
    """
    Prepare GRPO training data.

    Args:
        version: "v1" — NFCorpus + SciFact, no extra columns (original behaviour).
                 "v2" — Expanded datasets, adds `nl_query` and
                        `keyword_baseline_ndcg` for shaped reward computation.
    """
    config = get_config()
    datasets_dir = get_data_path("datasets")

    # Determine dataset names and output path for this version
    if version == "v2":
        dataset_names = list(config.datasets_v2.names)
        out_dir = datasets_dir / "train_v2"
        test_out_dir = datasets_dir / "test_v2"
        use_keyword_baseline = True
    else:
        dataset_names = list(config.datasets.names)
        out_dir = datasets_dir / "train"
        test_out_dir = datasets_dir / "test"
        use_keyword_baseline = False

    print("=" * 60)
    print(f"Preparing GRPO training data [{version}]")
    print(f"Datasets: {dataset_names}")
    print(f"Output:   {out_dir}")
    print("=" * 60)

    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Set up evaluator for keyword baseline (v2 only)
    evaluator = None
    if use_keyword_baseline:
        indices_dir = get_data_path("indices")
        print(f"\nLoading Tantivy index from {indices_dir} (for keyword baselines)...")
        evaluator = SearchEvaluator(index_path=str(indices_dir))

    # Per-dataset query caps (v2 only) — prevents large datasets from dominating
    query_caps: dict[str, int] = {}
    if version == "v2":
        fiqa_cap = getattr(config.datasets_v2, "fiqa_max_train_queries", None)
        if fiqa_cap:
            query_caps["fiqa"] = int(fiqa_cap)

    all_data = []

    for dataset_name in dataset_names:
        print(f"\nLoading {dataset_name}...")
        loader = create_loader(dataset_name)
        dataset_split = loader.load_split(split="train")
        queries = dataset_split.queries
        qrels = dataset_split.qrels

        # Apply per-dataset cap if configured
        cap = query_caps.get(dataset_name)
        if cap and len(queries) > cap:
            import random

            rng = random.Random(42)
            sampled_ids = rng.sample(list(queries.keys()), cap)
            queries = {k: queries[k] for k in sampled_ids}
            print(f"  Sampled {cap}/{len(dataset_split.queries)} queries (capped)")

        print(f"  {len(queries)} queries, {len(qrels)} with qrels")

        # Precompute keyword baselines for v2
        baselines: dict[str, float] = {}
        if use_keyword_baseline and evaluator is not None:
            print(f"  Computing keyword baselines for {dataset_name}...")
            baselines = compute_keyword_baselines(
                queries, qrels, evaluator, dataset_name, split="train"
            )
            non_zero = sum(1 for v in baselines.values() if v > 0)
            mean_baseline = (
                sum(baselines.values()) / len(baselines) if baselines else 0.0
            )
            print(
                f"  Baseline: {non_zero}/{len(baselines)} non-zero, mean={mean_baseline:.3f}"
            )

        for query_id, query in queries.items():
            row = {
                "prompt": create_chat_prompt(query.text),
                "query_id": query_id,
                "dataset_name": dataset_name,
            }
            if use_keyword_baseline:
                row["nl_query"] = query.text
                row["keyword_baseline_ndcg"] = baselines.get(query_id, 0.0)
            all_data.append(row)

    print(f"\nTotal: {len(all_data)} examples")

    dataset = Dataset.from_list(all_data)
    dataset = dataset.shuffle(seed=42)

    split_idx = int(len(dataset) * 0.9)
    train_dataset = dataset.select(range(split_idx))
    test_dataset = dataset.select(range(split_idx, len(dataset)))

    print(f"Saving to {out_dir}...")
    train_dataset.save_to_disk(str(out_dir))
    test_dataset.save_to_disk(str(test_out_dir))

    print(f"✓ {len(train_dataset)} train  |  {len(test_dataset)} test")
    train_df = train_dataset.to_pandas()
    print("\nDataset breakdown:")
    print(train_df["dataset_name"].value_counts().to_string())


if __name__ == "__main__":
    import sys

    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    prepare_training_data(version=ver)
