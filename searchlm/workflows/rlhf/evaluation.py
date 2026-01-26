"""Evaluation for GRPO-trained models."""

from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from searchlm import SearchEvaluator, create_loader
from searchlm.config import get_config
from searchlm.prompts import create_chat_prompt, extract_query_from_output


def get_latest_checkpoint() -> str:
    """Get path to latest checkpoint."""
    config = get_config()
    models_dir = Path(config.paths.models_dir)

    checkpoint_dirs = [
        d
        for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return str(models_dir / "final")

    # Get highest checkpoint number
    latest = max(checkpoint_dirs, key=lambda d: int(d.name.split("-")[1]))
    return str(latest)


def evaluate(checkpoint_path: str = None, compare_baseline: bool = False):
    """
    Evaluate a trained checkpoint on test splits.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses latest.
        compare_baseline: If True, show baseline comparison message.

    Returns:
        Dictionary with evaluation results for both datasets.
    """
    config = get_config()
    index_dir = Path(config.paths.index_dir)

    print("\n" + "=" * 60)
    print("SearchLM GRPO Model Evaluation")
    print("=" * 60)

    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Initialize vLLM
    llm = LLM(model=checkpoint_path)
    sampling_params = SamplingParams(
        temperature=config.evaluation.temperature,
        max_tokens=config.evaluation.max_tokens,
    )

    # Initialize evaluator
    evaluator = SearchEvaluator(index_path=str(index_dir))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    results = {}

    for dataset_name in config.evaluation.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name} test split")
        print(f"{'=' * 60}")

        # Load test queries
        loader = create_loader(dataset_name)
        test_split = loader.load_split(split="test")

        # Format prompts and generate
        prompts = []
        query_ids = []
        for qid, query in test_split.queries.items():
            prompt = create_chat_prompt(query.text, tokenizer)
            prompts.append(prompt)
            query_ids.append(qid)

        print(f"Generating {len(prompts)} queries...")
        outputs = llm.generate(prompts, sampling_params)

        # Extract queries
        generated_queries = [
            (extract_query_from_output(output.outputs[0].text), qid)
            for output, qid in zip(outputs, query_ids)
        ]

        # Batch evaluation
        print(f"Evaluating {len(generated_queries)} queries...")
        metrics = evaluator.evaluate_batch(
            queries=generated_queries,
            dataset_name=dataset_name,
            split="test",
            k=config.evaluation.default_k,
        )

        evaluator.print_metrics(metrics)
        results[dataset_name] = metrics

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  NDCG@10:  {metrics['ndcg@10']:.4f}")
        print(f"  NDCG@100: {metrics['ndcg@100']:.4f}")
        print(f"  MRR:      {metrics['mrr']:.4f}")
        print(f"  MAP:      {metrics['map']:.4f}")
        print(f"  Precision@10: {metrics['precision@10']:.4f}")
        print(f"  Recall@10:    {metrics['recall@10']:.4f}")
        total = metrics["num_queries"] + metrics["num_failed"]
        print(f"  Failed: {metrics['num_failed']}/{total}")

    if compare_baseline:
        print("\n" + "=" * 60)
        print("COMPARISON WITH BASELINE")
        print("=" * 60)
        print("\nTo compare with baseline:")
        print("1. Run baseline generation to get baseline metrics")
        print("2. Compare the numbers above with baseline results")
        print("3. Look for improvements in NDCG@10 and MRR")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60 + "\n")

    return results
