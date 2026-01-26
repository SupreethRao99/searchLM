"""
Evaluation script for GRPO-trained models.

This module evaluates trained checkpoints on test splits using vLLM for fast inference
and SearchEvaluator for computing IR metrics.
"""

import re
from pathlib import Path

from searchlm.config import get_config
from searchlm.prompts import SYSTEM_PROMPT

config = get_config()

# Directories
MODELS_DIR = Path(config.paths.models_dir)
INDEX_DIR = Path(config.paths.index_dir)


def get_latest_checkpoint():
    """Get path to latest checkpoint."""
    checkpoint_dirs = [
        d
        for d in MODELS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return str(MODELS_DIR / "final")

    # Get highest checkpoint number
    latest = max(checkpoint_dirs, key=lambda d: int(d.name.split("-")[1]))
    return str(latest)


def extract_query(text: str) -> str:
    """Extract query from model output."""
    match = re.search(r"<query>\s*(.*?)\s*</query>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def evaluate(checkpoint_path: str = None):
    """
    Evaluate a trained checkpoint on test splits.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses latest checkpoint.

    Returns:
        Dictionary with evaluation results for both datasets.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from searchlm import SearchEvaluator, create_loader

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
    evaluator = SearchEvaluator(index_path=str(INDEX_DIR))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    results = {}

    for dataset_name in config.evaluation.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name} test split")
        print(f"{'=' * 60}")

        # Load test queries
        loader = create_loader(dataset_name)
        test_split = loader.load_split(split="test")

        # Format prompts
        prompts = []
        query_ids = []
        for qid, query in test_split.queries.items():
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
            prompts.append(prompt)
            query_ids.append(qid)

        # Generate
        print(f"Generating {len(prompts)} queries...")
        outputs = llm.generate(prompts, sampling_params)

        # Extract queries
        generated_queries = []
        for output, qid in zip(outputs, query_ids):
            query_text = extract_query(output.outputs[0].text)
            generated_queries.append((query_text, qid))

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

    return results


def compare_with_baseline(checkpoint_path: str = None):
    """
    Compare trained model with baseline on test splits.

    This function evaluates both the trained model and generates a comparison
    report showing improvements over the baseline.

    Args:
        checkpoint_path: Path to checkpoint. If None, uses latest checkpoint.

    Returns:
        Dictionary with comparison results.
    """

    # First evaluate the trained model
    trained_results = evaluate(checkpoint_path)

    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)

    # Note: You would load baseline results from a saved file or run baseline evaluation
    # For now, we just return the trained model results
    print("\nTo compare with baseline:")
    print("1. Run scripts/base_evaluation.py to get baseline metrics")
    print("2. Compare the numbers above with baseline results")
    print("3. Look for improvements in NDCG@10 and MRR")

    return trained_results
