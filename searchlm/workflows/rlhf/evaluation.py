"""Unified evaluation for base and RLHF-trained models with multiple runs and aggregate metrics."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from searchlm import SearchEvaluator, create_loader
from searchlm.config import get_config, get_data_path
from searchlm.prompts import create_chat_prompt, extract_query_from_output
from searchlm.inference import VllmEngine


def get_latest_checkpoint() -> str:
    """Get path to latest checkpoint."""
    models_dir = get_data_path("models")

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


def save_evaluation_results(
    results: dict,
    model_name: str,
    model_path: str,
    run_number: int,
    output_dir: Path,
) -> Path:
    """
    Save evaluation results as JSON to volume mount.

    Args:
        results: Dictionary with evaluation results
        model_name: Name identifier for the model (e.g., 'base' or 'rlhf')
        model_path: Path to the model
        run_number: Run number for this evaluation
        output_dir: Directory to save results

    Returns:
        Path to saved JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_eval_run{run_number}_{timestamp}.json"
    output_path = output_dir / filename

    output_data = {
        "model_name": model_name,
        "model_path": model_path,
        "run_number": run_number,
        "timestamp": timestamp,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved evaluation results to: {output_path}")
    return output_path


def compute_aggregate_metrics(all_runs: list[dict]) -> dict:
    """
    Compute aggregate statistics across multiple evaluation runs.

    Args:
        all_runs: List of result dictionaries from multiple runs

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not all_runs:
        return {}

    # Get all dataset names
    datasets = list(all_runs[0].keys())
    aggregate = {}

    for dataset in datasets:
        metrics_names = [k for k in all_runs[0][dataset].keys() if isinstance(all_runs[0][dataset][k], (int, float))]
        aggregate[dataset] = {}

        for metric in metrics_names:
            values = [run[dataset][metric] for run in all_runs]
            aggregate[dataset][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    return aggregate


def print_aggregate_results(aggregate: dict, model_name: str):
    """Print aggregate statistics in a readable format."""
    print(f"\n{'=' * 70}")
    print(f"AGGREGATE RESULTS FOR {model_name.upper()}")
    print(f"{'=' * 70}")

    for dataset, metrics in aggregate.items():
        print(f"\n{dataset.upper()}:")
        print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 70)

        key_metrics = ['ndcg@10', 'ndcg@100', 'mrr', 'map', 'precision@10', 'recall@10']
        for metric in key_metrics:
            if metric in metrics:
                stats = metrics[metric]
                print(
                    f"{metric:<20} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
                    f"{stats['min']:<12.4f} {stats['max']:<12.4f}"
                )


def evaluate_single_run(
    model_path: str,
    model_name: str,
    run_number: int,
    output_dir: Path,
) -> dict:
    """
    Evaluate a model on all configured datasets (single run).

    Args:
        model_path: Path to model checkpoint or base model name
        model_name: Name identifier for the model
        run_number: Run number for this evaluation
        output_dir: Directory to save results

    Returns:
        Dictionary with evaluation results for all datasets
    """
    config = get_config()
    indices_dir = get_data_path("indices")

    print(f"\n{'=' * 70}")
    print(f"Run {run_number} - Evaluating {model_name.upper()}")
    print(f"Model: {model_path}")
    print(f"{'=' * 70}")

    # Initialize vLLM
    searchlm_engine = VllmEngine(model_name=model_path)

    # Initialize evaluator
    evaluator = SearchEvaluator(index_path=str(indices_dir))
    results = {}

    for dataset_name in config.evaluation.datasets:
        print(f"\n{'-' * 70}")
        print(f"Evaluating on {dataset_name} test split")
        print(f"{'-' * 70}")

        # Load test queries
        loader = create_loader(dataset_name)
        test_split = loader.load_split(split="test")

        # Format prompts and generate
        prompts = []
        query_ids = []
        for qid, query in test_split.queries.items():
            prompt = create_chat_prompt(query.text)
            prompts.append(prompt)
            query_ids.append(qid)

        print(f"Generating {len(prompts)} queries...")
        outputs = searchlm_engine.generate(prompts)

        # Extract queries
        generated_queries = [
            (extract_query_from_output(output), qid)
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

    # Save results to JSON
    save_evaluation_results(results, model_name, model_path, run_number, output_dir)

    return results


def evaluate_multiple_runs(
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    checkpoint_path: Optional[str] = None,
    num_runs: int = 3,
    evaluate_base: bool = True,
    evaluate_rlhf: bool = True,
) -> dict:
    """
    Run comprehensive evaluation with multiple runs for both base and RLHF models.

    Args:
        base_model_name: Name of the base model (e.g., 'Qwen/Qwen2.5-3B-Instruct')
        checkpoint_path: Path to RLHF checkpoint. If None, uses latest.
        num_runs: Number of evaluation runs per model
        evaluate_base: Whether to evaluate the base model
        evaluate_rlhf: Whether to evaluate the RLHF model

    Returns:
        Dictionary containing all results and aggregate metrics
    """
    # Setup output directory
    outputs_dir = get_data_path("outputs") / "evaluations"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print(f"Number of runs per model: {num_runs}")
    print(f"Output directory: {outputs_dir}")
    print("=" * 70)

    all_results = {
        "base": {"runs": [], "aggregate": {}},
        "rlhf": {"runs": [], "aggregate": {}},
    }

    # Evaluate base model
    if evaluate_base:
        print(f"\n{'#' * 70}")
        print("EVALUATING BASE MODEL")
        print(f"{'#' * 70}")

        for run in range(1, num_runs + 1):
            results = evaluate_single_run(
                model_path=base_model_name,
                model_name="base",
                run_number=run,
                output_dir=outputs_dir / "base",
            )
            all_results["base"]["runs"].append(results)

        # Compute aggregate metrics for base model
        all_results["base"]["aggregate"] = compute_aggregate_metrics(
            all_results["base"]["runs"]
        )
        print_aggregate_results(all_results["base"]["aggregate"], "base")

    # Evaluate RLHF model
    if evaluate_rlhf:
        if checkpoint_path is None:
            checkpoint_path = get_latest_checkpoint()

        print(f"\n{'#' * 70}")
        print("EVALUATING RLHF MODEL")
        print(f"{'#' * 70}")

        for run in range(1, num_runs + 1):
            results = evaluate_single_run(
                model_path=checkpoint_path,
                model_name="rlhf",
                run_number=run,
                output_dir=outputs_dir / "rlhf",
            )
            all_results["rlhf"]["runs"].append(results)

        # Compute aggregate metrics for RLHF model
        all_results["rlhf"]["aggregate"] = compute_aggregate_metrics(
            all_results["rlhf"]["runs"]
        )
        print_aggregate_results(all_results["rlhf"]["aggregate"], "rlhf")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = outputs_dir / f"evaluation_summary_{timestamp}.json"

    summary_data = {
        "base_model": base_model_name,
        "rlhf_checkpoint": checkpoint_path if evaluate_rlhf else None,
        "num_runs": num_runs,
        "timestamp": timestamp,
        "results": all_results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"Summary saved to: {summary_path}")
    print(f"{'=' * 70}\n")

    # Print comparison if both models evaluated
    if evaluate_base and evaluate_rlhf and all_results["base"]["runs"] and all_results["rlhf"]["runs"]:
        print_comparison(all_results)

    return all_results


def print_comparison(all_results: dict):
    """Print side-by-side comparison of base vs RLHF models."""
    print(f"\n{'=' * 70}")
    print("BASE vs RLHF COMPARISON")
    print(f"{'=' * 70}")

    base_agg = all_results["base"]["aggregate"]
    rlhf_agg = all_results["rlhf"]["aggregate"]

    for dataset in base_agg.keys():
        print(f"\n{dataset.upper()}:")
        print(f"{'Metric':<20} {'Base (mean)':<15} {'RLHF (mean)':<15} {'Improvement':<15}")
        print("-" * 70)

        key_metrics = ['ndcg@10', 'ndcg@100', 'mrr', 'map', 'precision@10', 'recall@10']
        for metric in key_metrics:
            if metric in base_agg[dataset] and metric in rlhf_agg[dataset]:
                base_val = base_agg[dataset][metric]["mean"]
                rlhf_val = rlhf_agg[dataset][metric]["mean"]
                improvement = ((rlhf_val - base_val) / base_val * 100) if base_val != 0 else 0
                improvement_str = f"{improvement:+.2f}%"
                print(
                    f"{metric:<20} {base_val:<15.4f} {rlhf_val:<15.4f} {improvement_str:<15}"
                )


# def evaluate(checkpoint_path: str = None, compare_baseline: bool = False):
#     """
#     Legacy single evaluation function for backwards compatibility.

#     Args:
#         checkpoint_path: Path to checkpoint. If None, uses latest.
#         compare_baseline: Ignored (kept for backwards compatibility).

#     Returns:
#         Dictionary with evaluation results.
#     """
#     config = get_config()
#     outputs_dir = get_data_path("outputs") / "evaluations"

#     if checkpoint_path is None:
#         checkpoint_path = get_latest_checkpoint()

#     results = evaluate_single_run(
#         model_path=checkpoint_path,
#         model_name="rlhf",
#         run_number=1,
#         output_dir=outputs_dir / "rlhf",
#     )

#     return results


if __name__ == "__main__":
    # Run comprehensive evaluation with multiple runs
    evaluate_multiple_runs(
        base_model_name="Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path=None,  # Uses latest checkpoint
        num_runs=3,
        evaluate_base=True,
        evaluate_rlhf=True,
    )