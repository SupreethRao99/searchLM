"""CLI entrypoint for RLHF workflow"""

import argparse


def main():
    """
    RLHF workflow CLI

    Examples:
        # Prepare training data:
        python -m searchlm.workflows.rlhf.cli prep

        # Train with colocate mode (1 GPU):
        python -m searchlm.workflows.rlhf.cli train

        # Train with server mode (2 GPUs):
        python -m searchlm.workflows.rlhf.cli train --use-vllm-server

        # Evaluate latest checkpoint:
        python -m searchlm.workflows.rlhf.cli eval

        # Evaluate specific checkpoint:
        python -m searchlm.workflows.rlhf.cli eval \\
            --checkpoint-path ./models/checkpoint-500

        # Compare with baseline:
        python -m searchlm.workflows.rlhf.cli eval --compare-baseline
    """
    parser = argparse.ArgumentParser(
        description="RLHF training workflow for SearchLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to run", required=True
    )

    # Prep command
    subparsers.add_parser("prep", help="Prepare training data")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--use-vllm-server",
        action="store_true",
        help="Use vLLM server mode for training (requires 2 GPUs)",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained model")
    eval_parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint (default: latest checkpoint)",
    )
    eval_parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline results",
    )

    args = parser.parse_args()

    if args.command == "prep":
        print("=" * 60)
        print("Preparing training data")
        print("=" * 60)
        from .data_prep import prep_dataset

        prep_dataset()

    elif args.command == "train":
        from .training import train, train_with_vllm_server

        if args.use_vllm_server:
            print("=" * 60)
            print("Training with vLLM server mode (2 GPUs)")
            print("=" * 60)
            train_with_vllm_server()
        else:
            print("=" * 60)
            print("Training with vLLM colocate mode (1 GPU)")
            print("=" * 60)
            train()

    elif args.command == "eval":
        print("\n" + "=" * 60)
        print("SearchLM GRPO Model Evaluation")
        print("=" * 60)

        from .evaluation import compare_with_baseline as compare_func
        from .evaluation import evaluate

        if args.compare_baseline:
            results = compare_func(args.checkpoint_path)
        else:
            results = evaluate(args.checkpoint_path)

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

        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
