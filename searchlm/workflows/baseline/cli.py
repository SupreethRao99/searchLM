"""CLI entrypoint for baseline sampling"""

import argparse

from searchlm.config import get_config

config = get_config()


def main():
    """
    Main entrypoint for baseline query generation.

    Examples:
        # Generate queries for SciFact:
        python -m searchlm.workflows.baseline.cli

        # Generate queries for NFCorpus:
        python -m searchlm.workflows.baseline.cli \\
            --dataset-name mteb/nfcorpus \\
            --output-filename nfcorpus_generated_queries.tsv

        # Use custom batch size:
        python -m searchlm.workflows.baseline.cli --batch-size 50
    """
    parser = argparse.ArgumentParser(
        description="Generate baseline queries using instruction-tuned LLMs"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mteb/scifact",
        help="HuggingFace dataset name (default: mteb/scifact)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="queries",
        help="Dataset subset (default: queries)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="queries",
        help="Dataset split (default: queries)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Number of queries per batch (default: {config.baseline.batch_size})",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="scifact_generated_queries.tsv",
        help="Name of the output TSV file (default: scifact_generated_queries.tsv)",
    )

    args = parser.parse_args()

    from .sampling import orchestrate

    # Use config default if not specified
    batch_size = (
        args.batch_size if args.batch_size is not None else config.baseline.batch_size
    )

    # Run the orchestration
    output_path = orchestrate(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        batch_size=batch_size,
        output_filename=args.output_filename,
    )

    print(f"Complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
