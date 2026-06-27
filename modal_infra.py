import time
from pathlib import Path

import modal

CONTAINER_AUTO_STOP_TIME = 3600
# Container project root (add_local_dir puts repo at /root/searchlm)
CONTAINER_PROJECT_ROOT = "/root/searchlm"

# Volume configuration - matches config paths.data_dir
DATA_DIR = "modal_data"
CONTAINER_DATA_PATH = f"{CONTAINER_PROJECT_ROOT}/{DATA_DIR}"

searchlm_image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_sync()
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(
        "~/searchlm",
        remote_path="/root",
        ignore=[
            ".git",
            "__pycache__",
            ".venv",
            ".vscode",
            "modal_data/",
            "data/",
            "output/",
            "*.pyc",
        ],
    )
)

volume = modal.Volume.from_name("searchlm", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(name="searchlm")


@app.function(
    image=searchlm_image,
    gpu="L4",
    timeout=CONTAINER_AUTO_STOP_TIME,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def dev_shell():
    """
    Keeps a container running for interactive development.

    Usage:
    1. Terminal 1: modal run modal_infra.py::dev_shell
    2. Terminal 2: modal shell $(modal container list --json | \\
                   jq -r '.[0]."Container ID"')
    
    IMPORTANT: Press Ctrl+C in Terminal 1 to exit and commit volume changes!
    """
    try:
        print(f"🚀 Container running for {CONTAINER_AUTO_STOP_TIME}s")
        print("📁 Volume mounted at:", CONTAINER_DATA_PATH)
        print("\n💡 Tip: Run workflows, then press Ctrl+C here to commit changes")
        time.sleep(CONTAINER_AUTO_STOP_TIME)
    finally:
        # Always commit volume, even on Ctrl+C or errors
        print("\n💾 Committing volume...")
        volume.commit()
        print("✅ Volume committed! Files persisted to 'searchlm' volume")
        print("   Verify with: modal volume ls searchlm")


SFT_TIMEOUT = 2 * 3600  # 2 hours — 5K examples × 1 epoch LoRA on A10G is ~30-40 min


@app.function(
    image=searchlm_image,
    gpu="A10G",
    timeout=SFT_TIMEOUT,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_sft(dataset_source: str = "hub"):
    """
    SFT warm-start: fine-tune Qwen2.5-3B-Instruct on nl2bm25-sft dataset.

    Saves checkpoint to modal_data/models/sft/final.
    Run this before run_grpo — GRPO will start from that checkpoint.

    Usage:
        modal run modal_infra.py::run_sft
        modal run modal_infra.py::run_sft --dataset-source local
    """
    import os
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    os.environ.setdefault("WANDB_PROJECT", "searchlm")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from searchlm.rlhf.sft import train
    try:
        train(dataset_source=dataset_source)
    finally:
        volume.commit()


@app.function(
    image=searchlm_image,
    gpu=None,
    timeout=1800,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_sft_to_hub():
    """Push the merged SFT checkpoint to HuggingFace Hub."""
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    from searchlm.config import get_data_path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    REPO_ID = "Supreeth/searchlm-nl2bm25-sft"

    models_dir = get_data_path("models")
    sft_final = models_dir / "sft" / "final"
    print(f"Loading from {sft_final}...")

    tokenizer = AutoTokenizer.from_pretrained(str(sft_final))
    model = AutoModelForCausalLM.from_pretrained(str(sft_final), torch_dtype="auto")

    print(f"Pushing model to {REPO_ID}...")
    model.push_to_hub(REPO_ID, private=False)
    tokenizer.push_to_hub(REPO_ID, private=False)
    print(f"Done: https://huggingface.co/{REPO_ID}")


SFT_V2_TIMEOUT   = 2 * 3600   # 2 hours — filtered dataset is ~3K examples
ANALYSIS_TIMEOUT = 4 * 3600   # 4 hours (3 models × 2 datasets × ~300 queries)
GRPO_V2_TIMEOUT  = 14 * 3600  # 14 hours — 4 datasets, num_generations=8
GRPO_TIMEOUT     = 12 * 3600  # 12 hours


@app.function(
    image=searchlm_image,
    gpu="A10G",
    timeout=SFT_V2_TIMEOUT,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_sft_v2(dataset_source: str = "hub"):
    """
    SFT v2: quality-filtered fine-tuning on nl2bm25-sft (ndcg_at_10 > 0 only).

    Saves to modal_data/models/sft_v2/{adapter,final}.
    Run before run_grpo_v2 — GRPO v2 will start from this checkpoint.

    Usage:
        modal run modal_infra.py::run_sft_v2
    """
    import os
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    os.environ.setdefault("WANDB_PROJECT", "searchlm")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from searchlm.rlhf.sft import train
    try:
        train(dataset_source=dataset_source, version="v2")
    finally:
        volume.commit()


@app.function(
    image=searchlm_image,
    gpu="H100",
    timeout=GRPO_V2_TIMEOUT,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_grpo_v2(use_vllm_server: bool = False):
    """
    GRPO v2: shaped reward, expanded datasets (+ FiQA + ArguAna), num_generations=8.

    Improvements over v1:
    - Reward = delta over keyword baseline + reasoning bonus + complexity multiplier
    - Expanded datasets: NFCorpus + SciFact + FiQA + ArguAna
    - num_generations=8 for more within-group diversity
    - Saves to modal_data/models/grpo_v2/

    Usage:
        modal run modal_infra.py::run_grpo_v2
    """
    import os
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    os.environ.setdefault("WANDB_PROJECT", "searchlm")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    indices_dir  = Path(CONTAINER_DATA_PATH) / "indices"
    datasets_dir = Path(CONTAINER_DATA_PATH) / "datasets"

    # ── Step 1: Build/verify Tantivy index (all 4 v2 datasets) ───────────────
    # FiQA and ArguAna need to be in the index for reward computation
    v2_datasets = ["nfcorpus", "scifact", "fiqa", "arguana"]
    if not indices_dir.exists() or not any(indices_dir.iterdir()):
        print("Building search index for v2 datasets...")
        from searchlm.data.ingesters.pipeline import ingest_all_datasets
        ingest_all_datasets(index_path=str(indices_dir), datasets=v2_datasets)
        volume.commit()
    else:
        print(f"Search index found at {indices_dir}")
        # Check if FiQA/ArguAna are in the index; rebuild if the index predates v2
        # (A simple heuristic: trust the user ran the right ingest for now)

    # ── Step 2: Prepare v2 training dataset ───────────────────────────────────
    train_v2_dir = datasets_dir / "train_v2"
    if not train_v2_dir.exists():
        print("Preparing v2 training dataset (with keyword baselines)...")
        from searchlm.rlhf.data_prep import prepare_training_data
        prepare_training_data(version="v2")
        volume.commit()
    else:
        print(f"v2 training dataset found at {train_v2_dir}")

    # ── Step 3: GRPO v2 training ──────────────────────────────────────────────
    from searchlm.rlhf.training import train
    try:
        train(use_vllm_server=use_vllm_server, version="v2")
    finally:
        volume.commit()


@app.function(
    image=searchlm_image,
    gpu="A10G",
    timeout=ANALYSIS_TIMEOUT,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_analysis_v2(analysis_version: str = "v2"):
    """
    Run reward hacking analysis for v2 (or cross-version comparison).

    Args:
        analysis_version: "v2"      — evaluate base / sft_v2 / grpo_v2
                          "compare" — evaluate all 5 checkpoints side-by-side

    Usage:
        modal run modal_infra.py::run_analysis_v2
        modal run modal_infra.py::run_analysis_v2 --analysis-version compare
    """
    import os
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    indices_dir = Path(CONTAINER_DATA_PATH) / "indices"
    if not indices_dir.exists() or not any(indices_dir.iterdir()):
        print("Building search index...")
        from searchlm.data.ingesters.pipeline import ingest_all_datasets
        ingest_all_datasets(index_path=str(indices_dir), datasets=["nfcorpus", "scifact"])
        volume.commit()

    from scripts.analyze_reward_hacking import run_analysis
    try:
        run_analysis(version=analysis_version)
    finally:
        volume.commit()


@app.function(
    image=searchlm_image,
    gpu="A10G",
    timeout=ANALYSIS_TIMEOUT,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_reward_hacking_analysis():
    """
    Evaluate base / SFT / GRPO models and produce a reward hacking report.

    Runs on NFCorpus + SciFact test splits; saves JSON + Markdown report to
    the Modal volume at modal_data/outputs/reward_hacking/.

    Usage:
        modal run modal_infra.py::run_reward_hacking_analysis
    """
    import os
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    indices_dir = Path(CONTAINER_DATA_PATH) / "indices"

    # Build index if not present (should already exist from GRPO run)
    if not indices_dir.exists() or not any(indices_dir.iterdir()):
        print("Search index not found — ingesting NFCorpus + SciFact …")
        from searchlm.data.ingesters.pipeline import ingest_all_datasets
        ingest_all_datasets(
            index_path=str(indices_dir),
            datasets=["nfcorpus", "scifact"],
        )
        volume.commit()

    from scripts.analyze_reward_hacking import run_analysis
    try:
        run_analysis()
    finally:
        volume.commit()


@app.function(
    image=searchlm_image,
    gpu=None,
    timeout=600,
    volumes={CONTAINER_DATA_PATH: volume},
)
def regenerate_report():
    """
    Load the most recent aggregate_metrics + qualitative_examples JSONs
    and re-run report generation without re-running inference.

    Usage:
        modal run modal_infra.py::regenerate_report
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    rh_dir = Path(CONTAINER_DATA_PATH) / "outputs" / "reward_hacking"
    agg_files  = sorted(rh_dir.glob("aggregate_metrics_*.json"),   reverse=True)
    qual_files = sorted(rh_dir.glob("qualitative_examples_*.json"), reverse=True)

    if not agg_files or not qual_files:
        print("Missing data files. Run run_reward_hacking_analysis first.")
        return

    print(f"Loading: {agg_files[0].name}")
    print(f"Loading: {qual_files[0].name}")
    with open(agg_files[0]) as f:
        agg_raw = json.load(f)
    with open(qual_files[0]) as f:
        qual_out = json.load(f)

    # Deserialise string keys back to (model, dataset) tuples
    aggregate = {tuple(k.split("|")): v for k, v in agg_raw.items()}

    from scripts.analyze_reward_hacking import generate_report
    report_text = generate_report(aggregate, qual_out, rh_dir)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = rh_dir / f"report_{ts}.md"
    report_path.write_text(report_text)
    print(f"Report saved → {report_path}")
    volume.commit()

    print("\n" + "=" * 70)
    print(report_text)


@app.function(
    image=searchlm_image,
    gpu=None,
    timeout=600,
    volumes={CONTAINER_DATA_PATH: volume},
)
def fetch_analysis_outputs():
    """
    Print the latest reward hacking report to stdout so you can redirect it locally.

    Usage:
        modal run modal_infra.py::fetch_analysis_outputs
    """
    from pathlib import Path

    rh_dir = Path(CONTAINER_DATA_PATH) / "outputs" / "reward_hacking"
    if not rh_dir.exists():
        print("No reward_hacking outputs found yet. Run run_reward_hacking_analysis first.")
        return

    reports = sorted(rh_dir.glob("report_*.md"), reverse=True)
    if not reports:
        print("No report files found.")
        return

    latest = reports[0]
    print(f"=== {latest.name} ===\n")
    print(latest.read_text())

    # Also list all output files
    print("\n\n=== All output files ===")
    for f in sorted(rh_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


PUSH_TIMEOUT        = 4 * 3600   # 4 hours — loading + pushing 4 × 3B models sequentially
CARD_UPDATE_TIMEOUT = 10 * 60    # 10 minutes — just HTTP uploads, no model loading


@app.function(
    image=searchlm_image,
    gpu=None,
    timeout=PUSH_TIMEOUT,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=32768,  # 32 GB RAM — enough to load a 3B bf16 model on CPU
)
def push_all_models_to_hub(models: str = "all", no_collection: bool = False):
    """
    Push all SearchLM checkpoints to HuggingFace Hub and create the SearchLM collection.

    Pushes (in order): sft → sft_v2 → grpo → grpo_v2.
    Skips any checkpoint whose local path doesn't exist.
    Creates/updates the SearchLM HuggingFace collection with all pushed models.

    Args:
        models: comma-separated list of models to push, or "all"
                choices: sft, sft_v2, grpo, grpo_v2
        no_collection: if True, skip collection creation

    Usage:
        modal run modal_infra.py::push_all_models_to_hub
        modal run modal_infra.py::push_all_models_to_hub --models sft_v2,grpo_v2
        modal run modal_infra.py::push_all_models_to_hub --no-collection
    """
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    from huggingface_hub import HfApi
    from scripts.push_models import (
        REPOS,
        push_model,
        create_searchlm_collection,
    )

    api = HfApi()
    names = list(REPOS.keys()) if models == "all" else [m.strip() for m in models.split(",")]
    pushed = []

    for name in names:
        repo_id = push_model(name, api, dry_run=False)
        pushed.append(repo_id)

    if not no_collection:
        create_searchlm_collection(pushed, api)

    print("\nAll done.")


@app.function(
    image=searchlm_image,
    gpu=None,
    timeout=CARD_UPDATE_TIMEOUT,
    volumes={CONTAINER_DATA_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def upload_cards_to_hub(models: str = "all"):
    """
    Re-upload enriched README.md model cards to already-published Hub repos.

    Does NOT reload or re-push model weights — pure HTTP uploads only (~seconds per repo).
    Run this after push_all_models_to_hub to update cards without re-pushing 3B models.

    Args:
        models: comma-separated list or "all" (sft, sft_v2, grpo, grpo_v2)

    Usage:
        modal run modal_infra.py::upload_cards_to_hub
        modal run modal_infra.py::upload_cards_to_hub --models grpo_v2
    """
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    from scripts.push_models import REPOS, upload_cards_only

    names = list(REPOS.keys()) if models == "all" else [m.strip() for m in models.split(",")]
    upload_cards_only(names)
    print("\nAll done.")


@app.function(
    image=searchlm_image,
    gpu="H100",
    timeout=GRPO_TIMEOUT,
    volumes={
        CONTAINER_DATA_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_grpo(use_vllm_server: bool = False):
    """
    GRPO training starting from the SFT checkpoint.

    Automatically ingests the search index and prepares the training dataset
    on first run — subsequent runs reuse what's already on the volume.

    Usage:
        modal run modal_infra.py::run_grpo
    """
    import os
    import sys
    sys.path.insert(0, CONTAINER_PROJECT_ROOT)

    os.environ.setdefault("WANDB_PROJECT", "searchlm")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    indices_dir = Path(CONTAINER_DATA_PATH) / "indices"
    datasets_dir = Path(CONTAINER_DATA_PATH) / "datasets"

    # ── Step 1: Build Tantivy index if not already on volume ─────────────────
    if not indices_dir.exists() or not any(indices_dir.iterdir()):
        print("Search index not found — ingesting datasets...")
        from searchlm.data.ingesters.pipeline import ingest_all_datasets
        ingest_all_datasets(
            index_path=str(indices_dir),
            datasets=["nfcorpus", "scifact"],  # GRPO reward datasets
        )
        volume.commit()
        print("Index committed to volume.")
    else:
        print(f"Search index found at {indices_dir}")

    # ── Step 2: Prepare GRPO training dataset if not already on volume ────────
    if not (datasets_dir / "train").exists():
        print("Training dataset not found — preparing...")
        from searchlm.rlhf.data_prep import prepare_training_data
        prepare_training_data()
        volume.commit()
        print("Training dataset committed to volume.")
    else:
        print(f"Training dataset found at {datasets_dir / 'train'}")

    # ── Step 3: GRPO training ─────────────────────────────────────────────────
    from searchlm.rlhf.training import train
    try:
        train(use_vllm_server=use_vllm_server)
    finally:
        volume.commit()
