import time
from pathlib import Path

import modal
from omegaconf import OmegaConf

CONTAINER_AUTO_STOP_TIME = 3600
# Container project root (add_local_dir puts repo at /root/searchlm)
CONTAINER_PROJECT_ROOT = "/root/searchlm"


def _get_output_mount_path() -> str:
    """Return container path for output volume from config paths.output_dir."""
    config_path = Path(__file__).parent / "config" / "default.yaml"
    conf = OmegaConf.load(config_path)
    output_dir = conf.paths.output_dir  # e.g. "./output"
    # Strip leading ./ or /
    output_name = output_dir.replace("./", "").strip("/") or "output"
    return f"{CONTAINER_PROJECT_ROOT}/{output_name}"


OUTPUT_MOUNT_PATH = _get_output_mount_path()

searchlm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_sync()
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(
        "~/searchlm",
        remote_path="/root",
        ignore=[".git", "__pycache__", ".venv", ".vscode", "data", "output"],
    )
)

volume = modal.Volume.from_name("searchlm-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(name="searchlm")


@app.function(
    image=searchlm_image,
    gpu="L4",
    timeout=3600,
    volumes={
        OUTPUT_MOUNT_PATH: volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def dev_shell():
    """
    Keeps a container running for interactive development.

    Usage:
    1. Terminal 1: modal run modal_infra.py::dev_shell
    2. Terminal 2: modal shell $(modal container list --json | jq -r '.[0]."Container ID"')
    """

    time.sleep(CONTAINER_AUTO_STOP_TIME)
    print("\nContainer shutting down, committing volume...")
    volume.commit()
    print("✅ Volume committed!")
