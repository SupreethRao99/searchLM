import time

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
