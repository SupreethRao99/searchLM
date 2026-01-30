# Modal Interactive Shell Guide

## Overview

This guide shows you how to use Modal like a traditional VM where you SSH in, run code, see output, debug, and iterate - all with GPU access and persistent data storage through volumes.

## Quick Start

Launch an interactive shell in your Modal container using two terminals:

**Terminal 1: Start the container**
```bash
modal run modal_infra.py::dev_shell
```

**Terminal 2: Attach to the container**
```bash
# Get the container ID
modal container list

# Attach to it (replace with your container ID)
modal shell ta-XXXXXXXXXXXXXXXXXXXXX
```

This drops you into a bash shell inside a container with:
- ✅ Full GPU access (L4)
- ✅ Your code mounted at `/root/searchlm`
- ✅ Persistent volume at `output/` (from `config/default.yaml` paths.output_dir)
- ✅ All your dependencies installed
- ✅ CUDA 12.8 environment ready

## Basic Workflow

### 1. Start the Dev Container

Open Terminal 1 and run:
```bash
modal run modal_infra.py::dev_shell
```

You'll see:
```
✓ Created objects.
✓ App initialized. View run at https://modal.com/apps/ap-XXXXX
🚀 Dev container is READY!

In another terminal, run:
  1. modal container list
  2. modal shell ta-XXXXXXXXXXXXXXXXXXXXX

Container will stay alive for 1 hour...
```

### 2. Attach to the Container

Open Terminal 2 and get the container ID:
```bash
modal container list
```

You'll see something like:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Container ID                  ┃ App ID ┃ App Name ┃ Start Time ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ta-01JK47GVDMWMGPH8MQ0EW30Y25 │ ap-... │ searchlm │ 16:02 EST  │
└───────────────────────────────┴────────┴──────────┴────────────┘
```

Now attach to it:
```bash
modal shell ta-01JK47GVDMWMGPH8MQ0EW30Y25  # Use your container ID
```

You'll get an interactive bash prompt:
```
root@modal-container:~# 
```

### 3. Navigate and Run Your Code

```bash
# Change to project root (output volume is mounted here)
cd /root/searchlm

# Run your script with arguments
python searchlm/workflows/baseline/baseline.py --arg1 value1 --arg2 value2

# Or run it directly from anywhere
python /root/searchlm/searchlm/workflows/baseline/baseline.py
```

### 4. Debug and Iterate

```bash
# View your code
cat searchlm/workflows/baseline/baseline.py

# Edit with vim (pre-installed)
vim searchlm/workflows/baseline/baseline.py

# Or use nano (also pre-installed)
nano searchlm/workflows/baseline/baseline.py

# Run again after making changes
python searchlm/workflows/baseline/baseline.py
```

### 5. Work with Persistent Data

```bash
# Volume is mounted at output/ (paths.output_dir in config/default.yaml)
ls output

# Save data that persists across sessions
echo "experiment_results" > output/results.txt

# Run baseline (writes to output/ via config)
python -m searchlm.workflows.baseline.baseline

# Check what's stored
ls -lh output
```

### 6. Exit When Done

In Terminal 2 (the shell), type:
```bash
exit
```

In Terminal 1 (the dev container), press `Ctrl+C` to stop the container.

**Note:** The volume automatically commits when the container shuts down, so your data in `output/` persists!

## Common Tasks

### Check GPU Availability

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Install Additional Packages

**Runtime installation (temporary):**
```bash
# Install Python packages
pip install some-package

# Install system packages
apt-get update && apt-get install -y some-tool
```

**Permanent installation** (edit `modal_infra.py`):
```python
searchlm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_sync()
    .pip_install("ipython", "some-package")  # Add here
    .apt_install("some-tool")  # Or here for system packages
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(".", remote_path="/root/searchlm")
)
```

### Monitor Process

```bash
# Watch GPU usage while code runs
watch -n 1 nvidia-smi

# Or in another terminal, get the container ID and attach
# Terminal 1: modal container list
# Terminal 2: modal container exec ta-XXXXX nvidia-smi
```

### Run Background Jobs

```bash
# Run in background
nohup python -m searchlm.workflows.baseline.baseline > output/output.log 2>&1 &

# Check if still running
ps aux | grep python

# View live logs
tail -f output/output.log
```

## Advanced Usage

### Multiple Terminal Sessions

You can attach multiple shells to the same running container:

**Terminal 1:**
```bash
# Start container
modal run modal_infra.py::dev_shell
# Keep this running...
```

**Terminal 2:**
```bash
# Get the container ID and attach
modal container list
modal shell ta-XXXXXXXXXXXXXXXXXXXXX

# Start long-running script
python long_running_script.py
```

**Terminal 3:**
```bash
# Attach another shell to the SAME container
modal shell ta-XXXXXXXXXXXXXXXXXXXXX  # Same container ID!

# Now you can monitor, debug, or run other commands
nvidia-smi
tail -f output/logs/training.log
```

### Execute Single Commands

Run a command without entering interactive mode:

```bash
# List files
modal container exec ta-XXXXX ls /root/searchlm

# Check GPU
modal container exec ta-XXXXX nvidia-smi

# Run a quick script
modal container exec ta-XXXXX python /root/searchlm/test.py
```

### Debug with Python Debugger

Add breakpoints in your code:

```python
def my_function():
    x = compute_something()
    breakpoint()  # Execution pauses here
    result = x * 2
    return result
```

Run with interactive mode:
```bash
modal run -i modal_infra.py::launcher --cmd-args "your args"
```

## Pre-installed Tools

Your debug shell comes with:
- `vim` - Text editor
- `nano` - Simpler text editor
- `ps` - Process viewer
- `strace` - System call tracer
- `curl` - HTTP client
- `py-spy` - Python profiler
- Standard Unix tools (grep, find, etc.)

## File Locations

- **Your code:** `/root/searchlm/` (synced from local directory)
- **Persistent output:** `output/` (Modal volume; path from `config/default.yaml` paths.output_dir)
- **Working directory:** `/root/searchlm` (cd here to run workflows)
- **Python packages:** Managed by uv in the image

## Tips & Best Practices

### 1. **Data Persistence**
- Save outputs to `output/` (config paths.output_dir; volume-mounted)
- The synced code is read-only; the volume at `output/` persists
- Volume automatically commits on container exit
- Check volume contents: `modal volume ls searchlm-data`

### 2. **Code Changes**
- For quick edits, use `vim` or `nano` inside the container
- For larger changes, edit locally and restart the shell
- Local changes sync when you launch a new shell

### 3. **Long-Running Jobs**
- Use `nohup` and background jobs for long processes
- Save logs to `output/` so they persist
- Monitor with `ps aux` and `nvidia-smi`

### 4. **Resource Management**
- Shell terminates when container stops (after timeout)
- Default timeout is 3600 seconds (1 hour)
- Increase in `modal_infra.py` if needed: `timeout=7200`

### 5. **Debugging Stuck Processes**
```bash
# Find the process
ps aux | grep python

# Kill if needed
kill -9 <PID>

# Check what's using GPU
nvidia-smi
```

## Troubleshooting

### Why Two Terminals?

Modal requires a container to be running before you can attach a shell to it. The two-terminal approach:
1. **Terminal 1** keeps a container alive with the right GPU, volumes, and environment
2. **Terminal 2** attaches an interactive shell to that running container

This is the most reliable way to get VM-like SSH access to Modal.

### Error: `.git/FETCH_HEAD was modified during build process`

This happens when Modal syncs your `.git` directory and detects modifications. 

**Fix:** Create a `.modalignore` file in your project root:

```bash
# .modalignore
.git/
__pycache__/
.venv/
.vscode/
```

The `.modalignore` file in this repo already excludes `.git/`, so you should be good to go!

### Container Exits Immediately

If the shell closes right away, increase timeout:

```python
@app.function(
    timeout=7200,  # 2 hours
    ...
)
```

### Code Changes Not Reflected

The code is synced when the container starts. To see new changes:
1. Exit the shell in Terminal 2 (`exit`)
2. Stop the container in Terminal 1 (`Ctrl+C`)
3. Restart: `modal run modal_infra.py::dev_shell`
4. Attach again in Terminal 2

### Volume Data Not Persisting

Volumes auto-commit on exit, but you can manually commit:

```bash
# Inside shell, run Python
python -c "import modal; modal.Volume.from_name('searchlm-data').commit()"
```

### Permission Denied Errors

You're running as root, but if you get permission errors:

```bash
chmod +x /root/searchlm/your_script.py
```

## Workflow Examples

### Example 1: Train a Model

**Terminal 1:**
```bash
# Start container
modal run modal_infra.py::dev_shell
# Keep this running...
```

**Terminal 2:**
```bash
# Get container ID and attach
modal container list
modal shell ta-XXXXX

# Navigate to code
cd /root/searchlm

# Run training
python searchlm/workflows/baseline/baseline.py \
    --model-name "my-model" \
    --epochs 10 \
    --output-dir output/checkpoints

# Check results
ls -lh output/checkpoints

# Exit shell
exit

# Back in Terminal 1, press Ctrl+C to stop container
```

### Example 2: Debug a Failed Run

**Terminal 1:**
```bash
modal run modal_infra.py::dev_shell
```

**Terminal 2:**
```bash
# Attach to container
modal container list
modal shell ta-XXXXX

# Try running the script
cd /root/searchlm
python -m searchlm.workflows.baseline.baseline

# If it fails, edit and fix
vim searchlm/workflows/baseline/baseline.py

# Run again
python searchlm/workflows/baseline/baseline.py

# Success! Exit
exit
```

### Example 3: Explore Data

**Terminal 1:**
```bash
modal run modal_infra.py::dev_shell
```

**Terminal 2:**
```bash
# Attach to container
modal container list
modal shell ta-XXXXX

# Check what's in the volume (from repo root)
cd /root/searchlm && ls -lh data

# Run a data processing script
python scripts/process_data.py --input data/raw --output output/processed

# Verify output
ls -lh output/processed

# Exit
exit
```

## Alternative: Non-Interactive Function Call

If you just want to run a script and see output (no interactive shell):

```bash
# Run function directly (sees STDOUT)
modal run modal_infra.py::launcher --cmd-args "--arg1 value1"
```

This executes the script and shows all output in your terminal, but doesn't give you an interactive shell.

## Next Steps

- **View logs and metrics:** Check the Modal dashboard link shown on launch
- **List volumes:** `modal volume ls searchlm-data`
- **Monitor containers:** `modal container list`
- **Hot reload for development:** Try `modal serve modal_infra.py`

---

**Questions or issues?** Check Modal docs: https://modal.com/docs/guide/developing-debugging
