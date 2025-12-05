# Running Nanochat on Amarel with Docker/Singularity

This guide explains how to run the project on the Amarel HPC cluster using a container. This avoids the need to build Python from source or manage complex environments on the cluster.

## 1. Build the Docker Image (Locally)

You generally cannot build Docker images directly on Amarel. Build it on your local machine.

**Important for Mac (Apple Silicon) users:** You must specify `--platform linux/amd64` to build a compatible image for the HPC cluster.

```bash
# Build the image (replace 'yourusername' with your Docker Hub username)
# The --platform linux/amd64 flag is CRITICAL if building on an Apple Silicon Mac
docker build --platform linux/amd64 -t yourusername/nanochat:latest .

# Push to Docker Hub (or another registry)
docker push yourusername/nanochat:latest
```

## 2. Use Singularity/Apptainer on Amarel

Amarel uses Singularity (now Apptainer) to run containers.

### Option A: Pull from Docker Hub (Recommended)

1.  **SSH into Amarel**:
    ```bash
    ssh username@amarel.rutgers.edu
    cd ~/llm-research
    ```

2.  **Pull the image**:
    ```bash
    # This converts the Docker image to a Singularity image (.sif)
    singularity pull nanochat.sif docker://yourusername/nanochat:latest
    ```

### Option B: Build Singularity Image Locally and Upload

If you can't push to Docker Hub, you can save the Docker image and convert it, or build the SIF locally if you have Singularity installed.
Otherwise, just use Option A.

## 3. Run the Job

Create a SLURM script (e.g., `run_container_job.sh`) that uses the container.

### Example SLURM Script

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=nanochat_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load Singularity module
module load singularity

# Define paths
IMAGE_PATH="$HOME/llm-research/nanochat.sif"
REPO_DIR="$HOME/llm-research"

# Run the evaluation script inside the container
# We bind mount the current directory to /app so changes are reflected
# The virtual environment inside the container (/venv) is used
singularity exec --nv --bind "$REPO_DIR:/app" "$IMAGE_PATH" \
    python /app/eval_gsm8k_all_models.py \
    --output-dir "/app/gsm8k_results_$SLURM_JOB_ID"
```

## 4. Interactive Mode

To debug or run interactively:

```bash
# Start an interactive session on a GPU node
srun --partition=gpu --gres=gpu:1 --time=1:00:00 --pty bash

# Load singularity
module load singularity

# Shell into the container
# --nv enables GPU support
singularity shell --nv --bind .:/app nanochat.sif

# Now you are inside the container
(nanochat) $ python eval_gsm8k_all_models.py --help
```

## Notes

*   **GPU Support**: The `--nv` flag is crucial for Singularity to pass the GPU drivers from the host to the container.
*   **Virtual Environment**: The container has a pre-configured virtual environment at `/venv`. The `PATH` is already set, so you can just run `python`.
*   **Binding**: `--bind .:/app` maps your current directory on Amarel to `/app` inside the container. This allows you to edit code on Amarel and run it immediately without rebuilding the container. The installed packages (like `torch`) remain in `/venv`.

