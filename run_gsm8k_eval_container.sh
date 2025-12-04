#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --job-name=nanochat_container
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_container_%j.out
#SBATCH --error=slurm_container_%j.err

# ==========================================
# GSM8K Evaluation on Amarel (via Container)
# ==========================================

# 1. Load Singularity module
module load singularity

# 2. Configuration
# Path to your Singularity image (upload it or pull it here)
IMAGE_PATH="$HOME/llm-research/nanochat.sif"

# Directory for results
OUTPUT_DIR="$HOME/gsm8k_container_results/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# 3. Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Singularity image not found at $IMAGE_PATH"
    echo "Please follow instructions in AMAREL_CONTAINER_SETUP.md to build/pull it."
    exit 1
fi

echo "Starting evaluation in container..."
echo "Image: $IMAGE_PATH"
echo "Output: $OUTPUT_DIR"

# 4. Run evaluation
# --nv: Enable NVIDIA GPU support
# --bind $PWD:/app: Mount current directory to /app to use latest code
# Note: Environment variables (like WANDB_API_KEY) are passed to the container by default
singularity exec --nv --bind "$PWD:/app" "$IMAGE_PATH" \
    python /app/eval_gsm8k_all_models.py \
    --output-dir "$OUTPUT_DIR" \
    --models default

echo "Job completed."

