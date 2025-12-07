#!/bin/bash
#SBATCH --job-name=gsm8k_finetune
#SBATCH --output=gsm8k_finetune_%j.out
#SBATCH --error=gsm8k_finetune_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@rutgers.edu

# ==========================================
# Amarel Cluster GSM8K Finetuning Script
# ==========================================
# This script finetunes the default model on GSM8K dataset.
# It uses SLURM for scheduling and torchrun for multi-GPU (DDP) support.
#
# Usage: sbatch run_finetune_gsm8k_amarel.sh

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "User: $USER"
echo "Home: $HOME"
echo "=========================================="

# --- 🐍 CONDA ENVIRONMENT SETUP ---

# Determine repository location
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(pwd)"
fi
echo "Repository directory: $REPO_DIR"
cd "$REPO_DIR" || { echo "ERROR: Cannot change to repository directory"; exit 1; }

# Set up environment variables
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
echo "Cache directory: $NANOCHAT_BASE_DIR"

# Activate Conda Environment
echo "Activating Conda environment: nanochat_proper"
source ~/.bashrc  # Ensure conda is initialized
# Fallback if .bashrc doesn't initialize conda correctly in non-interactive shell
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi

conda activate nanochat_proper || { echo "ERROR: Failed to activate conda environment 'nanochat_proper'"; exit 1; }
echo "Python path: $(which python)"

# Load required modules
module use /projects/community/modulefiles
module load gcc/12.1

# --- 🩺 VERIFICATION ---

# Verify GPU availability
echo "=========================================="
echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"
echo "=========================================="

# Check Python and PyTorch
echo "Python version: $(python --version 2>&1 || echo 'ERROR: Python not found')"
if python -c "import sys; print(f'Python {sys.version}')" 2>&1; then
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>&1 || echo 'PyTorch not installed')"
    # Check CUDA
    if python -c 'import torch; print(torch.cuda.is_available())' 2>&1 | grep -q True; then
        echo "CUDA available: True"
        echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
        echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>&1)"
    else
        echo "CUDA available: False. Check PyTorch/CUDA installation and GPU allocation."
    fi
else
    echo "ERROR: Python is not working correctly."
fi
echo "=========================================="

# --- 🚀 RUN JOB ---

# Create output directory with job ID
if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    OUTPUT_DIR="/scratch/$USER/gsm8k_finetune/job_${SLURM_JOB_ID}"
    echo "Using /scratch for output (auto-purged after 90 days)"
else
    OUTPUT_DIR="$HOME/gsm8k_finetune/job_${SLURM_JOB_ID}"
    echo "Using /home for output (note: limited to 100GB quota)"
fi
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Set wandb run name (optional - set to "dummy" to disable wandb)
WANDB_RUN="${WANDB_RUN:-gsm8k-default-${SLURM_JOB_ID}}"
echo "Wandb run name: $WANDB_RUN"

# Run finetuning
echo "Starting GSM8K finetuning on default model..."
echo "Results will be saved to: $OUTPUT_DIR"
echo "=========================================="

# Run the finetuning script
cd "$REPO_DIR"
# Use torchrun to utilize both GPUs (DDP)
# We use a random port based on job ID to avoid collisions
PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
torchrun --nproc_per_node=2 --master_port=$PORT "$REPO_DIR/finetune_gsm8k.py" \
    --model default \
    --run "$WANDB_RUN" \
    --device-type cuda \
    --dtype bfloat16 \
    --device-batch-size 4 \
    --target-examples-per-step 32 \
    --num-epochs 3 \
    --eval-every 100 \
    --eval-steps 50 \
    --save-every 200 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training_log.txt"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# --- 📋 POST-PROCESSING ---

echo "=========================================="
echo "Training Complete!"
echo "Results directory: $OUTPUT_DIR"
echo "End Time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/model_*.pt 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo ""
    echo "Latest checkpoint: $LATEST_CHECKPOINT"
    echo "Checkpoint size: $(du -h "$LATEST_CHECKPOINT" | cut -f1)"
fi

# Generate summary report
echo ""
echo "Generating summary report..."
python - << EOF
import json
import sys
import os
import glob
from datetime import datetime

try:
    output_dir = "$OUTPUT_DIR"
    summary_file = os.path.join(output_dir, "summary.txt")
    
    # Find latest meta file
    meta_files = glob.glob(os.path.join(output_dir, "meta_*.json"))
    if not meta_files:
        print("No checkpoint metadata found")
        sys.exit(0)
    
    latest_meta = max(meta_files, key=os.path.getmtime)
    
    with open(latest_meta, 'r') as f:
        meta = json.load(f)
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GSM8K Finetuning Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Job ID: $SLURM_JOB_ID\n")
        f.write(f"Node: $SLURM_NODELIST\n")
        f.write(f"Wandb Run: $WANDB_RUN\n")
        f.write("\n")
        
        f.write("Training Parameters:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Model: default\n")
        f.write(f"  Device: cuda (2 GPUs)\n")
        f.write(f"  Dtype: bfloat16\n")
        f.write(f"  Device Batch Size: 4\n")
        f.write(f"  Target Examples per Step: 32\n")
        f.write(f"  Number of Epochs: 3\n")
        f.write("\n")
        
        f.write("Final Checkpoint:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Step: {meta.get('step', 'N/A')}\n")
        f.write(f"  Validation Loss: {meta.get('val_loss', 'N/A')}\n")
        f.write(f"  Source Model: {meta.get('source_model', 'default')}\n")
        f.write(f"  Number of Epochs: {meta.get('num_epochs', 'N/A')}\n")
        f.write("\n")
        
        if 'model_config' in meta:
            f.write("Model Configuration:\n")
            f.write("-" * 80 + "\n")
            config = meta['model_config']
            f.write(f"  Layers: {config.get('n_layer', 'N/A')}\n")
            f.write(f"  Embedding Dim: {config.get('n_embd', 'N/A')}\n")
            f.write(f"  Attention Heads: {config.get('n_head', 'N/A')}\n")
            f.write(f"  Vocab Size: {config.get('vocab_size', 'N/A')}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint Directory: {output_dir}\n")
        f.write(f"Latest Checkpoint: {latest_meta.replace('meta_', 'model_').replace('.json', '.pt')}\n")
        f.write(f"Training Log: {os.path.join(output_dir, 'training_log.txt')}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved to: {summary_file}")
    
    # Also print to stdout
    with open(summary_file, 'r') as f:
        print(f.read())
        
except Exception as e:
    print(f"Error generating summary: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "Job completed successfully!"
echo "Checkpoint directory: $OUTPUT_DIR"
echo "=========================================="
