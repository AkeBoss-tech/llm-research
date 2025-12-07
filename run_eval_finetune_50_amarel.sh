#!/bin/bash
#SBATCH --job-name=eval_finetune_50
#SBATCH --output=eval_finetune_50_%j.out
#SBATCH --error=eval_finetune_50_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@rutgers.edu

# ==========================================
# Amarel Cluster Script: Evaluate Finetune 50 Steps Model
# ==========================================
# This script evaluates the checkpoint at step 748 (final from 50-step finetune) on GSM8K.

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
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

# Activate Conda Environment
echo "Activating Conda environment: nanochat_proper"
source ~/.bashrc
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi

conda activate nanochat_proper || { echo "ERROR: Failed to activate conda environment"; exit 1; }
echo "Python path: $(which python)"

# Load required modules
module use /projects/community/modulefiles
module load gcc/12.1

# --- 🩺 VERIFICATION ---

echo "=========================================="
echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"
echo "=========================================="

echo "Python version: $(python --version 2>&1)"
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1; then
    if python -c 'import torch; print(torch.cuda.is_available())' 2>&1 | grep -q True; then
        echo "CUDA available: True"
        echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
        echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>&1)"
    else
        echo "CUDA available: False"
    fi
fi
echo "=========================================="

# --- 🚀 RUN EVALUATION ---

# Set checkpoint directory and step
CHECKPOINT_DIR="$HOME/llm-research/output/default_finetune_50"
STEP=748

echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Step: $STEP"
echo "=========================================="

# Create output directory
if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    OUTPUT_DIR="/scratch/$USER/gsm8k_eval_finetune_50/job_${SLURM_JOB_ID}"
    echo "Using /scratch for output"
else
    OUTPUT_DIR="$HOME/gsm8k_eval_finetune_50/job_${SLURM_JOB_ID}"
    echo "Using /home for output"
fi
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run evaluation
echo "Starting evaluation..."
cd "$REPO_DIR"

# Use torchrun to utilize both GPUs (DDP)
PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
torchrun --nproc_per_node=2 --master_port=$PORT "$REPO_DIR/eval_checkpoint_698.py" -- \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --step $STEP \
    --max-problems 1000 \
    --num-samples 1 \
    --temperature 0.0 \
    --top-k 50 \
    --max-new-tokens 512 \
    --dtype bfloat16 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/eval_log.txt"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

# --- 📋 POST-PROCESSING ---

echo "=========================================="
echo "Evaluation Complete!"
echo "Results directory: $OUTPUT_DIR"
echo "End Time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"

# Find the results file and print summary
RESULTS_FILE=$(ls -t "$OUTPUT_DIR"/eval_step_*.json 2>/dev/null | head -1)
if [ -n "$RESULTS_FILE" ]; then
    echo ""
    echo "Results file: $RESULTS_FILE"
    echo ""
    echo "Summary:"
    python - << EOF
import json
import sys

try:
    with open("$RESULTS_FILE", 'r') as f:
        data = json.load(f)
    
    print(f"Checkpoint: {data.get('checkpoint_dir', 'N/A')} (step {data.get('step', 'N/A')})")
    print(f"GSM8K Accuracy: {data.get('accuracy_percent', 0):.2f}%")
    print(f"Correct: {data.get('num_correct', 0)}/{data.get('num_total', 0)}")
    print(f"Evaluation parameters:")
    for key, value in data.get('eval_params', {}).items():
        print(f"  {key}: {value}")
        
except Exception as e:
    print(f"Error reading results: {e}", file=sys.stderr)
    sys.exit(1)
EOF
fi

echo ""
echo "=========================================="
echo "Evaluation job completed!"
echo "=========================================="
