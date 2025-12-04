#!/bin/bash
#SBATCH --job-name=gsm8k_eval
#SBATCH --output=gsm8k_eval_%j.out
#SBATCH --error=gsm8k_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@rutgers.edu

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

# --- 🎯 UV ENVIRONMENT SETUP (Python 3.10 & PyTorch) ---

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
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
echo "Cache directory: $NANOCHAT_BASE_DIR"

# 1. Install uv if not already installed
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    # Try to source .bashrc first (uv might be in PATH after sourcing)
    source "$HOME/.bashrc" 2>/dev/null || true
    if ! command -v uv &> /dev/null; then
        # Install uv to ~/.cargo/bin (default location)
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not available. Please install it manually:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is available: $(which uv)"

# 2. Create virtual environment with Python 3.10 (if it doesn't exist)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.10..."
    uv venv --python 3.10 || {
        echo "WARNING: uv venv with Python 3.10 failed, trying with system Python 3.10+..."
        # Fallback: try to find Python 3.10+ and create venv
        for py in python3.10 python3.11 python3.12 python3; do
            if command -v "$py" &> /dev/null; then
                PY_VER=$("$py" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
                PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
                PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
                if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ]; then
                    "$py" -m venv .venv
                    echo "Created venv with $py ($PY_VER)"
                    break
                fi
            fi
        done
    }
else
    echo "Virtual environment already exists at .venv"
fi

# 3. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || { echo "ERROR: Failed to activate virtual environment"; exit 1; }

# 4. Install dependencies using uv (includes PyTorch with CUDA 11.8 for GLIBC compatibility)
echo "Installing dependencies with uv (including PyTorch with CUDA 11.8 for GLIBC compatibility)..."
# Remove lock file if it exists to force regeneration with correct PyTorch version
if [ -f "uv.lock" ]; then
    echo "Removing old uv.lock to regenerate with compatible PyTorch version..."
    rm -f uv.lock
fi
uv sync --extra gpu || {
    echo "WARNING: uv sync failed, trying manual installation with compatible PyTorch version..."
    pip install --upgrade pip setuptools wheel
    # Install PyTorch 2.1.2 with CUDA 11.8 (compatible with manylinux_2_17 / older GLIBC)
    echo "Installing PyTorch 2.1.2 with CUDA 11.8..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    # Install other dependencies
    pip install datasets>=4.0.0 fastapi>=0.117.1 files-to-prompt>=0.6 psutil>=7.1.0 \
        regex>=2025.9.1 tiktoken>=0.11.0 tokenizers>=0.22.0 uvicorn>=0.36.0 wandb>=0.21.3 pandas>=2.0.0
    pip install "maturin>=1.7,<2.0" || echo "Warning: maturin installation failed, rustbpe may not work"
}

# --- 🩺 VERIFICATION ---

# Verify GPU availability
echo "=========================================="
echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"
echo "=========================================="

# Check Python and PyTorch
echo "Virtual Environment: .venv"
echo "Python version: $(python --version 2>&1 || echo 'ERROR: Python not found')"
echo "Python path: $(which python)"
if python -c "import sys; print(f'Python {sys.version}')" 2>&1; then
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>&1 || echo 'PyTorch not installed')"
    # Check CUDA
    if python -c 'import torch; print(torch.cuda.is_available())' 2>&1 | grep -q True; then
        echo "CUDA available: True"
        echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
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
    OUTPUT_DIR="/scratch/$USER/gsm8k_eval_results/job_${SLURM_JOB_ID}"
    echo "Using /scratch for output (auto-purged after 90 days)"
else
    OUTPUT_DIR="$HOME/gsm8k_eval_results/job_${SLURM_JOB_ID}"
    echo "Using /home for output (note: limited to 100GB quota)"
fi
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run evaluation
echo "Starting GSM8K evaluation..."
echo "Results will be saved to: $OUTPUT_DIR"
echo "=========================================="

# Run the evaluation script
cd "$REPO_DIR"
python "$REPO_DIR/eval_gsm8k_all_models.py" \
    --models default d32 d34 \
    --max-problems 1000 \
    --num-samples 1 \
    --temperature 0.0 \
    --top-k 50 \
    --max-new-tokens 512 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/eval_log.txt"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

# --- 📋 POST-PROCESSING AND CLEANUP ---

# Find the most recent results file
RESULTS_FILE=$(ls -t "$OUTPUT_DIR"/gsm8k_results_*.json 2>/dev/null | head -1)

if [ -z "$RESULTS_FILE" ]; then
    echo "ERROR: No results file found!"
    exit 1
fi

echo "Found results file: $RESULTS_FILE"

# Generate HTML visualization
echo "Generating HTML visualization..."
cd "$REPO_DIR"
python "$REPO_DIR/visualize_gsm8k_results.py" "$RESULTS_FILE" --output "$OUTPUT_DIR/report.html"

VIS_EXIT_CODE=$?

if [ $VIS_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Visualization generation failed with exit code $VIS_EXIT_CODE"
else
    echo "HTML report generated: $OUTPUT_DIR/report.html"
fi

# Generate summary report (Inline Python)
echo "Generating summary report..."
python - << EOF
import json
import sys
from datetime import datetime

try:
    with open("$RESULTS_FILE", 'r') as f:
        data = json.load(f)
    
    summary_file = "$OUTPUT_DIR/summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GSM8K Evaluation Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Job ID: $SLURM_JOB_ID\n")
        f.write(f"Node: $SLURM_NODELIST\n")
        f.write("\n")
        
        f.write("Evaluation Parameters:\n")
        f.write("-" * 80 + "\n")
        for key, value in data.get('eval_params', {}).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Results Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Repository':<35} {'Step':<10} {'Accuracy':<12} {'Correct/Total':<15}\n")
        f.write("-" * 80 + "\n")
        for result in data.get('summary', []):
            f.write(f"{result['model_name']:<15} {result['hf_repo_id']:<35} {result['step']:<10} {result['accuracy_percent']:>6.2f}% {result.get('num_correct', 0)}/{result.get('num_total', 0)}\n")
        f.write("\n")
        
        # Find best model
        if data.get('summary'):
            best = max(data['summary'], key=lambda x: x['accuracy'])
            f.write(f"Best Model: {best['model_name']} with {best['accuracy_percent']:.2f}% accuracy\n")
            f.write(f"  Repository: {best['hf_repo_id']}\n")
            f.write(f"  Step: {best['step']}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"Results JSON: $RESULTS_FILE\n")
        f.write(f"HTML Report: $OUTPUT_DIR/report.html\n")
        f.write(f"Summary: {summary_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved to: {summary_file}")
    
    # Also print to stdout
    with open(summary_file, 'r') as f:
        print(f.read())
        
except Exception as e:
    print(f"Error generating summary: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo "=========================================="
echo "Evaluation Complete!"
echo "Results directory: $OUTPUT_DIR"
echo "End Time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"