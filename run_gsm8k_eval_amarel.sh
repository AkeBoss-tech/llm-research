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

# Check for Python 3.10+ requirement
# Amarel only has Python 3.8.2 via modules, so we need conda or custom Python
echo "Checking Python availability..."
echo "Note: nanochat requires Python >=3.10, but Amarel modules only provide 3.8.2"

# Check for custom Python installation (built from source)
CUSTOM_PYTHON_DIR="$HOME/python/3.10.13"  # Default version from setup script
if [ -f "$CUSTOM_PYTHON_DIR/bin/python3" ]; then
    echo "Found custom Python installation at $CUSTOM_PYTHON_DIR"
    export PATH="$CUSTOM_PYTHON_DIR/bin:$PATH"
    # Set LD_LIBRARY_PATH to include Python's lib directory (critical for venv to work)
    export LD_LIBRARY_PATH="$CUSTOM_PYTHON_DIR/lib:${LD_LIBRARY_PATH:-}"
    # Also set PYTHONHOME if needed
    export PYTHONHOME="$CUSTOM_PYTHON_DIR"
    CUSTOM_PYTHON_AVAILABLE=true
else
    # Try to find any Python 3.10+ in ~/python/
    for py_dir in "$HOME"/python/3.1* "$HOME"/python/3.2*; do
        if [ -f "$py_dir/bin/python3" ]; then
            PYTHON_VER=$(basename "$py_dir")
            echo "Found custom Python installation: $py_dir"
            export PATH="$py_dir/bin:$PATH"
            export LD_LIBRARY_PATH="$py_dir/lib:${LD_LIBRARY_PATH:-}"
            export PYTHONHOME="$py_dir"
            CUSTOM_PYTHON_AVAILABLE=true
            break
        fi
    done
    if [ "${CUSTOM_PYTHON_AVAILABLE:-false}" != "true" ]; then
        echo "No custom Python found. Will try conda/miniconda or system Python."
        CUSTOM_PYTHON_AVAILABLE=false
    fi
fi

# Determine repository location
# SLURM sets SLURM_SUBMIT_DIR to the directory where sbatch was run
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    # Fallback: use current working directory
    REPO_DIR="$(pwd)"
fi
echo "Repository directory: $REPO_DIR"
cd "$REPO_DIR" || { echo "ERROR: Cannot change to repository directory"; exit 1; }

# Set up environment
export OMP_NUM_THREADS=1
# Use /home for cache (persistent, but limited to 100GB)
# Model files will be cached here for reuse across jobs
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
echo "Cache directory: $NANOCHAT_BASE_DIR"

# Load modules (adjust based on Amarel's available modules)
# Uncomment and modify as needed for your Amarel setup
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9

# Set up Python environment
# Priority: custom Python (from source) > conda/miniconda > uv > system Python
# Note: Project requires Python >=3.10, but Amarel's Python 3.8.2 has _ctypes issues

# Option 1: Use custom Python built from source (BEST - full control)
if [ "$CUSTOM_PYTHON_AVAILABLE" = true ]; then
    echo "Using custom Python installation..."
    # Verify Python version
    PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo "Python version check passed: $(python3 --version)"
        # Remove old venv if it exists (may have been created with wrong Python version)
        if [ -d "$REPO_DIR/.venv" ]; then
            echo "Removing old virtual environment (may have been created with wrong Python version)..."
            rm -rf "$REPO_DIR/.venv"
        fi
        # Create fresh venv with Python 3.10+
        echo "Creating virtual environment with Python 3.10+..."
        python3 -m venv "$REPO_DIR/.venv"
        source "$REPO_DIR/.venv/bin/activate"
        # Verify the venv is using the correct Python
        VENV_PYTHON_VER=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
        echo "Virtual environment Python version: $VENV_PYTHON_VER"
        pip install --upgrade pip setuptools wheel
        # Install PyTorch with CUDA support
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        # Install other dependencies manually
        pip install datasets>=4.0.0 fastapi>=0.117.1 files-to-prompt>=0.6 psutil>=7.1.0 \
            regex>=2025.9.1 tiktoken>=0.11.0 tokenizers>=0.22.0 uvicorn>=0.36.0 wandb>=0.21.3
        # Install maturin for building rustbpe (if needed)
        pip install "maturin>=1.7,<2.0" || echo "Warning: maturin installation failed, rustbpe may not work"
    else
        echo "ERROR: Custom Python version $PYTHON_VER is too old (need >=3.10)"
        CUSTOM_PYTHON_AVAILABLE=false
    fi
fi

# Option 2: Use conda/miniconda (if custom Python not available)
if [ "$CUSTOM_PYTHON_AVAILABLE" != true ]; then
    # Check for conda in common locations
    if command -v conda &> /dev/null; then
        CONDA_AVAILABLE=true
    elif [ -f "$HOME/miniconda3/bin/conda" ]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
        CONDA_AVAILABLE=true
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        export PATH="$HOME/anaconda3/bin:$PATH"
        CONDA_AVAILABLE=true
    else
        CONDA_AVAILABLE=false
    fi

    if [ "$CONDA_AVAILABLE" = true ]; then
    echo "Using conda for environment setup..."
    # Initialize conda if needed
    if [ -f "$HOME/.bashrc" ] && grep -q "conda initialize" "$HOME/.bashrc"; then
        source "$HOME/.bashrc"
    fi
    # Try to activate existing environment or create new one
    if conda env list | grep -q "nanochat"; then
        echo "Activating existing conda environment 'nanochat'..."
        conda activate nanochat
    else
        echo "Creating new conda environment 'nanochat' with Python 3.10..."
        conda create -n nanochat python=3.10 -y
        conda activate nanochat
        pip install --upgrade pip setuptools wheel
        # Install PyTorch with CUDA support
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        # Install other dependencies manually (skip problematic pip install -e .)
        pip install datasets>=4.0.0 fastapi>=0.117.1 files-to-prompt>=0.6 psutil>=7.1.0 \
            regex>=2025.9.1 tiktoken>=0.11.0 tokenizers>=0.22.0 uvicorn>=0.36.0 wandb>=0.21.3
        # Install maturin for building rustbpe (if needed)
        pip install maturin>=1.7,<2.0 || echo "Warning: maturin installation failed, rustbpe may not work"
    fi
# Option 2: Use uv (if available)
elif command -v uv &> /dev/null; then
    echo "Using uv for environment setup..."
    # Install uv if not in PATH but might be in ~/.cargo/bin
    if ! command -v uv &> /dev/null && [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    # Remove old venv if it exists
    if [ -d "$REPO_DIR/.venv" ]; then
        echo "Removing old virtual environment..."
        rm -rf "$REPO_DIR/.venv"
    fi
    # Create fresh venv
    uv venv "$REPO_DIR/.venv" --python 3.10 || uv venv "$REPO_DIR/.venv" || {
        echo "uv venv failed, trying with system Python 3.10+..."
        # Find Python 3.10+
        for py in python3.10 python3.11 python3.12 python3; do
            if command -v "$py" &> /dev/null; then
                PY_VER=$("$py" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
                PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
                PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
                if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ]; then
                    "$py" -m venv "$REPO_DIR/.venv"
                    break
                fi
            fi
        done
    }
    source "$REPO_DIR/.venv/bin/activate"
    uv sync --extra gpu || {
        echo "uv sync failed, trying manual installation..."
        pip install --upgrade pip setuptools wheel
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install datasets fastapi files-to-prompt psutil regex tiktoken tokenizers uvicorn wandb
    }
    # Option 3: Try system Python (may have _ctypes issues)
    else
        echo "ERROR: No suitable Python 3.10+ installation found!"
        echo ""
        echo "Please choose one of these options:"
        echo ""
        echo "Option A: Build Python from source (RECOMMENDED)"
        echo "  Run this script ONCE on a login node:"
        echo "    bash setup_python_amarel.sh"
        echo "  This will build Python 3.10.13 in ~/python/3.10.13/"
        echo ""
        echo "Option B: Install Miniconda"
        echo "  cd ~"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
        echo "  ~/miniconda3/bin/conda init bash"
        echo "  source ~/.bashrc"
        echo ""
        exit 1
    fi
fi

# Verify GPU availability
echo "=========================================="
echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"
echo "=========================================="

# Check Python and PyTorch
echo "Python version: $(python --version 2>&1 || echo 'ERROR: Python not found')"
echo "Python path: $(which python)"
if python -c "import sys; print(f'Python {sys.version}')" 2>&1; then
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>&1 || echo 'PyTorch not installed')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1 || echo 'PyTorch not installed')"
    if python -c 'import torch; print(torch.cuda.is_available())' 2>&1 | grep -q True; then
        echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
    fi
else
    echo "ERROR: Python is not working correctly. Check for _ctypes module issues."
    echo "Try using conda: conda create -n nanochat python=3.10 -y"
fi
echo "=========================================="

# Create output directory with job ID
# Use /scratch for results (larger quota, auto-purged after 90 days)
# If /scratch doesn't exist or isn't writable, fall back to home
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
# Adjust parameters as needed:
# --max-problems: Limit number of problems for faster testing
# --models: Specify which models to evaluate
# Make sure we're in the repo directory and use absolute path
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

# Generate summary report
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
