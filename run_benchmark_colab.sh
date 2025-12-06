#!/bin/bash
# ==========================================
# Google Colab Benchmark Configuration
# ==========================================
# Use this script inside a Google Colab cell (e.g., !bash run_benchmark_colab.sh).
# It handles dependency installation and runs the models efficiently to avoid OOM.
#
# Usage: !bash run_benchmark_colab.sh

echo "=========================================="
echo "Setting up environment for Google Colab..."
echo "=========================================="

# 1. Install Rust (needed for tokenizer)
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust already installed."
fi

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install maturin datasets fastapi files-to-prompt psutil regex tiktoken tokenizers torch uvicorn wandb huggingface_hub

# 3. Install local package
echo "Installing nanochat..."
pip install -e .

# --- 🚀 RUN JOBS ---

echo "=========================================="
echo "Starting Benchmark Run"
echo "Warning: Running models SEQUENTIALLY to avoid OOM on Colab 12GB/16GB instances."
echo "=========================================="

REPO_DIR="$(pwd)"

# Mount Google Drive
echo "Mounting Google Drive..."
if [ ! -d "/content/drive" ]; then
    python -c "from google.colab import drive; drive.mount('/content/drive')"
fi

# Create output directory in Drive
OUTPUT_DIR="/content/drive/MyDrive/gsm8k_eval_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Results will be saved to: $OUTPUT_DIR"

# Function to run evaluation for a single model
run_eval() {
    MODEL_NAME=$1
    echo "------------------------------------------"
    echo "Evaluating Model: $MODEL_NAME"
    echo "------------------------------------------"
    
    python "$REPO_DIR/eval_gsm8k_all_models.py" \
        --models "$MODEL_NAME" \
        --max-problems 1000 \
        --num-samples 1 \
        --temperature 0.0 \
        --top-k 50 \
        --max-new-tokens 256 \
        --dtype float16 \
        --output-dir "$OUTPUT_DIR" \
        --no-save # Don't save intermediate big JSONs, we'll rely on the individual run logs or final aggregation if we wanted
    
    # Note: If you want to save results per run, remove --no-save. 
    # But Python script saves by timestamp, so it's fine.
}

# Run d32
echo "Running d32..."
python "$REPO_DIR/eval_gsm8k_all_models.py" \
    --models d32 \
    --max-problems 1000 \
    --dtype float16 \
    --max-new-tokens 256 \
    --output-dir "$OUTPUT_DIR"

# Suggest user to restart runtime if they want to run d34 safely
echo "=========================================="
echo "d32 Run Complete."
echo "To run d34 (Larger Model), it is HIGHLY RECOMMENDED to restart the runtime now to clear RAM."
echo "Then run: !python eval_gsm8k_all_models.py --models d34 --dtype float16 --max-new-tokens 256"
echo "=========================================="

# Optionally try running d34 anyway (uncomment if you have High-RAM)
# echo "Attempting d34..."
# python "$REPO_DIR/eval_gsm8k_all_models.py" --models d34 --dtype float16 --max-new-tokens 256 --output-dir "$OUTPUT_DIR"

# Visualization
LATEST_RESULT=$(ls -t "$OUTPUT_DIR"/gsm8k_results_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_RESULT" ]; then
    echo "Generating report for latest run..."
    python "$REPO_DIR/visualize_gsm8k_results.py" "$LATEST_RESULT" --output "$OUTPUT_DIR/report.html"
    echo "Report saved to $OUTPUT_DIR/report.html"
fi

