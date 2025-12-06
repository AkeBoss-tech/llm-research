#!/bin/bash
# ==========================================
# Local Mac Benchmark Configuration
# ==========================================
# Use this script for running benchmarks on your local Mac.
# It uses Metal (MPS) backend for acceleration.
#
# Usage: ./run_benchmark_mac.sh

# Check for MPS support
echo "=========================================="
echo "Checking for Mac (Apple Silicon) MPS support..."
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())" || { echo "ERROR: Python/PyTorch check failed."; exit 1; }
echo "=========================================="

# --- 🔧 SETUP ---

REPO_DIR="$(pwd)"
echo "Repository directory: $REPO_DIR"

# Set up output directory
OUTPUT_DIR="$REPO_DIR/output/local_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# --- 🚀 RUN JOB ---

echo "Starting GSM8K evaluation on Mac..."
echo "Note: Using float32/float16 on MPS. Multi-GPU (DDP) is disabled."
echo "=========================================="

# Run the evaluation script
# Note: We run sequentially on 1 device. No torchrun needed for local single-device run.
python "$REPO_DIR/eval_gsm8k_all_models.py" \
    --models default d32 d34 \
    --max-problems 100 \
    --num-samples 1 \
    --temperature 0.0 \
    --top-k 50 \
    --max-new-tokens 256 \
    --device-type mps \
    --dtype float16 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/eval_log.txt"

# Note: Reduced max-problems to 100 for local testing speed.
# Note: Used --dtype float16 as bfloat16 support on MPS can be flaky depending on macOS version.

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

# --- 📋 POST-PROCESSING ---

RESULTS_FILE=$(ls -t "$OUTPUT_DIR"/gsm8k_results_*.json 2>/dev/null | head -1)

if [ -z "$RESULTS_FILE" ]; then
    echo "ERROR: No results file found!"
    exit 1
fi

# Generate HTML visualization
echo "Generating HTML visualization..."
python "$REPO_DIR/visualize_gsm8k_results.py" "$RESULTS_FILE" --output "$OUTPUT_DIR/report.html"

echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Open $OUTPUT_DIR/report.html to view results."
echo "=========================================="

