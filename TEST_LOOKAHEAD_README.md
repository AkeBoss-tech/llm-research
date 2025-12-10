# Testing Lookahead Decoding on GSM8K

## Quick Start

Test lookahead decoding on 10 GSM8K problems:

```bash
python test_lookahead_gsm8k.py -i default
```

## Usage

```bash
# Basic usage (default model, 10 problems)
python test_lookahead_gsm8k.py -i default

# Test with different model
python test_lookahead_gsm8k.py -i d32

# Customize lookahead parameters
python test_lookahead_gsm8k.py -i default \
    --lookahead-k 5 \
    --lookahead-depth 15

# Test more problems
python test_lookahead_gsm8k.py -i default --num-problems 20

# Adjust standard generation parameters
python test_lookahead_gsm8k.py -i default \
    --temperature 0.5 \
    --top-k 30
```

## Arguments

### Model Selection
- `-i, --source`: Model source (default, d32, d34, sft, mid, rl)
- `-g, --model-tag`: Model tag to load
- `-s, --step`: Step to load from checkpoint

### Generation Parameters
- `-t, --temperature`: Temperature for standard generation (default: 0.7)
- `-k, --top-k`: Top-k for standard generation (default: 50)
- `--max-tokens`: Maximum tokens to generate (default: 512)

### Lookahead Parameters
- `--lookahead-k`: Number of candidates to explore (default: 3)
- `--lookahead-depth`: How many tokens to look ahead (default: 10)
- `--lookahead-temp`: Temperature for lookahead (default: 0.7)

### Other
- `-n, --num-problems`: Number of problems to test (default: 10)
- `--device-type`: Device type (cuda, cpu, mps)
- `-d, --dtype`: Data type (float32, bfloat16)

## Output

The script will:
1. Load the model and GSM8K test set
2. For each problem:
   - Show the question and ground truth answer
   - Run standard generation and show results
   - Run lookahead decoding and show results
   - Compare the two methods
3. Show a final summary with:
   - Accuracy for both methods
   - Average time per problem
   - Improvement metrics

## Example Output

```
================================================================================
Problem 1/10
================================================================================
Question: Janet has 16 candy bars. She eats 3 of them. How many does she have left?
Ground Truth: 13

--- Standard Generation ---
Time: 2.34s
Answer: 13
Correct: ✓
Response:
Janet has 16 candy bars. She eats 3 of them. So she has 16 - 3 = 13 candy bars left.
#### 13

--- Lookahead Decoding ---
Time: 45.67s
Answer: 13
Correct: ✓
Response:
Janet has 16 candy bars. She eats 3 of them. So she has 16 - 3 = 13 candy bars left.
#### 13

--- Comparison ---
Result: Same
Speed: 19.52x slower
```

## Tips

1. **Start with small K and D**: `--lookahead-k 3 --lookahead-depth 5` for faster testing
2. **Use smaller model**: `-i default` (d20) is faster than `-i d32` or `-i d34`
3. **Test on fewer problems first**: `--num-problems 3` to verify it works
4. **Lookahead is slower**: Expect 10-30x slower than standard generation

## Troubleshooting

**Out of memory**: Reduce `--lookahead-k` or `--lookahead-depth`

**Very slow**: Lookahead is intentionally slower. Try smaller K and D values.

**No improvement**: 
- Try increasing `--lookahead-k` or `--lookahead-depth`
- Check if the model is well-calibrated
- Some problems may not benefit from lookahead
