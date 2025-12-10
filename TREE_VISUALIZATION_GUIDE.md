# Tree of Thought Visualization Guide

## New Features

### 1. Progress Printing
The lookahead decoding now prints progress as it explores candidates:
- Shows which candidates are being explored
- Displays rollout previews for each candidate
- Shows which candidate was selected at each step

### 2. Tree Data Collection
Tree data is automatically collected and saved to JSON files for visualization.

### 3. Web Visualization
An interactive HTML visualization shows the entire tree structure.

## Usage

### Running with Progress Output

```bash
python test_lookahead_gsm8k.py -i default
```

You'll see output like:
```
[Step 1/50] Exploring 3 candidates...
  Candidate 1: 'The' (prob: 0.1234)
    Rollout preview: 'The answer is...' (score: -2.3456)
  Candidate 2: 'To' (prob: 0.0987)
    Rollout preview: 'To solve this...' (score: -3.4567)
  Candidate 3: 'We' (prob: 0.0876)
    Rollout preview: 'We need to...' (score: -4.5678)
  ✓ Selected candidate 1: 'The'
  Generated so far: 'The answer is 42'
```

### Viewing the Tree Visualization

1. Run the test script:
```bash
python test_lookahead_gsm8k.py -i default --num-problems 1
```

2. The script will create JSON files like `tree_data_problem_1.json`

3. Open `visualize_tree_of_thought.html` in your browser

4. Click "Choose File" and select the JSON file

5. The visualization will show:
   - Problem question and answer
   - Each step of generation
   - All candidates explored at each step
   - Which candidate was selected (highlighted in green)
   - Rollout previews for each candidate
   - Scores and probabilities

## Visualization Features

- **Step-by-step view**: See each generation step
- **Candidate comparison**: Compare all candidates side-by-side
- **Selected candidate**: Highlighted in green
- **Rollout previews**: See what each candidate would generate
- **Scores**: See cumulative log probability scores
- **Probabilities**: Initial token probabilities

## Example Output

After running, you'll see:
- Console output with progress
- JSON files with tree data
- Instructions to open the visualization

## Tips

1. **Start with 1 problem**: Use `--num-problems 1` to test quickly
2. **Reduce depth**: Use `--lookahead-depth 5` for faster testing
3. **Reduce candidates**: Use `--lookahead-k 2` for faster exploration
4. **View in browser**: The HTML visualization is interactive and shows the full tree

## Files Created

- `tree_data_problem_N.json` - Tree data for each problem
- `visualize_tree_of_thought.html` - Visualization interface

## Troubleshooting

**No progress output**: Make sure `verbose=True` is set (it's automatic in the test script)

**Visualization not loading**: Check browser console for errors, ensure JSON file is valid

**Too slow**: Reduce `--lookahead-k` or `--lookahead-depth` parameters
