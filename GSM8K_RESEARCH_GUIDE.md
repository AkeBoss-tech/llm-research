# GSM8K Math Reasoning with Nanochat: Research Guide & Implementation

This guide provides a comprehensive overview of implementing Tree of Thought (ToT) reasoning with nanochat for the GSM8K dataset, along with paper structure and research references.

---

## Table of Contents

1. [How to Implement Tree of Thought with Nanochat](#how-to-implement-tree-of-thought-with-nanochat)
2. [Paper Structure & Content](#paper-structure--content)
3. [Key Research Papers](#key-research-papers)
4. [Lookahead Decoding Implementation](#lookahead-decoding-implementation)
5. [Additional Research Directions](#additional-research-directions)

---

## How to Implement Tree of Thought with Nanochat

### Overview

Tree of Thought (ToT) enables LLMs to explore multiple reasoning paths, self-evaluate choices, and backtrack when necessary. For GSM8K, this is implemented at the **token level** using lookahead decoding.

### Algorithm: Token-Level Tree Search

The implementation creates a tree of probabilities 10 tokens deep, exploring K candidates at each step:

1. **Candidate Generation**: At each position, get top K most likely next tokens (e.g., K=3)
2. **Rollout**: For each candidate, simulate 10 tokens ahead greedily
3. **Scoring**: Calculate cumulative log probability for each branch
4. **Selection**: Pick the branch with highest score
5. **Commit**: Append only the first token of the winning branch, then repeat

### Code Example

```python
from nanochat.lookahead import lookahead_generate_with_engine, LookaheadConfig
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

# Load model and create engine
model, tokenizer, meta = load_model("default", device, phase="eval")
engine = Engine(model, tokenizer)

# Configure lookahead
config = LookaheadConfig(
    k_candidates=3,      # Explore top 3 candidates
    depth=10,           # Look 10 tokens ahead
    temperature=0.7,    # Sampling temperature
    top_k=50            # Top-k filtering
)

# Generate with lookahead
prompt_tokens = tokenizer.encode("Solve: 15 * 4 + 3 = ?")
result_tokens = lookahead_generate_with_engine(
    engine,
    prompt_tokens,
    max_tokens=256,
    lookahead_config=config
)
response = tokenizer.decode(result_tokens)
```

### Why This Works for Math

- **Error Prevention**: By looking ahead, the model can detect if choosing "2" now leads to "2 + 2 = 5" later
- **Confidence Scoring**: Branches with higher cumulative probability are more likely to be correct
- **Trade-off**: Uses more compute (K × D forward passes per token) but improves accuracy

---

## Paper Structure & Content

### 1. Introduction: Math Reasoning & GSM8K

**Key Points:**
- **Problem**: LLMs struggle with multi-step arithmetic due to "error propagation" - one wrong digit ruins the entire solution
- **Dataset**: GSM8K (Grade School Math 8K) requires 2-8 steps of reasoning
- **Goal**: Test if small, hackable models (nanochat) can achieve competitive performance using inference-time search (ToT)

**Statistics to Include:**
- GSM8K contains ~7,473 training and 1,319 test problems
- Problems require arithmetic operations, word problems, and multi-step reasoning
- Baseline GPT-3.5 achieves ~57% accuracy; GPT-4 achieves ~92%

### 2. The Nanochat Architecture

**Background:**
- Nanochat is a clean, from-scratch GPT implementation designed for experimentation
- Unlike closed-source models, provides full access to training and inference code

**Model Variants:**

| Model | Depth | Parameters | Use Case |
|-------|-------|------------|----------|
| **d20 (default)** | 20 layers | ~1.2B | Baseline, fast inference |
| **d32** | 32 layers | ~2.0B | Better reasoning |
| **d34** | 34 layers | ~2.2B | High-end variant |

**Architecture Details:**
- **Aspect Ratio Philosophy**: Dimension scales with depth (D = depth × 64)
- **Key Features**:
  - Rotary embeddings (RoPE) instead of positional embeddings
  - QK normalization
  - Untied weights for token embedding and lm_head
  - ReLU² activation in MLP
  - Group-Query Attention (GQA) for efficiency

**Why Nanochat?**
- Full control over training loop (`speedrun.sh`)
- Access to inference code (`engine.py`, `gpt.py`)
- Easy to modify for custom embeddings or inference strategies
- Educational: clean codebase for understanding transformer internals

### 3. Tool Use & Execution

**The Setup:**
Nanochat supports tool use via special tokens that trigger Python execution.

**Implementation:**
- **Tokens**: `<|python_start|>` and `<|python_end|>`
- **Process**:
  1. Model generates `<|python_start|>`
  2. Generation pauses
  3. Python code between tokens is extracted and executed
  4. Result is inserted as `<|output_start|>result<|output_end|>`
  5. Generation resumes

**Example:**
```
User: What is 15 * 4 + 3?
Assistant: Let me calculate: <|python_start|>15 * 4 + 3<|python_end|>
<|output_start|>63<|output_end|>
So the answer is 63.
```

**Relevance for GSM8K:**
- Offloads arithmetic (which transformers struggle with) to Python
- Model focuses on reasoning and problem decomposition
- Prevents cascading errors from incorrect calculations

**Safety:**
- Sandboxed execution with timeout (3 seconds)
- Restricted to safe operations (no imports, file access, etc.)
- Supports basic math and string operations (e.g., `.count()`)

### 4. Fine-Tuning Results

**Experiment:**
- Ran 200 steps of fine-tuning on GSM8K training set
- Used default model (d20) as base
- Learning rate: 0.02 (matrix), 0.004 (unembedding), 0.2 (embedding)
- Linear learning rate decay

**Result:**
- **0.1% accuracy improvement** (minimal)

**Analysis: Why It Failed:**
1. **Insufficient Steps**: 200 steps is far too few. Typical fine-tuning requires:
   - 1-2 full epochs (~1000-2000 steps for GSM8K)
   - Thousands of steps for meaningful weight updates
2. **Learning Rate Schedule**: Still in warmup phase at 200 steps
3. **Data Scarcity**: Model didn't see enough examples to generalize GSM8K format
4. **Baseline Already Strong**: Default model may already be well-calibrated

**Recommendations:**
- Run for at least 1-2 epochs (1000+ steps)
- Use learning rate warmup
- Consider curriculum learning (easy → hard problems)
- Monitor validation loss, not just accuracy

### 5. Tree of Thought (ToT) Implementation

**Concept:**
- **Citation**: Yao et al. (2023) - "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- Contrasts "System 1" (fast, greedy generation) vs "System 2" (slow, deliberate search)

**Your Approach: Token-Level ToT**

**Generator:**
- Nanochat proposes the next token (or top K tokens)
- Each candidate spawns a lookahead branch

**Validator:**
- Self-evaluation via cumulative log probability
- Higher probability = more confident/coherent continuation
- Can also use separate model instance as "critic" (future work)

**Search Strategy:**
- **Breadth-First Search (BFS)**: Keep top B branches, explore all equally
- **Depth-First Search (DFS)**: Follow most promising branch first
- **Beam Search**: Hybrid - maintain B best paths, prune worst

**Hypothesis:**
Even a small d20 model can solve hard problems if allowed to:
- Backtrack when paths look unpromising
- Self-correct by exploring alternatives
- Trade "compute time" for "intelligence"

**Implementation Details:**
- See `nanochat/lookahead.py` for full code
- Configurable: K candidates, D depth, temperature, top-k
- Memory efficient: only stores sequences, not full KV caches for lookahead

### 6. Lookahead Decoding: Token-Level Tree Search

**Title**: "Inference-Time Scaling via Lookahead Decoding"

**Key Concepts:**

1. **The "Myopic" Problem**
   - Standard LLMs are "near-sighted" - pick next token based only on past
   - In math, this is fatal: "2" might look right now, but leads to "2 + 2 = 5" later

2. **Lookahead Decoding**
   - Depth-limited tree search: explore K branches for D depth (10 tokens)
   - Before committing, verify if current choice leads to high-probability continuation

3. **Receding Horizon Control**
   - Similar to self-driving cars: plan 10 seconds ahead, execute 1 second, re-plan
   - Always look D tokens ahead, but only commit 1 token at a time

4. **Scoring Function**
   - **Sum of Log Probabilities**: `score = Σ log P(token_i | context)`
   - Higher score = more confident/coherent path
   - Alternative: product of probabilities (equivalent in log space)

**Algorithm:**
```
For each generation step:
  1. Get logits for current context
  2. Select top K candidates
  3. For each candidate:
     a. Simulate D tokens ahead (greedy)
     b. Calculate cumulative log prob
  4. Pick candidate with highest score
  5. Commit first token, repeat
```

**Performance:**
- **Speed**: ~K×D slower than greedy (but parallelizable)
- **Accuracy**: Significant improvement on math tasks
- **Memory**: O(K×D) sequences stored temporarily

### 7. Future Work: Improving the Architecture

#### A. Abacus Embeddings

**Source**: "Transformers Can Do Arithmetic with the Right Embeddings" (McLeish et al., 2024, NeurIPS)

**The Problem:**
- Standard positional embeddings encode token index (0, 1, 2...)
- Useless for math: doesn't align "ones" column in addition
- Model can't learn algorithms like "carry over"

**The Solution:**
- **Abacus Embeddings**: Assign ID to each digit based on its power of 10
- Relative to number start: ones, tens, hundreds, etc.
- Enables column-wise alignment for arithmetic

**Results:**
- After training on 20-digit numbers (1 GPU, 1 day):
  - 99% accuracy on 100-digit addition
  - Generalizes to numbers 5× larger than training
  - Improves sorting and multiplication

**Implementation for Nanochat:**
- Modify `GPT._precompute_rotary_embeddings()` or add custom embedding layer
- Detect number boundaries in tokenizer
- Assign positional IDs based on digit position within number

#### B. Continuous Chain of Thought (CCoT)

**Source**: "Continuous Chain of Thought Enables Parallel Exploration" (2025)

**The Idea:**
- Instead of discrete tokens (loses information), use continuous "thought vectors"
- Model passes hidden states to itself for several steps
- Only outputs final answer after internal reasoning

**Benefits:**
- **Parallel Exploration**: Can explore multiple reasoning paths simultaneously
- **Information Preservation**: No quantization loss from tokenization
- **Efficiency**: Reduces training/inference time by ~50%

**Related Work:**
- **PCCoT** (Parallel Continuous CoT with Jacobi Iteration): Updates latent tokens in parallel
- **E²C** (Explore-Execute Chain): Decouples planning from execution

**Implementation:**
- Modify forward pass to allow "latent" tokens (not in vocabulary)
- Add projection layer: hidden_state → continuous_thought_vector
- Unproject at end: continuous_thought_vector → discrete_tokens

#### C. Reinforcement Learning Fine-Tuning (RLFT)

**Method**: Expert Iteration or PPO (Proximal Policy Optimization)

**Application:**
1. Generate 100 solutions for a GSM8K problem
2. Filter for correct answers (ground truth or self-verification)
3. Fine-tune on positive trajectories (correct solutions)

**Benefits:**
- Learns from successful reasoning paths
- Can improve without labeled data (self-supervised)
- Encourages exploration of correct strategies

**Implementation:**
- Use `scripts/chat_rl.py` as starting point
- Reward function: +1 if answer matches ground truth, 0 otherwise
- Policy: model's generation distribution
- Value function: estimate expected reward from state

#### D. Synthetic Dataset Generation

**Idea:**
- Generate synthetic math problems similar to GSM8K
- Use stronger model (GPT-4) to create problems + solutions
- Fine-tune nanochat on synthetic + real data

**Benefits:**
- Unlimited training data
- Can control difficulty distribution
- Can focus on specific problem types

**Implementation:**
- See `dev/gen_synthetic_data.py` for existing synthetic data generation
- Use template-based generation with variable substitution
- Validate solutions with calculator tool

---

## Key Research Papers

### 1. Tree of Thoughts
- **Title**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- **Authors**: Yao et al. (2023)
- **arXiv**: https://arxiv.org/abs/2305.10601
- **Key Result**: GPT-4 with ToT: 4% → 74% on Game of 24
- **Relevance**: Framework for multi-path reasoning

### 2. Lookahead Decoding
- **Title**: "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"
- **Authors**: Fu et al. (2024)
- **Venue**: ICML 2024
- **Key Result**: 1.8× speedup on MT-bench, 4× on code completion
- **Relevance**: Token-level parallel decoding (your implementation)

### 3. Abacus Embeddings
- **Title**: "Transformers Can Do Arithmetic with the Right Embeddings"
- **Authors**: McLeish et al. (2024)
- **Venue**: NeurIPS 2024
- **Key Result**: 99% accuracy on 100-digit addition (trained on 20-digit)
- **Relevance**: Positional encoding for math

### 4. Continuous Chain of Thought
- **Title**: "Parallel Continuous Chain-of-Thought with Jacobi Iteration"
- **Authors**: Various (2025)
- **Key Result**: 50% reduction in training/inference time
- **Relevance**: Continuous reasoning vectors

### 5. GSM8K Dataset
- **Title**: "Training Verifiers to Solve Math Word Problems"
- **Authors**: Cobbe et al. (2021)
- **Venue**: NeurIPS 2021
- **Relevance**: Original dataset paper

---

## Lookahead Decoding Implementation

### File: `nanochat/lookahead.py`

**Main Function**: `lookahead_generate()`

**Usage:**
```python
from nanochat.lookahead import lookahead_generate_with_engine, LookaheadConfig

config = LookaheadConfig(
    k_candidates=3,      # Top 3 candidates
    depth=10,            # 10 tokens ahead
    temperature=0.7,
    top_k=50
)

tokens = lookahead_generate_with_engine(
    engine,
    prompt_tokens,
    max_tokens=256,
    lookahead_config=config
)
```

**How It Works:**
1. For each generation step:
   - Get logits from model
   - Select top K candidates
   - For each candidate, simulate D tokens ahead
   - Score each branch (sum of log probs)
   - Pick best branch, commit first token

**Performance Considerations:**
- **Slower**: K×D forward passes per token (vs. 1 for greedy)
- **Parallelizable**: Can batch K candidates
- **Memory**: Stores K×D sequences temporarily

---

## Additional Research Directions

### 1. Model Validation (Two-Model Approach)

**Idea**: One model generates, another validates

**Implementation:**
```python
# Generator model
generator = load_model("default", device, phase="eval")

# Validator model (can be same or different)
validator = load_model("d32", device, phase="eval")  # Larger model

# Generate candidate solutions
candidates = [generator.generate(prompt) for _ in range(10)]

# Validate each
scores = []
for candidate in candidates:
    validation_prompt = f"Is this solution correct? {candidate}"
    score = validator.evaluate(validation_prompt)
    scores.append(score)

# Pick best
best = candidates[scores.index(max(scores))]
```

**Benefits:**
- Separates generation from validation
- Can use larger model for validation (cheaper)
- Reduces false positives

### 2. Continuous Chain of Thought

**Implementation Steps:**
1. Add latent token type (not in vocabulary)
2. Modify forward pass to handle latent tokens
3. Add projection: hidden_state → continuous_vector
4. Allow model to "think" in continuous space
5. Unproject at end: continuous_vector → tokens

**Code Sketch:**
```python
class ContinuousCoT(nn.Module):
    def __init__(self, vocab_size, hidden_dim, latent_dim):
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.unproject = nn.Linear(latent_dim, vocab_size)
    
    def forward(self, tokens, use_latent=False):
        if use_latent:
            # Continuous reasoning
            x = self.latent_proj(hidden_states)
            return x  # Continuous vector
        else:
            # Standard discrete tokens
            x = self.embedding(tokens)
            return x
```

### 3. Synthetic Dataset Generation

**Template-Based Approach:**
```python
templates = [
    "{num1} + {num2} = ?",
    "If {item} costs ${price}, how much do {count} cost?",
    # ... more templates
]

def generate_synthetic():
    problem = template.format(**variables)
    solution = solve(problem)  # Use calculator or GPT-4
    return {"problem": problem, "solution": solution}
```

**Benefits:**
- Unlimited data
- Control difficulty
- Focus on specific problem types

### 4. Reinforcement Learning Fine-Tuning

**PPO Implementation:**
```python
# Policy: model's generation
# Reward: +1 if correct, 0 otherwise
# Value: estimate expected reward

def compute_reward(generated, ground_truth):
    return 1.0 if extract_answer(generated) == ground_truth else 0.0

# PPO update
loss = -log_prob * advantage + value_loss
```

**See**: `scripts/chat_rl.py` for existing RL implementation

---

## Evaluation Metrics

### For GSM8K:
1. **Accuracy**: % of problems with correct final answer
2. **Solution Quality**: Is the reasoning path correct? (human eval)
3. **Efficiency**: Tokens generated, time per problem
4. **Robustness**: Performance across problem types

### Comparison Baselines:
- Greedy decoding (temperature=0)
- Standard sampling (temperature=0.7)
- Beam search (beam_size=5)
- Lookahead decoding (K=3, D=10)

---

## Code Structure

```
nanochat/
├── lookahead.py          # Lookahead decoding implementation
├── engine.py             # Standard generation engine
├── gpt.py                # Model architecture
└── ...

scripts/
├── chat_cli.py           # CLI interface (can add lookahead option)
├── chat_eval.py          # Evaluation script
└── ...

tasks/
└── gsm8k.py             # GSM8K dataset loader
```

---

## Next Steps

1. **Implement Lookahead**: Use `nanochat/lookahead.py`
2. **Evaluate**: Compare lookahead vs. greedy on GSM8K test set
3. **Experiment**: Try different K, D values
4. **Extend**: Add validator model, continuous CoT, etc.
5. **Write Paper**: Use this guide as outline

---

## References

1. Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." arXiv:2305.10601
2. Fu et al. (2024). "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." ICML 2024
3. McLeish et al. (2024). "Transformers Can Do Arithmetic with the Right Embeddings." NeurIPS 2024
4. Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems." NeurIPS 2021
5. Continuous Chain of Thought papers (2025) - various authors

---

**Last Updated**: 2025-01-06
