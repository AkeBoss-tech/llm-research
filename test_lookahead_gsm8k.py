"""
Test Lookahead Decoding on 10 GSM8K Problems

This script compares standard generation vs. lookahead decoding on 10 example
GSM8K problems to demonstrate the improvement.

Usage:
    python test_lookahead_gsm8k.py -i default
    python test_lookahead_gsm8k.py -i default --lookahead-k 5 --lookahead-depth 10
"""

import argparse
import time
import json
import torch
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.lookahead import lookahead_generate_with_engine, LookaheadConfig
from tasks.gsm8k import GSM8K, extract_answer

def format_problem(conversation):
    """Extract the question from a GSM8K conversation."""
    messages = conversation['messages']
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content']
    return "Unknown question"

def get_ground_truth(conversation):
    """Extract the ground truth answer from a GSM8K conversation."""
    messages = conversation['messages']
    for msg in messages:
        if msg['role'] == 'assistant':
            if isinstance(msg['content'], list):
                # Find the last text part which contains the answer
                for part in reversed(msg['content']):
                    if part.get('type') == 'text':
                        return extract_answer(part['text'])
    return None

def generate_standard(engine, tokenizer, conversation, max_tokens=512, temperature=0.7, top_k=50):
    """Generate using standard (greedy/sampling) method."""
    encoded_prompt = tokenizer.render_for_completion(conversation)
    
    start_time = time.time()
    results, _ = engine.generate_batch(
        encoded_prompt,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    elapsed = time.time() - start_time
    
    prefix_length = len(encoded_prompt)
    completion = tokenizer.decode(results[0][prefix_length:])
    return completion, elapsed

def generate_lookahead(engine, tokenizer, conversation, lookahead_config, max_tokens=512):
    """Generate using lookahead decoding."""
    encoded_prompt = tokenizer.render_for_completion(conversation)
    
    # Set tokenizer in config for verbose output
    lookahead_config.tokenizer = tokenizer
    lookahead_config.verbose = True  # Enable progress printing
    
    start_time = time.time()
    result_tokens, tree_data = lookahead_generate_with_engine(
        engine,
        encoded_prompt,
        max_tokens=max_tokens,
        lookahead_config=lookahead_config
    )
    elapsed = time.time() - start_time
    
    prefix_length = len(encoded_prompt)
    completion = tokenizer.decode(result_tokens[prefix_length:])
    return completion, elapsed, tree_data

def main():
    parser = argparse.ArgumentParser(description='Test lookahead decoding on GSM8K')
    parser.add_argument('-i', '--source', type=str, default='default', 
                       help='Model source: default|d32|d34|sft|mid|rl')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('--device-type', type=str, default='', 
                       choices=['cuda', 'cpu', 'mps'], help='Device type')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', 
                       choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.7, 
                       help='Temperature for standard generation')
    parser.add_argument('-k', '--top-k', type=int, default=50, 
                       help='Top-k for standard generation')
    parser.add_argument('--lookahead-k', type=int, default=3, 
                       help='Number of candidates for lookahead')
    parser.add_argument('--lookahead-depth', type=int, default=10, 
                       help='Depth for lookahead (tokens to look ahead)')
    parser.add_argument('--lookahead-temp', type=float, default=0.7, 
                       help='Temperature for lookahead')
    parser.add_argument('-n', '--num-problems', type=int, default=10, 
                       help='Number of problems to test')
    parser.add_argument('--max-tokens', type=int, default=512, 
                       help='Max tokens to generate')
    args = parser.parse_args()

    # Initialize compute
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # Model configurations for HuggingFace models
    MODEL_CONFIGS = {
        'default': {
            'hf_repo_id': 'sdobson/nanochat',
            'step': 650,
            'model_tag': None,
            'use_standard': False,
            'checkpoint_type': 'base',
        },
        'd32': {
            'hf_repo_id': 'karpathy/nanochat-d32',
            'step': 650,
            'model_tag': 'd32',
            'use_standard': True,
            'checkpoint_type': 'sft',
        },
        'd34': {
            'hf_repo_id': 'karpathy/nanochat-d34',
            'step': 169150,
            'model_tag': 'd34',
            'use_standard': True,
            'checkpoint_type': 'sft',
        },
    }
    
    # Load model
    print("=" * 80)
    print(f"Loading model: {args.source}")
    print("=" * 80)
    
    # Check if source is a predefined model config
    if args.source in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.source]
        model, tokenizer, meta = load_model(
            config['hf_repo_id'],
            device=device,
            phase="eval",
            step=args.step or config['step'],
            model_tag=args.model_tag or config['model_tag'],
            use_standard_structure=config['use_standard'],
            checkpoint_type=config['checkpoint_type']
        )
    else:
        # Use source directly (for "base", "mid", "sft", "rl", or HuggingFace repo ID)
        model, tokenizer, meta = load_model(args.source, device, phase="eval", 
                                            model_tag=args.model_tag, step=args.step)
    
    # Convert to appropriate dtype
    if device_type in ['cuda', 'mps'] and args.dtype != 'float32':
        target_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
        if device_type == 'cuda' and args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported, using float16")
            target_dtype = torch.float16
        model.to(dtype=target_dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if device_type == 'mps':
            torch.mps.empty_cache()
    
    engine = Engine(model, tokenizer)

    # Load GSM8K test set
    print("\n" + "=" * 80)
    print("Loading GSM8K test set")
    print("=" * 80)
    task = GSM8K(subset="main", split="test")
    num_problems = min(args.num_problems, len(task))
    print(f"Testing on {num_problems} problems\n")

    # Configure lookahead
    lookahead_config = LookaheadConfig(
        k_candidates=args.lookahead_k,
        depth=args.lookahead_depth,
        temperature=args.lookahead_temp,
        top_k=args.top_k
    )

    # Results tracking
    results = {
        'standard': {'correct': 0, 'total': 0, 'time': 0.0},
        'lookahead': {'correct': 0, 'total': 0, 'time': 0.0}
    }

    print("=" * 80)
    print("Running Evaluation")
    print("=" * 80)
    print(f"Standard: temperature={args.temperature}, top_k={args.top_k}")
    print(f"Lookahead: k={args.lookahead_k}, depth={args.lookahead_depth}, temp={args.lookahead_temp}")
    print("=" * 80 + "\n")

    # Test each problem
    for i in range(num_problems):
        conversation = task.get_example(i)
        question = format_problem(conversation)
        ground_truth = get_ground_truth(conversation)

        print(f"\n{'='*80}")
        print(f"Problem {i+1}/{num_problems}")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")

        # Standard generation
        print("\n--- Standard Generation ---")
        with autocast_ctx:
            std_completion, std_time = generate_standard(
                engine, tokenizer, conversation,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
        std_answer = extract_answer(std_completion)
        std_correct = (std_answer == ground_truth) if std_answer else False
        
        print(f"Time: {std_time:.2f}s")
        print(f"Answer: {std_answer if std_answer else 'NOT FOUND'}")
        print(f"Correct: {'✓' if std_correct else '✗'}")
        print(f"Response:\n{std_completion[:500]}...")  # First 500 chars
        
        results['standard']['total'] += 1
        results['standard']['correct'] += int(std_correct)
        results['standard']['time'] += std_time

        # Lookahead generation
        print("\n--- Lookahead Decoding ---")
        with autocast_ctx:
            la_completion, la_time, tree_data = generate_lookahead(
                engine, tokenizer, conversation,
                lookahead_config,
                max_tokens=args.max_tokens
            )
            
            # Save tree data for visualization
            tree_data['problem'] = {
                'question': question,
                'ground_truth': ground_truth,
                'completion': la_completion,
                'answer': extract_answer(la_completion),
            }
            tree_file = f"tree_data_problem_{i+1}.json"
            with open(tree_file, 'w') as f:
                json.dump(tree_data, f, indent=2)
            print(f"  Tree data saved to: {tree_file}")
            print(f"  Open visualize_tree_of_thought.html and load {tree_file} to view the tree!")
        la_answer = extract_answer(la_completion)
        la_correct = (la_answer == ground_truth) if la_answer else False
        
        print(f"Time: {la_time:.2f}s")
        print(f"Answer: {la_answer if la_answer else 'NOT FOUND'}")
        print(f"Correct: {'✓' if la_correct else '✗'}")
        print(f"Response:\n{la_completion[:500]}...")  # First 500 chars
        
        results['lookahead']['total'] += 1
        results['lookahead']['correct'] += int(la_correct)
        results['lookahead']['time'] += la_time

        # Comparison
        print(f"\n--- Comparison ---")
        improvement = "Improved" if la_correct and not std_correct else \
                     "Same" if la_correct == std_correct else \
                     "Worse" if not la_correct and std_correct else "Both wrong"
        speedup = std_time / la_time if la_time > 0 else 0
        print(f"Result: {improvement}")
        print(f"Speed: {speedup:.2f}x {'slower' if speedup < 1 else 'faster'}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    std_acc = results['standard']['correct'] / results['standard']['total'] * 100
    la_acc = results['lookahead']['correct'] / results['lookahead']['total'] * 100
    std_avg_time = results['standard']['time'] / results['standard']['total']
    la_avg_time = results['lookahead']['time'] / results['lookahead']['total']
    
    print(f"\nStandard Generation:")
    print(f"  Accuracy: {results['standard']['correct']}/{results['standard']['total']} ({std_acc:.1f}%)")
    print(f"  Avg Time: {std_avg_time:.2f}s per problem")
    
    print(f"\nLookahead Decoding:")
    print(f"  Accuracy: {results['lookahead']['correct']}/{results['lookahead']['total']} ({la_acc:.1f}%)")
    print(f"  Avg Time: {la_avg_time:.2f}s per problem")
    
    print(f"\nImprovement:")
    print(f"  Accuracy: {la_acc - std_acc:+.1f}% ({la_acc - std_acc:.1f} percentage points)")
    print(f"  Speed: {std_avg_time / la_avg_time:.2f}x slower")
    
    if la_acc > std_acc:
        print(f"\n✓ Lookahead decoding improved accuracy by {la_acc - std_acc:.1f}%!")
    elif la_acc == std_acc:
        print(f"\n→ Lookahead decoding had same accuracy as standard.")
    else:
        print(f"\n✗ Lookahead decoding had lower accuracy (may need tuning).")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
