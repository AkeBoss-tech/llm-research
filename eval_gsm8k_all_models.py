"""
Evaluate GSM8K task on all available models from HuggingFace.

Usage:
    python eval_gsm8k_all_models.py
    python eval_gsm8k_all_models.py --max-problems 100  # Limit to first 100 problems
    python eval_gsm8k_all_models.py --models d32 d34    # Only test specific models
"""
import os
import json
import argparse
import torch
import gc
from datetime import datetime
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0, get_dist_info
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from tasks.gsm8k import GSM8K

# Model configurations
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

def evaluate_model_detailed(model_name, config, device, autocast_ctx, max_problems=None, num_samples=1, temperature=0.0, top_k=50, max_new_tokens=512):
    """Load a model and evaluate it on GSM8K with detailed results."""
    print0("=" * 80)
    print0(f"Evaluating model: {model_name}")
    print0(f"Repository: {config['hf_repo_id']}")
    print0(f"Step: {config['step']}")
    print0("=" * 80)
    
    # Load model
    print0(f"Loading model from HuggingFace...")
    model, tokenizer, meta = load_model(
        config['hf_repo_id'],
        device=device,
        phase="eval",
        step=config['step'],
        model_tag=config['model_tag'],
        use_standard_structure=config['use_standard'],
        checkpoint_type=config['checkpoint_type']
    )
    
    # Print model info
    base_dir = get_base_dir()
    if config['use_standard'] and config['model_tag']:
        checkpoint_dir_map = {
            "base": "base_checkpoints",
            "mid": "mid_checkpoints",
            "sft": "chatsft_checkpoints",
            "rl": "chatrl_checkpoints",
        }
        checkpoint_dir_name = checkpoint_dir_map.get(config['checkpoint_type'], "base_checkpoints")
        checkpoint_dir = os.path.join(base_dir, checkpoint_dir_name, config['model_tag'])
    else:
        cache_dir = os.path.join(base_dir, "hf_checkpoints", config['hf_repo_id'].replace("/", "_"))
        checkpoint_dir = os.path.join(cache_dir, f"step_{config['step']:06d}")
    
    print0(f"Model checkpoint directory: {checkpoint_dir}")
    print0(f"Model config: {meta.get('model_config', 'N/A')}")
    
    # Force model to float16 (or bfloat16) to save memory if requested
    # This is critical because checkpoints might be in float32
    # We access the global ARGS to check dtype
    if 'ARGS' in globals() and (device.type == 'cuda' or device.type == 'mps'):
        target_dtype = None
        if ARGS.dtype == 'bfloat16':
            # Check if bfloat16 is supported (Ampere+ or MPS)
            # Note: MPS support for bfloat16 is experimental/limited in some versions, but we allow it if requested
            if device.type == 'cuda' and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
                print0("Converting model to bfloat16...")
            elif device.type == 'mps':
                 # MPS technically supports bf16 in newer macOS/torch, but float16 is safer default
                 target_dtype = torch.bfloat16
                 print0("Converting model to bfloat16 (MPS)...")
            else:
                print0("Warning: bfloat16 requested but not supported on this GPU. Falling back to float16.")
                target_dtype = torch.float16
        elif ARGS.dtype == 'float16':
            target_dtype = torch.float16
            print0(f"Converting model to float16 ({device.type})...")
            
        if target_dtype is not None:
            model.to(dtype=target_dtype)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    # Create engine
    engine = Engine(model, tokenizer)
    
    # Load GSM8K task
    task_object = GSM8K(subset="main", split="test")
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    
    print0(f"\nRunning GSM8K evaluation...")
    print0(f"Parameters: num_samples={num_samples}, temperature={temperature}, top_k={top_k}, max_new_tokens={max_new_tokens}")
    print0(f"Evaluating {num_problems} problems")
    
    # Run evaluation with detailed tracking
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    num_passed, total = 0, 0
    detailed_results = []
    
    with autocast_ctx:
        for i in range(ddp_rank, num_problems, ddp_world_size):
            conversation = task_object[i]
            
            # Extract question and ground truth answer
            question = conversation['messages'][0]['content']
            assistant_message = conversation['messages'][-1]
            # Extract ground truth answer from the last text part
            if isinstance(assistant_message['content'], list):
                # Find the last text part which contains the answer
                ground_truth_text = ""
                for part in reversed(assistant_message['content']):
                    if part.get('type') == 'text':
                        ground_truth_text = part['text']
                        break
            else:
                ground_truth_text = str(assistant_message['content'])
            
            # Extract the numerical answer
            from tasks.gsm8k import extract_answer
            ground_truth_answer = extract_answer(ground_truth_text)
            
            # Tokenize the prompt
            encoded_prompt = tokenizer.render_for_completion(conversation)
            # Get the completions
            results, _ = engine.generate_batch(
                encoded_prompt,
                num_samples=num_samples,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            # Decode the completions as text
            prefix_length = len(encoded_prompt)
            completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
            
            # Evaluate success criteria
            outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
            passed = any(outcomes)
            
            # Extract predicted answer from first completion
            predicted_answer = extract_answer(completions[0]) if completions else None
            
            # Store detailed result
            detailed_results.append({
                'problem_id': i,
                'question': question,
                'ground_truth_answer': ground_truth_answer,
                'ground_truth_text': ground_truth_text[:500],  # Truncate for storage
                'model_response': completions[0][:1000] if completions else "",  # Truncate for storage
                'predicted_answer': predicted_answer,
                'correct': bool(passed),
                'all_completions': completions[:3],  # Store up to 3 completions
                'all_outcomes': outcomes[:3],
            })
            
            # Keep stats
            total += 1
            num_passed += int(passed)
            
            # Logging
            if (i + 1) % 10 == 0 or i == num_problems - 1:
                print0(f"Progress: {i+1}/{num_problems} ({100*(i+1)/num_problems:.1f}%) | Correct: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
    
    # Aggregate results across ranks if using DDP
    if ddp:
        import torch.distributed as dist
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()
        
        # Gather detailed results from all ranks
        all_detailed_results = [None for _ in range(ddp_world_size)]
        dist.all_gather_object(all_detailed_results, detailed_results)
        # Flatten list of lists
        detailed_results = [item for sublist in all_detailed_results for item in sublist]
        # Sort by problem_id to keep them in order
        detailed_results.sort(key=lambda x: x['problem_id'])
    
    accuracy = num_passed / total if total > 0 else 0.0
    
    print0(f"\n{'=' * 80}")
    print0(f"Model: {model_name}")
    print0(f"GSM8K Accuracy: {100 * accuracy:.2f}% ({num_passed}/{total})")
    print0(f"{'=' * 80}\n")
    
    # Explicitly clean up model resources
    del model
    del tokenizer
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'model_name': model_name,
        'hf_repo_id': config['hf_repo_id'],
        'step': config['step'],
        'accuracy': accuracy,
        'accuracy_percent': 100 * accuracy,
        'num_correct': num_passed,
        'num_total': total,
        'detailed_results': detailed_results,
        'eval_params': {
            'num_samples': num_samples,
            'temperature': temperature,
            'top_k': top_k,
            'max_new_tokens': max_new_tokens,
            'max_problems': max_problems,
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate GSM8K on all models from HuggingFace')
    parser.add_argument('--models', nargs='+', choices=['default', 'd32', 'd34'], default=['default', 'd32', 'd34'],
                        help='Models to evaluate (default: all)')
    parser.add_argument('--max-problems', type=int, default=None,
                        help='Maximum number of problems to evaluate per model (default: all)')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples per problem (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for generation (default: 0.0)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter (default: 50)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Maximum new tokens to generate (default: 512)')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'],
                        help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                        help='Data type for evaluation (default: bfloat16)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: ~/.cache/nanochat/gsm8k_eval/)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')
    args = parser.parse_args()
    
    # Pass args to evaluate_model_detailed so it can access args.dtype
    global ARGS
    ARGS = args
    
    # Setup device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    print0("=" * 80)
    print0("GSM8K Evaluation on All Models")
    print0("=" * 80)
    print0(f"Device: {device_type}")
    print0(f"Models to evaluate: {', '.join(args.models)}")
    print0(f"Max problems per model: {args.max_problems if args.max_problems else 'all'}")
    print0("=" * 80)
    print0()
    
    # Evaluate each model
    results = []
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print0(f"Warning: Unknown model '{model_name}', skipping...")
            continue
        
        try:
            result = evaluate_model_detailed(
                model_name,
                MODEL_CONFIGS[model_name],
                device,
                autocast_ctx,
                max_problems=args.max_problems,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(result)
        except Exception as e:
            print0(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup memory after each model
        print0(f"Cleaning up memory after {model_name}...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print0(f"Memory stats: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated, {torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")
            
    # Print summary
    print0("\n" + "=" * 80)
    print0("SUMMARY")
    print0("=" * 80)
    print0(f"{'Model':<15} {'Repository':<30} {'Step':<10} {'Accuracy':<15}")
    print0("-" * 80)
    for result in results:
        print0(f"{result['model_name']:<15} {result['hf_repo_id']:<30} {result['step']:<10} {result['accuracy_percent']:>6.2f}%")
    print0("=" * 80)
    
    # Find best model
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        print0(f"\nBest model: {best_result['model_name']} with {best_result['accuracy_percent']:.2f}% accuracy")
    
    # Save results to file
    if not args.no_save:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(get_base_dir(), "gsm8k_eval")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"gsm8k_results_{timestamp}.json")
        
        # Prepare output data
        output_data = {
            'timestamp': timestamp,
            'eval_params': {
                'models': args.models,
                'max_problems': args.max_problems,
                'num_samples': args.num_samples,
                'temperature': args.temperature,
                'top_k': args.top_k,
                'max_new_tokens': args.max_new_tokens,
                'device_type': device_type,
            },
            'summary': [
                {
                    'model_name': r['model_name'],
                    'hf_repo_id': r['hf_repo_id'],
                    'step': r['step'],
                    'accuracy': r['accuracy'],
                    'accuracy_percent': r['accuracy_percent'],
                    'num_correct': r.get('num_correct', 0),
                    'num_total': r.get('num_total', 0),
                }
                for r in results
            ],
            'detailed_results': {
                r['model_name']: {
                    'summary': {
                        'accuracy': r['accuracy'],
                        'num_correct': r.get('num_correct', 0),
                        'num_total': r.get('num_total', 0),
                    },
                    'problems': r.get('detailed_results', [])
                }
                for r in results
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print0(f"\nResults saved to: {output_file}")
        print0(f"To visualize, run: python visualize_gsm8k_results.py {output_file}")
    
    compute_cleanup()

if __name__ == "__main__":
    main()

