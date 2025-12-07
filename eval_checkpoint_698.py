"""
Evaluate a specific checkpoint on GSM8K.

Usage:
    python eval_checkpoint_698.py --checkpoint-dir ~/llm-research/output/default_finetune --step 698
"""
import os
import json
import argparse
import torch
from datetime import datetime
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, print0, get_dist_info
from nanochat.engine import Engine
from nanochat.checkpoint_manager import build_model
from tasks.gsm8k import GSM8K

def evaluate_checkpoint(checkpoint_dir, step, device, autocast_ctx, max_problems=None, num_samples=1, temperature=0.0, top_k=50, max_new_tokens=512, output_file=None):
    """Load a checkpoint and evaluate it on GSM8K."""
    print0("=" * 80)
    print0(f"Evaluating checkpoint: {checkpoint_dir}")
    print0(f"Step: {step}")
    print0("=" * 80)
    
    # Load model from checkpoint
    print0(f"Loading model from checkpoint...")
    model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase="eval")
    
    print0(f"Model config: {meta.get('model_config', 'N/A')}")
    print0(f"Source model: {meta.get('source_model', 'N/A')}")
    print0(f"Checkpoint step: {meta.get('step', step)}")
    
    # Convert to bfloat16 for memory efficiency
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        model.to(dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        print0("Converted model to bfloat16")
    elif device.type == 'cuda':
        model.to(dtype=torch.float16)
        torch.cuda.empty_cache()
        print0("Converted model to float16")
    
    # Create engine
    engine = Engine(model, tokenizer)
    
    # Load GSM8K task
    task_object = GSM8K(subset="main", split="test")
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    
    print0(f"\nRunning GSM8K evaluation...")
    print0(f"Parameters: num_samples={num_samples}, temperature={temperature}, top_k={top_k}, max_new_tokens={max_new_tokens}")
    print0(f"Evaluating {num_problems} problems")
    
    # Run evaluation
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    num_passed, total = 0, 0
    detailed_results = []
    
    with autocast_ctx:
        for i in range(ddp_rank, num_problems, ddp_world_size):
            conversation = task_object[i]
            
            # Extract question and ground truth answer
            question = conversation['messages'][0]['content']
            assistant_message = conversation['messages'][-1]
            if isinstance(assistant_message['content'], list):
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
                'ground_truth_text': ground_truth_text[:500],
                'model_response': completions[0][:1000] if completions else "",
                'predicted_answer': predicted_answer,
                'correct': bool(passed),
                'all_completions': completions[:3],
                'all_outcomes': outcomes[:3],
            })
            
            # Keep stats
            total += 1
            num_passed += int(passed)
            
            # Logging
            if (i + 1) % 10 == 0 or i == num_problems - 1:
                print0(f"Progress: {i+1}/{num_problems} ({100*(i+1)/num_problems:.1f}%) | Correct: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
                
                # Save intermediate results
                if output_file and ddp_rank == 0:
                    intermediate_accuracy = num_passed / total if total > 0 else 0.0
                    intermediate_data = {
                        'checkpoint_dir': checkpoint_dir,
                        'step': step,
                        'accuracy': intermediate_accuracy,
                        'accuracy_percent': 100 * intermediate_accuracy,
                        'num_correct': num_passed,
                        'num_total': total,
                        'detailed_results': detailed_results,
                        'eval_params': {
                            'num_samples': num_samples,
                            'temperature': temperature,
                            'top_k': top_k,
                            'max_new_tokens': max_new_tokens,
                            'max_problems': max_problems,
                        },
                        'checkpoint_meta': meta,
                    }
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print0(f"Warning: Failed to save intermediate results: {e}")
    
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
        detailed_results = [item for sublist in all_detailed_results for item in sublist]
        detailed_results.sort(key=lambda x: x['problem_id'])
    
    accuracy = num_passed / total if total > 0 else 0.0
    
    print0(f"\n{'=' * 80}")
    print0(f"Checkpoint: {checkpoint_dir} (step {step})")
    print0(f"GSM8K Accuracy: {100 * accuracy:.2f}% ({num_passed}/{total})")
    print0(f"{'=' * 80}\n")
    
    # Clean up
    del model
    del tokenizer
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'checkpoint_dir': checkpoint_dir,
        'step': step,
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
        },
        'checkpoint_meta': meta,
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint on GSM8K')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing the checkpoint (e.g., ~/llm-research/output/default_finetune)')
    parser.add_argument('--step', type=int, required=True,
                        help='Checkpoint step to evaluate (e.g., 698)')
    parser.add_argument('--max-problems', type=int, default=None,
                        help='Maximum number of problems to evaluate (default: all)')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples per problem (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for generation (default: 0.0)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter (default: 50)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Maximum new tokens to generate (default: 512)')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'],
                        help='Device type: cuda|cpu|mps (empty => autodetect)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                        help='Data type (default: bfloat16)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: same as checkpoint-dir)')
    args = parser.parse_args()
    
    # Expand user path
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        print0(f"ERROR: Checkpoint directory does not exist: {checkpoint_dir}")
        return 1
    
    # Setup device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    print0("=" * 80)
    print0("GSM8K Checkpoint Evaluation")
    print0("=" * 80)
    print0(f"Checkpoint directory: {checkpoint_dir}")
    print0(f"Step: {args.step}")
    print0(f"Device: {device_type}")
    print0(f"Max problems: {args.max_problems if args.max_problems else 'all'}")
    print0("=" * 80)
    print0()
    
    # Setup output directory
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    else:
        output_dir = checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_step_{args.step:06d}_{timestamp}.json")
    
    # Evaluate
    try:
        result = evaluate_checkpoint(
            checkpoint_dir,
            args.step,
            device,
            autocast_ctx,
            max_problems=args.max_problems,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            output_file=output_file,
        )
        
        # Save final results
        if ddp_rank == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print0(f"\nResults saved to: {output_file}")
            print0(f"Accuracy: {result['accuracy_percent']:.2f}% ({result['num_correct']}/{result['num_total']})")
            
    except Exception as e:
        print0(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    compute_cleanup()
    return 0

if __name__ == "__main__":
    exit(main())
