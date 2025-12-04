"""
Test script to load nanochat model from HuggingFace and answer a simple question.

Usage:
    python test_hf_model.py                    # Uses default model (sdobson/nanochat)
    python test_hf_model.py --model d32       # Uses karpathy/nanochat-d32
    python test_hf_model.py --model d34       # Uses karpathy/nanochat-d34
    python test_hf_model.py --repo karpathy/nanochat-d32 --step 650 --model-tag d32 --use-standard
"""
import os
import argparse
import torch
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

def main():
    parser = argparse.ArgumentParser(description='Test loading nanochat model from HuggingFace')
    parser.add_argument('--repo', type=str, default=None, help='HuggingFace repo ID (e.g., sdobson/nanochat, karpathy/nanochat-d32, or karpathy/nanochat-d34)')
    parser.add_argument('--step', type=int, default=None, help='Checkpoint step (e.g., 650 or 169150)')
    parser.add_argument('--model', type=str, default=None, choices=['d32', 'd34', 'default'], help='Model preset: d32, d34, or default')
    parser.add_argument('--model-tag', type=str, default=None, help='Model tag for standard structure (e.g., d32, d34)')
    parser.add_argument('--use-standard', action='store_true', help='Use standard directory structure')
    parser.add_argument('--checkpoint-type', type=str, default='base', choices=['base', 'mid', 'sft', 'rl'], 
                        help='Checkpoint type for standard structure (default: base, but d32/d34 use sft)')
    args = parser.parse_args()
    
    # Setup device
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    
    # Determine model configuration based on presets or arguments
    if args.model == 'd32':
        hf_repo_id = "karpathy/nanochat-d32"
        step = 650
        model_tag = "d32"
        use_standard = True
        checkpoint_type = "sft"  # Following same pattern as d34
    elif args.model == 'd34':
        hf_repo_id = "karpathy/nanochat-d34"
        step = 169150
        model_tag = "d34"
        use_standard = True
        checkpoint_type = "sft"  # User specified chatsft_checkpoints
    elif args.repo:
        hf_repo_id = args.repo
        step = args.step
        model_tag = args.model_tag
        use_standard = args.use_standard
        checkpoint_type = args.checkpoint_type
    else:
        # Default model
        hf_repo_id = "sdobson/nanochat"
        step = 650
        model_tag = None
        use_standard = False
        checkpoint_type = "base"
    
    print(f"Loading model from HuggingFace: {hf_repo_id}")
    if step:
        print(f"Step: {step}")
    if use_standard:
        print(f"Using standard structure: {checkpoint_type}/{model_tag}")
    print(f"Device: {device_type}")
    print("-" * 50)
    
    # Load model from HuggingFace
    model, tokenizer, meta = load_model(
        hf_repo_id, 
        device=device, 
        phase="eval", 
        step=step,
        model_tag=model_tag,
        use_standard_structure=use_standard,
        checkpoint_type=checkpoint_type
    )
    
    # Compute and print the model storage path
    base_dir = get_base_dir()
    if use_standard and model_tag:
        checkpoint_dir_map = {
            "base": "base_checkpoints",
            "mid": "mid_checkpoints",
            "sft": "chatsft_checkpoints",
            "rl": "chatrl_checkpoints",
        }
        checkpoint_dir_name = checkpoint_dir_map.get(checkpoint_type, "base_checkpoints")
        checkpoint_dir = os.path.join(base_dir, checkpoint_dir_name, model_tag)
    else:
        cache_dir = os.path.join(base_dir, "hf_checkpoints", hf_repo_id.replace("/", "_"))
        checkpoint_dir = os.path.join(cache_dir, f"step_{step:06d}" if step else "step_unknown")
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    
    print(f"Model loaded successfully!")
    print(f"Model checkpoint directory: {checkpoint_dir}")
    print(f"Tokenizer directory: {tokenizer_dir}")
    print(f"Model config: {meta.get('model_config', 'N/A')}")
    print("-" * 50)
    
    # Create Engine for generation
    engine = Engine(model, tokenizer)
    
    # Get special tokens for chat format
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    
    # Format the question in chat format
    question = "What is 2+2?"
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Build conversation tokens
    conversation_tokens = [bos]
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(question))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)
    
    # Generate response
    print("\nAnswer: ", end="", flush=True)
    response_tokens = []
    
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": 0.6,
        "top_k": 50,
    }
    
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]  # Get the token from batch dimension
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
            
            # Stop if we hit the assistant_end token
            if token == assistant_end:
                break
    
    print("\n" + "-" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()

