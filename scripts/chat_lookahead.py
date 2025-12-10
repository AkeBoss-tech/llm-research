"""
Chat interface with Lookahead Decoding for GSM8K math reasoning.

This script demonstrates how to use lookahead decoding (token-level tree search)
to improve math reasoning performance on GSM8K problems.

Usage:
    python -m scripts.chat_lookahead -i default --use-lookahead
    python -m scripts.chat_lookahead -i default --lookahead-k 5 --lookahead-depth 10
"""

import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from nanochat.lookahead import lookahead_generate_with_engine, LookaheadConfig

parser = argparse.ArgumentParser(description='Chat with lookahead decoding')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source: sft|mid|rl|default")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Single prompt mode')
parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'])
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])

# Lookahead-specific arguments
parser.add_argument('--use-lookahead', action='store_true', help='Enable lookahead decoding')
parser.add_argument('--lookahead-k', type=int, default=3, help='Number of candidates to explore')
parser.add_argument('--lookahead-depth', type=int, default=10, help='How many tokens to look ahead')
parser.add_argument('--lookahead-temp', type=float, default=0.7, help='Temperature for lookahead')

args = parser.parse_args()

# Initialize compute
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Load model
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

print("\n" + "=" * 60)
print("NanoChat with Lookahead Decoding")
print("=" * 60)
if args.use_lookahead:
    print(f"Lookahead: K={args.lookahead_k}, Depth={args.lookahead_depth}")
    print("This will be slower but more accurate for math problems.")
else:
    print("Standard generation (no lookahead)")
print("Type 'quit' to exit, 'clear' to reset conversation")
print("=" * 60 + "\n")

conversation_tokens = [bos]

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Add user message
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    # Generate response
    print("\nAssistant: ", end="", flush=True)
    response_tokens = []

    with autocast_ctx:
        if args.use_lookahead:
            # Use lookahead decoding
            lookahead_config = LookaheadConfig(
                k_candidates=args.lookahead_k,
                depth=args.lookahead_depth,
                temperature=args.lookahead_temp,
                top_k=args.top_k
            )
            
            # Get prompt tokens (everything up to assistant_start)
            prompt_tokens = conversation_tokens.copy()
            
            # Generate with lookahead
            result_tokens = lookahead_generate_with_engine(
                engine,
                prompt_tokens,
                max_tokens=256,
                lookahead_config=lookahead_config
            )
            
            # Extract only the new tokens (after prompt)
            response_tokens = result_tokens[len(prompt_tokens):]
            
            # Decode and print
            response_text = tokenizer.decode(response_tokens)
            print(response_text)
            
        else:
            # Standard generation
            generate_kwargs = {
                "num_samples": 1,
                "max_tokens": 256,
                "temperature": args.temperature,
                "top_k": args.top_k,
            }
            
            for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                token = token_column[0]
                response_tokens.append(token)
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)
            print()

    # Ensure assistant_end token
    if response_tokens and response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    
    conversation_tokens.extend(response_tokens)

    # Exit after single prompt
    if args.prompt:
        break
