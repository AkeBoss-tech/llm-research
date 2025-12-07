"""
Finetune models on GSM8K dataset.

This script can finetune models loaded from HuggingFace (default, d32, d34) 
or from local checkpoints on the GSM8K training dataset.

Usage:
    # Finetune default model from HuggingFace
    python finetune_gsm8k.py --model default

    # Finetune d32 model from HuggingFace
    python finetune_gsm8k.py --model d32

    # Finetune d34 model from HuggingFace
    python finetune_gsm8k.py --model d34

    # Finetune with custom settings
    python finetune_gsm8k.py --model d32 --num-epochs 3 --device-batch-size 2

    # Multi-GPU training (single node)
    torchrun --standalone --nproc_per_node=8 finetune_gsm8k.py --model d32

    # For MPS (Apple Silicon) with memory issues, try:
    # 1. Reduce batch size: --device-batch-size 1 --target-examples-per-step 4
    # 2. Set environment variable: export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    #    (Warning: may cause system instability if memory is exhausted)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

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

# -----------------------------------------------------------------------------
# Training Hyperparameters
parser = argparse.ArgumentParser(description='Finetune models on GSM8K')
parser.add_argument('--model', type=str, choices=['default', 'd32', 'd34'], required=True,
                    help='Model to finetune: default, d32, or d34')
parser.add_argument('--source', type=str, default=None,
                    help='Override: Load from local checkpoint (base|mid|sft|rl) or HuggingFace repo ID')
parser.add_argument('--model-tag', type=str, default=None,
                    help='Override: Model tag for local checkpoints')
parser.add_argument('--step', type=int, default=None,
                    help='Override: Step to load from local checkpoint')
parser.add_argument('--run', type=str, default='gsm8k-finetune',
                    help='Wandb run name (default: gsm8k-finetune)')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'],
                    help='Device type: cuda|cpu|mps (empty => autodetect)')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                    help='Data type (default: bfloat16)')
parser.add_argument('--device-batch-size', type=int, default=None,
                    help='Device batch size (default: 1 for MPS, 4 for CUDA)')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='Number of epochs (default: 3)')
parser.add_argument('--target-examples-per-step', type=int, default=None,
                    help='Target examples per step (default: 8 for MPS, 32 for CUDA)')
parser.add_argument('--unembedding-lr', type=float, default=0.004,
                    help='Learning rate for unembedding layer (default: 0.004)')
parser.add_argument('--embedding-lr', type=float, default=0.2,
                    help='Learning rate for embedding layer (default: 0.2)')
parser.add_argument('--matrix-lr', type=float, default=0.02,
                    help='Learning rate for matrix layers (default: 0.02)')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay (default: 0.0)')
parser.add_argument('--init-lr-frac', type=float, default=0.02,
                    help='Initial learning rate fraction (default: 0.02)')
parser.add_argument('--eval-every', type=int, default=100,
                    help='Evaluate validation loss every N steps (default: 100)')
parser.add_argument('--eval-steps', type=int, default=100,
                    help='Number of validation steps (default: 100)')
parser.add_argument('--save-every', type=int, default=200,
                    help='Save checkpoint every N steps (default: 200)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Directory to save checkpoints (default: ~/.cache/nanochat/gsm8k_finetune/)')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Set default batch size based on device type if not specified
if args.device_batch_size is None:
    args.device_batch_size = 1 if device_type == 'mps' else 4
    print0(f"Auto-set device_batch_size to {args.device_batch_size} for {device_type}")

# Set default target examples per step based on device type if not specified
if args.target_examples_per_step is None:
    args.target_examples_per_step = 8 if device_type == 'mps' else 32
    print0(f"Auto-set target_examples_per_step to {args.target_examples_per_step} for {device_type}")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-gsm8k-finetune", 
    name=args.run, 
    config=vars(args),
    save_code=True
)

# -----------------------------------------------------------------------------
# Load the model
print0("=" * 80)
print0(f"Loading model: {args.model}")
print0("=" * 80)

if args.source:
    # Load from custom source (local checkpoint or HuggingFace repo)
    if args.source in ["base", "mid", "sft", "rl"]:
        # Local checkpoint
        model, tokenizer, meta = load_model(
            args.source,
            device=device,
            phase="train",
            model_tag=args.model_tag,
            step=args.step
        )
    else:
        # HuggingFace repo ID
        config = MODEL_CONFIGS.get(args.model, {})
        model, tokenizer, meta = load_model(
            args.source,
            device=device,
            phase="train",
            step=args.step or config.get('step'),
            model_tag=args.model_tag or config.get('model_tag'),
            use_standard_structure=config.get('use_standard', False),
            checkpoint_type=config.get('checkpoint_type', 'base')
        )
else:
    # Load from predefined model config
    config = MODEL_CONFIGS[args.model]
    model, tokenizer, meta = load_model(
        config['hf_repo_id'],
        device=device,
        phase="train",
        step=config['step'],
        model_tag=config['model_tag'],
        use_standard_structure=config['use_standard'],
        checkpoint_type=config['checkpoint_type']
    )

print0(f"Model loaded successfully!")
print0(f"Model config: {meta.get('model_config', 'N/A')}")

# Force model to target dtype if needed
if device_type in ['cuda', 'mps'] and args.dtype != 'float32':
    target_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    if device_type == 'cuda' and args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        print0("Warning: bfloat16 not supported, using float16")
        target_dtype = torch.float16
    model.to(dtype=target_dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device_type == 'mps':
        torch.mps.empty_cache()

engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# GSM8K dataset
print0("=" * 80)
print0("Loading GSM8K dataset")
print0("=" * 80)

train_ds = GSM8K(subset="main", split="train")
val_ds = GSM8K(subset="main", split="test")

print0(f"Training examples: {len(train_ds)}")
print0(f"Validation examples: {len(val_ds)}")

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets
    
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = args.device_batch_size * ddp_world_size
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Examples per step: {examples_per_step}")
assert args.target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"Gradient accumulation steps: {grad_accum_steps}")

num_iterations = (len(train_ds) // args.target_examples_per_step) * args.num_epochs
print0(f"Total iterations: {num_iterations} ({args.num_epochs} epochs)")

train_loader = sft_data_generator(train_ds, batch_size=args.device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size)

# -----------------------------------------------------------------------------
# Initialize Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Setup output directory
if args.output_dir:
    output_dir = args.output_dir
else:
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "gsm8k_finetune", args.model)
os.makedirs(output_dir, exist_ok=True)
print0(f"Checkpoints will be saved to: {output_dir}")

# -----------------------------------------------------------------------------
# Training loop

def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

step = 0
train_iter = iter(train_loader)

print0("=" * 80)
print0("Starting training")
print0("=" * 80)

for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # Validation loss
    if last_step or step % args.eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(args.eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()
        
        # Clear MPS cache after validation
        if device_type == 'mps':
            torch.mps.empty_cache()

    # Save checkpoint
    if last_step or (step > 0 and step % args.save_every == 0):
        if master_process:
            model_config_kwargs = model.config.__dict__
            save_checkpoint(
                output_dir,
                step,
                model.state_dict(),
                None,  # Don't save optimizer state
                {
                    "step": step,
                    "val_loss": val_loss if 'val_loss' in locals() else None,
                    "model_config": model_config_kwargs,
                    "source_model": args.model,
                    "num_epochs": args.num_epochs,
                }
            )
            print0(f"✅ Saved checkpoint at step {step}")

    if last_step:
        break

    # Training step
    num_tokens = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum()
        
        # Clear MPS cache after each micro-step to reduce memory pressure
        if device_type == 'mps':
            torch.mps.empty_cache()
    
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    # Learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Optimizer step
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    # Clear MPS cache after optimizer step
    if device_type == 'mps':
        torch.mps.empty_cache()

    # Logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Train loss: {train_loss_item:.6f} | LR: {lrm:.6f} | Tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })

# Final save
if master_process:
    model_config_kwargs = model.config.__dict__
    save_checkpoint(
        output_dir,
        step,
        model.state_dict(),
        None,
        {
            "step": step,
            "val_loss": val_loss if 'val_loss' in locals() else None,
            "model_config": model_config_kwargs,
            "source_model": args.model,
            "num_epochs": args.num_epochs,
        }
    )
    print0(f"✅ Final checkpoint saved at step {step}")

wandb_run.finish()
compute_cleanup()

print0("=" * 80)
print0("Training complete!")
print0(f"Checkpoints saved to: {output_dir}")
print0("=" * 80)
