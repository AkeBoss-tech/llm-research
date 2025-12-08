"""
Script to load all models (default, d32, d34), update their tokenizers, and test them.

This script assumes you already have the model weights locally. It will:
1. Load each model from local checkpoints
2. Ensure tokenizers are properly set up in each model's directory
3. Test each model with a simple question to verify they work

Usage:
    python test_all_models.py
    python test_all_models.py --skip-tokenizer-update  # Skip tokenizer updates, just test
    python test_all_models.py --models default d32     # Only test specific models
"""
import os
import shutil
import argparse
import torch
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model, load_model_from_huggingface
from nanochat.tokenizer import get_tokenizer

# Model configurations
MODEL_CONFIGS = {
    'default': {
        'source': 'base',
        'model_tag': 'default',
        'hf_repo_id': 'sdobson/nanochat',
        'step': 650,
        'use_standard': False,
        'checkpoint_type': 'base',
    },
    'd32': {
        'source': 'sft',
        'model_tag': 'd32',
        'hf_repo_id': 'karpathy/nanochat-d32',
        'step': 650,
        'use_standard': True,
        'checkpoint_type': 'sft',
    },
    'd34': {
        'source': 'sft',
        'model_tag': 'd34',
        'hf_repo_id': 'karpathy/nanochat-d34',
        'step': 169150,
        'use_standard': True,
        'checkpoint_type': 'sft',
    },
}

def find_tokenizer_in_parent_dirs(checkpoint_dir, max_depth=3):
    """
    Search for tokenizer.pkl in the checkpoint directory and parent directories.
    """
    current_dir = checkpoint_dir
    for depth in range(max_depth):
        tokenizer_pkl = os.path.join(current_dir, "tokenizer.pkl")
        if os.path.exists(tokenizer_pkl):
            return current_dir
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # Reached root
            break
        current_dir = parent
    return None

def ensure_tokenizer_in_checkpoint_dir(checkpoint_dir, default_tokenizer_dir):
    """
    Ensure tokenizer files exist in the checkpoint directory.
    If not found, try to copy from default location, parent directories, or download from HuggingFace.
    """
    tokenizer_pkl = os.path.join(checkpoint_dir, "tokenizer.pkl")
    token_bytes_pt = os.path.join(checkpoint_dir, "token_bytes.pt")
    
    # Check if tokenizer already exists
    if os.path.exists(tokenizer_pkl):
        if os.path.exists(token_bytes_pt):
            print(f"  ✓ Tokenizer already exists in {checkpoint_dir}")
            return True
        else:
            print(f"  ⚠ tokenizer.pkl exists but token_bytes.pt is missing")
    
    # Try to find tokenizer in parent directories
    found_dir = find_tokenizer_in_parent_dirs(checkpoint_dir)
    if found_dir:
        print(f"  → Found tokenizer in parent directory: {found_dir}")
        source_pkl = os.path.join(found_dir, "tokenizer.pkl")
        source_bytes = os.path.join(found_dir, "token_bytes.pt")
        
        if not os.path.exists(tokenizer_pkl):
            shutil.copy2(source_pkl, tokenizer_pkl)
            print(f"    ✓ Copied tokenizer.pkl")
        if os.path.exists(source_bytes) and not os.path.exists(token_bytes_pt):
            shutil.copy2(source_bytes, token_bytes_pt)
            print(f"    ✓ Copied token_bytes.pt")
        return True
    
    # Try to copy from default location
    default_pkl = os.path.join(default_tokenizer_dir, "tokenizer.pkl")
    default_bytes = os.path.join(default_tokenizer_dir, "token_bytes.pt")
    
    if os.path.exists(default_pkl):
        print(f"  → Copying tokenizer from default location...")
        if not os.path.exists(tokenizer_pkl):
            shutil.copy2(default_pkl, tokenizer_pkl)
            print(f"    ✓ Copied tokenizer.pkl")
        if os.path.exists(default_bytes) and not os.path.exists(token_bytes_pt):
            shutil.copy2(default_bytes, token_bytes_pt)
            print(f"    ✓ Copied token_bytes.pt")
        return True
    
    print(f"  ⚠ Tokenizer not found in checkpoint, parent dirs, or default location")
    return False

def update_tokenizer_from_hf(model_name, config, checkpoint_dir, force_redownload=False):
    """
    Download tokenizer from HuggingFace if local copy doesn't exist.
    If force_redownload is True, remove existing tokenizer files first.
    """
    if force_redownload:
        print(f"  → Force re-downloading tokenizer from HuggingFace...")
        tokenizer_pkl = os.path.join(checkpoint_dir, "tokenizer.pkl")
        token_bytes_pt = os.path.join(checkpoint_dir, "token_bytes.pt")
        if os.path.exists(tokenizer_pkl):
            os.remove(tokenizer_pkl)
            print(f"    → Removed existing tokenizer.pkl")
        if os.path.exists(token_bytes_pt):
            os.remove(token_bytes_pt)
            print(f"    → Removed existing token_bytes.pt")
    else:
        print(f"  → Attempting to download tokenizer from HuggingFace...")
    
    try:
        class DummyDevice:
            type = "cpu"
        
        # This will download tokenizer files to the checkpoint directory
        load_model_from_huggingface(
            hf_repo_id=config['hf_repo_id'],
            device=DummyDevice(),
            phase="eval",
            model_tag=config['model_tag'],
            use_standard_structure=config['use_standard'],
            checkpoint_type=config['checkpoint_type'],
            step=config['step']
        )
        print(f"    ✓ Downloaded tokenizer from HuggingFace")
        return True
    except Exception as e:
        print(f"    ✗ Failed to download tokenizer: {e}")
        return False

def test_model(model_name, config, device, autocast_ctx, skip_tokenizer_update=False):
    """
    Load a model, ensure tokenizer is set up, and test it with a simple question.
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    base_dir = get_base_dir()
    default_tokenizer_dir = os.path.join(base_dir, "tokenizer")
    
    # Determine checkpoint directory
    if config['source'] == 'base':
        checkpoint_dir_name = "base_checkpoints"
    elif config['source'] == 'sft':
        checkpoint_dir_name = "chatsft_checkpoints"
    else:
        checkpoint_dir_name = f"{config['source']}_checkpoints"
    
    checkpoints_dir = os.path.join(base_dir, checkpoint_dir_name)
    checkpoint_dir = os.path.join(checkpoints_dir, config['model_tag'])
    
    # Check if we should load from HuggingFace format
    use_hf_loading = False
    hf_checkpoint_dir = None
    
    # Check if checkpoint directory exists, if not try HF cache format
    if not os.path.exists(checkpoint_dir):
        # For default model, it might be in hf_checkpoints if downloaded from HF
        if config['model_tag'] == 'default' or config['source'] == 'base':
            hf_checkpoints_dir = os.path.join(base_dir, "hf_checkpoints")
            if os.path.exists(hf_checkpoints_dir):
                # Look for sdobson_nanochat directory
                default_hf_dir = os.path.join(hf_checkpoints_dir, "sdobson_nanochat")
                if os.path.exists(default_hf_dir):
                    # Find the step directory
                    import glob
                    step_dirs = glob.glob(os.path.join(default_hf_dir, "step_*"))
                    if step_dirs:
                        hf_checkpoint_dir = max(step_dirs, key=os.path.getmtime)
                        use_hf_loading = True
                        checkpoint_dir = hf_checkpoint_dir
                        print(f"  → Found model in HuggingFace cache: {checkpoint_dir}")
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Update tokenizer if needed
    if not skip_tokenizer_update:
        print(f"\nUpdating tokenizer for {model_name}...")
        tokenizer_ok = ensure_tokenizer_in_checkpoint_dir(checkpoint_dir, default_tokenizer_dir)
        
        if not tokenizer_ok:
            # Try downloading from HuggingFace
            tokenizer_ok = update_tokenizer_from_hf(model_name, config, checkpoint_dir)
        
        if not tokenizer_ok:
            print(f"  ⚠ Warning: Could not update tokenizer for {model_name}")
            print(f"    Model may still work if tokenizer exists elsewhere")
    else:
        print(f"\nSkipping tokenizer update (--skip-tokenizer-update)")
    
    # Load the model
    print(f"\nLoading model {model_name}...")
    try:
        if use_hf_loading:
            # Load from HuggingFace format using repo ID
            print(f"  → Loading from HuggingFace format...")
            model, tokenizer, meta = load_model(
                config['hf_repo_id'],
                device=device,
                phase="eval",
                model_tag=config['model_tag'],
                step=config.get('step'),
                use_standard_structure=config.get('use_standard', False),
                checkpoint_type=config.get('checkpoint_type', 'base')
            )
        else:
            # Try loading from local checkpoint first
            try:
                model, tokenizer, meta = load_model(
                    config['source'],
                    device=device,
                    phase="eval",
                    model_tag=config['model_tag'],
                    step=config.get('step')
                )
            except (FileNotFoundError, OSError) as e:
                # If local loading fails, try HuggingFace format
                print(f"  → Local checkpoint not found, trying HuggingFace format...")
                model, tokenizer, meta = load_model(
                    config['hf_repo_id'],
                    device=device,
                    phase="eval",
                    model_tag=config['model_tag'],
                    step=config.get('step'),
                    use_standard_structure=config.get('use_standard', False),
                    checkpoint_type=config.get('checkpoint_type', 'base')
                )
        
        print(f"✓ Model loaded successfully!")
        print(f"  Model config: {meta.get('model_config', 'N/A')}")
        
        # Verify tokenizer compatibility
        vocab_size = tokenizer.get_vocab_size()
        model_vocab_size = meta.get('model_config', {}).get('vocab_size')
        if model_vocab_size and vocab_size != model_vocab_size:
            print(f"  ⚠ WARNING: Tokenizer vocab size ({vocab_size}) != Model vocab size ({model_vocab_size})")
            print(f"    This may cause incorrect output!")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test the model
    print(f"\nTesting model {model_name}...")
    try:
        engine = Engine(model, tokenizer)
        
        # Get special tokens for chat format
        bos = tokenizer.get_bos_token_id()
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        
        # Test question
        question = "What is 2+2? Please explain briefly."
        print(f"Question: {question}")
        print("-" * 60)
        
        # Build conversation tokens
        conversation_tokens = [bos]
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(question))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        
        # Generate response
        print("Answer: ", end="", flush=True)
        response_tokens = []
        
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": 128,
            "temperature": 0.6,
            "top_k": 50,
        }
        
        with autocast_ctx:
            for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                token = token_column[0]
                response_tokens.append(token)
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)
                
                # Stop if we hit the assistant_end token
                if token == assistant_end:
                    break
        
        print("\n" + "-" * 60)
        
        # Check if output looks reasonable (not just garbage characters)
        response_text = tokenizer.decode(response_tokens)
        # Simple heuristic: if response is mostly non-ASCII or special chars, it's probably wrong
        ascii_ratio = sum(1 for c in response_text if ord(c) < 128 and c.isprintable()) / max(len(response_text), 1)
        if ascii_ratio < 0.3 and len(response_text) > 10:
            print(f"  ⚠ WARNING: Model output looks suspicious (mostly non-printable characters)")
            print(f"    This suggests a tokenizer mismatch.")
            print(f"    Response preview: {response_text[:100]}")
            
            # Try to fix by re-downloading tokenizer
            if not skip_tokenizer_update:
                print(f"\n  → Attempting to fix by re-downloading tokenizer...")
                # Determine the correct checkpoint directory for tokenizer fix
                checkpoint_dir_for_fix = checkpoint_dir
                if use_hf_loading and hf_checkpoint_dir:
                    checkpoint_dir_for_fix = hf_checkpoint_dir
                elif not os.path.exists(checkpoint_dir_for_fix):
                    # If checkpoint_dir doesn't exist, try to determine it from the loaded model
                    # For now, just use the config to determine where it should be
                    base_dir = get_base_dir()
                    if config.get('use_standard', False):
                        checkpoint_dir_map = {
                            "base": "base_checkpoints",
                            "mid": "mid_checkpoints",
                            "sft": "chatsft_checkpoints",
                            "rl": "chatrl_checkpoints",
                        }
                        checkpoint_dir_name = checkpoint_dir_map.get(config.get('checkpoint_type', 'base'), "base_checkpoints")
                        checkpoint_dir_for_fix = os.path.join(base_dir, checkpoint_dir_name, config['model_tag'])
                    else:
                        hf_repo_safe = config['hf_repo_id'].replace("/", "_")
                        checkpoint_dir_for_fix = os.path.join(base_dir, "hf_checkpoints", hf_repo_safe, f"step_{config.get('step', 650):06d}")
                
                if update_tokenizer_from_hf(model_name, config, checkpoint_dir_for_fix, force_redownload=True):
                    print(f"  → Tokenizer re-downloaded. Please run the test again.")
                else:
                    print(f"  ✗ Could not re-download tokenizer. Manual fix may be needed.")
            
            return False
        
        print(f"✓ Model {model_name} test completed successfully!")
        
        # Clean up
        del model, tokenizer, engine
        torch.cuda.empty_cache() if device.type == "cuda" else None
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Load all models, update tokenizers, and test them')
    parser.add_argument('--models', nargs='+', choices=['default', 'd32', 'd34'], 
                        default=['default', 'd32', 'd34'],
                        help='Models to test (default: all)')
    parser.add_argument('--skip-tokenizer-update', action='store_true',
                        help='Skip tokenizer updates, just test models')
    parser.add_argument('--device-type', type=str, default='', 
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device type (default: autodetect)')
    args = parser.parse_args()
    
    # Setup device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    
    print(f"Device: {device_type}")
    print(f"Testing models: {', '.join(args.models)}")
    print()
    
    # Test each model
    results = {}
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            print(f"⚠ Unknown model: {model_name}, skipping...")
            continue
        
        config = MODEL_CONFIGS[model_name]
        success = test_model(model_name, config, device, autocast_ctx, args.skip_tokenizer_update)
        results[model_name] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name:10s}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All models tested successfully!")
    else:
        print("✗ Some models failed. Check the output above for details.")
    print(f"{'='*60}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
