"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
        # Check if source is a HuggingFace repo ID (contains '/')
    if isinstance(source, str) and '/' in source and not source in ["base", "mid", "sft", "rl"]:
        # It's a HuggingFace repo ID
        return load_model_from_huggingface(source, *args, **kwargs)
        
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_model_from_huggingface(hf_repo_id: str, step: int = None, device=None, phase="eval", model_tag=None, use_standard_structure=False, checkpoint_type="base"):
    """
    Load a nanochat model from a HuggingFace repository.
    
    Args:
        hf_repo_id: HuggingFace repository ID (e.g., "sdobson/nanochat" or "karpathy/nanochat-d34")
        step: Checkpoint step to load (e.g., 650, 169150). If None, will try to find the latest step.
        device: PyTorch device to load the model on. If None, will auto-detect.
        phase: "train" or "eval"
        model_tag: Model tag for standard directory structure (e.g., "d34"). If None and use_standard_structure=True, will try to infer from repo name.
        use_standard_structure: If True, place files in standard structure instead of hf_checkpoints/
        checkpoint_type: Type of checkpoint directory when using standard structure: "base", "mid", "sft", "rl" (default: "base")
    
    Returns:
        model, tokenizer, meta_data
    """
    from huggingface_hub import hf_hub_download
    import tempfile
    import shutil
    
    # Auto-detect device if not provided
    if device is None:
        from nanochat.common import autodetect_device_type
        device_type = autodetect_device_type()
        device = torch.device(device_type)
        log0(f"Auto-detected device: {device}")
    
    base_dir = get_base_dir()
    
    # Determine where to store the model files
    if use_standard_structure:
        # Use standard directory structure (e.g., base_checkpoints/d34/ or chatsft_checkpoints/d34/)
        if model_tag is None:
            # Try to infer model_tag from repo name (e.g., "nanochat-d34" -> "d34")
            repo_name = hf_repo_id.split("/")[-1]
            if "d34" in repo_name or "d32" in repo_name or "d20" in repo_name:
                # Extract depth from repo name
                import re
                match = re.search(r'd(\d+)', repo_name)
                if match:
                    model_tag = f"d{match.group(1)}"
                else:
                    model_tag = repo_name.replace("nanochat-", "")
            else:
                model_tag = repo_name.replace("nanochat-", "")
        
        # Map checkpoint_type to directory name
        checkpoint_dir_map = {
            "base": "base_checkpoints",
            "mid": "mid_checkpoints",
            "sft": "chatsft_checkpoints",
            "rl": "chatrl_checkpoints",
        }
        checkpoint_dir_name = checkpoint_dir_map.get(checkpoint_type, "base_checkpoints")
        cache_dir = os.path.join(base_dir, checkpoint_dir_name)
        log0(f"Using standard directory structure: {checkpoint_dir_name}/{model_tag}")
    else:
        # Use HuggingFace-specific cache directory
        cache_dir = os.path.join(base_dir, "hf_checkpoints", hf_repo_id.replace("/", "_"))
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # If step is not provided, try to find the latest step
    if step is None:
        # Try to list files and find the highest step number
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo_id=hf_repo_id, repo_type="model")
            model_files = [f for f in files if f.startswith("model_") and f.endswith(".pt")]
            if model_files:
                # Extract step numbers and find the max
                steps = []
                for f in model_files:
                    try:
                        step_num = int(f.replace("model_", "").replace(".pt", ""))
                        steps.append(step_num)
                    except ValueError:
                        continue
                if steps:
                    step = max(steps)
                    log0(f"No step provided, using latest step: {step}")
        except Exception as e:
            log0(f"Could not auto-detect step, trying step 650: {e}")
            step = 650  # Default fallback
    
    assert step is not None, "Could not determine checkpoint step"
    
    # Determine checkpoint directory
    if use_standard_structure and model_tag:
        # Use standard structure: base_checkpoints/model_tag/
        checkpoint_dir = os.path.join(cache_dir, model_tag)
    else:
        # Use step-based directory: hf_checkpoints/repo/step_XXXXXX/
        checkpoint_dir = os.path.join(cache_dir, f"step_{step:06d}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tokenizer files go to the base tokenizer directory (shared across checkpoints)
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Files to download to checkpoint directory
    checkpoint_files = {
        f"model_{step:06d}.pt": f"model_{step:06d}.pt",
        f"meta_{step:06d}.json": f"meta_{step:06d}.json",
    }
    
    # Files to download to tokenizer directory
    tokenizer_files = {
        "token_bytes.pt": "token_bytes.pt",
        "tokenizer.pkl": "tokenizer.pkl",
    }
    
    log0(f"Downloading model files from {hf_repo_id} (step {step})...")
    
    # Download checkpoint files
    for hf_filename, local_filename in checkpoint_files.items():
        local_path = os.path.join(checkpoint_dir, local_filename)
        if not os.path.exists(local_path):
            try:
                downloaded_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=hf_filename,
                    cache_dir=cache_dir,
                    local_dir=checkpoint_dir,
                    local_dir_use_symlinks=False,
                )
                # Move to the expected location if needed
                if downloaded_path != local_path:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    shutil.move(downloaded_path, local_path)
                log0(f"Downloaded {hf_filename}")
            except Exception as e:
                raise RuntimeError(f"Failed to download {hf_filename} from {hf_repo_id}: {e}")
        else:
            log0(f"Using cached {local_filename}")
    
    # Download tokenizer files to the tokenizer directory
    for hf_filename, local_filename in tokenizer_files.items():
        local_path = os.path.join(tokenizer_dir, local_filename)
        if not os.path.exists(local_path):
            try:
                downloaded_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=hf_filename,
                    cache_dir=cache_dir,
                    local_dir=tokenizer_dir,
                    local_dir_use_symlinks=False,
                )
                # Move to the expected location if needed
                if downloaded_path != local_path:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    shutil.move(downloaded_path, local_path)
                log0(f"Downloaded {hf_filename} to tokenizer directory")
            except Exception as e:
                log0(f"Warning: Could not download {hf_filename}: {e}")
                # Tokenizer files might not be critical if they already exist locally
        else:
            log0(f"Using cached {local_filename} in tokenizer directory")
    
    # Now use the existing build_model function to load
    return build_model(checkpoint_dir, step, device, phase)