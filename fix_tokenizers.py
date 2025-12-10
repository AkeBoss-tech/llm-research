"""
Script to download and fix tokenizers for all nanochat models.
This ensures each model checkpoint has its own tokenizer files to prevent conflicts.
"""
import os
import argparse
from nanochat.checkpoint_manager import load_model_from_huggingface
from nanochat.common import compute_init, autodetect_device_type

def fix_tokenizers():
    print("Fixing tokenizers for all models...")
    print("-" * 50)
    
    # Setup minimal compute (CPU is fine for just downloading)
    device_type = "cpu"
    compute_init(device_type)
    
    models_to_fix = [
        # (repo_id, model_tag, use_standard, checkpoint_type)
        ("sdobson/nanochat", None, False, "base"),
        ("karpathy/nanochat-d32", "d32", True, "sft"),
        ("karpathy/nanochat-d34", "d34", True, "sft"),
    ]
    
    for repo, tag, use_std, ckpt_type in models_to_fix:
        print(f"\nProcessing {repo}...")
        try:
            # We just need to trigger the download logic
            # This will now download tokenizer files to the model directory
            # Pass a dummy device object that has a .type attribute
            class DummyDevice:
                type = "cpu"
            
            load_model_from_huggingface(
                hf_repo_id=repo,
                device=DummyDevice(), # Pass object with .type attribute instead of string
                phase="eval",
                model_tag=tag,
                use_standard_structure=use_std,
                checkpoint_type=ckpt_type
            )
            print(f"✅ Successfully updated tokenizer for {repo}")
            
        except Exception as e:
            print(f"❌ Error processing {repo}: {e}")

    print("\n" + "-" * 50)
    print("All done! Tokenizers should now be isolated per model.")

if __name__ == "__main__":
    fix_tokenizers()



