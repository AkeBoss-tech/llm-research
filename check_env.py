import sys
import os
try:
    import cmake  # Preload newer libstdc++ for pandas compatibility on older systems
except ImportError:
    pass

def check_import(module_name):
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {module_name:<15} (v{version}) imported successfully")
        return module
    except ImportError as e:
        print(f"❌ {module_name:<15} FAILED to import: {e}")
        return None

print(f"Checking environment: {sys.prefix}")
print(f"Python version: {sys.version}")
print("-" * 40)

# Core dependencies
torch = check_import("torch")
numpy = check_import("numpy")
pandas = check_import("pandas")
wandb = check_import("wandb")
datasets = check_import("datasets")

# Check for nanochat (local package)
try:
    import nanochat
    print(f"✅ nanochat        imported successfully (from {os.path.dirname(nanochat.__file__)})")
except ImportError as e:
    print(f"❌ nanochat        FAILED to import: {e}")
    print("   (Did you run 'pip install -e .'?)")

print("-" * 40)

# Check GPU/CUDA availability
if torch:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:   {torch.version.cuda}")
        print(f"Device count:   {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  WARNING: CUDA is NOT available. PyTorch is running on CPU.")

# Check Numpy Version Compatibility
if numpy:
    major_ver = int(numpy.__version__.split('.')[0])
    if major_ver >= 2:
        print("⚠️  WARNING: NumPy version is >= 2.0. This might cause issues with PyTorch < 2.3.")
