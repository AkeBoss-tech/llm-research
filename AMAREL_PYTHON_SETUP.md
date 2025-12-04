# Setting Up Python 3.10+ on Amarel

Since nanochat requires Python >=3.10 but Amarel only provides Python 3.8.2 via modules, you need to build Python from source or use conda.

## Quick Start: Build Python from Source

### Step 1: Run the setup script (one-time setup)

On a login node or interactive compute node:

```bash
cd ~/llm-research
bash setup_python_amarel.sh
```

This script will:
1. Install libffi (required for _ctypes module)
2. Download and build Python 3.10.13 from source
3. Install it to `~/python/3.10.13/`
4. Update your `~/.bashrc` with the necessary paths

**Time required**: ~15-20 minutes

### Step 2: Reload your shell

```bash
source ~/.bashrc
```

### Step 3: Verify installation

```bash
which python3
python3 --version
# Should show: Python 3.10.13
```

### Step 4: Submit your job

The batch script will automatically detect and use your custom Python:

```bash
sbatch run_gsm8k_eval_amarel.sh
```

## Alternative: Using Miniconda

If you prefer conda instead of building from source:

```bash
# Install Miniconda (one-time)
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment (one-time)
conda create -n nanochat python=3.10 -y
conda activate nanochat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets fastapi files-to-prompt psutil regex tiktoken tokenizers uvicorn wandb
```

The batch script will automatically detect and use conda if available.

## What Gets Installed

### Custom Python Build:
- **Location**: `~/python/3.10.13/`
- **Size**: ~200-300 MB
- **Includes**: Python interpreter, pip, standard library

### libffi:
- **Location**: `~/libffi/3.4.2/`
- **Size**: ~10-20 MB
- **Purpose**: Required for Python's _ctypes module (needed by PyTorch)

## Troubleshooting

### Setup script fails

If the setup script fails, you can run it step by step:

```bash
# 1. Install libffi
cd /tmp
wget https://github.com/libffi/libffi/releases/download/v3.4.2/libffi-3.4.2.tar.gz
tar -zxf libffi-3.4.2.tar.gz
cd libffi-3.4.2
./configure --prefix=$HOME/libffi/3.4.2
make -j 4
make install

# 2. Build Python
cd /tmp
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
tar -zxf Python-3.10.13.tgz
cd Python-3.10.13
./configure --prefix=$HOME/python/3.10.13 \
    CXX=g++ \
    --with-ensurepip=install \
    LDFLAGS=-L$HOME/libffi/3.4.2/lib64 \
    CPPFLAGS=-I$HOME/libffi/3.4.2/include
make -j 8
make install
```

### Python not found in batch job

Make sure `~/.bashrc` was updated and contains:

```bash
export PATH="$HOME/python/3.10.13/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/python/3.10.13/lib:$LD_LIBRARY_PATH"
```

The batch script will also set these paths automatically.

### Out of space

Check your home directory quota:

```bash
mmlsquota --block-size=auto home
```

Python installation uses ~200-300 MB. If you're low on space, consider:
- Using `/scratch` for temporary files
- Cleaning up old files
- Using conda instead (can be installed to a different location)

## Notes

- The setup script only needs to be run **once**
- Python will be available for all future jobs
- The batch script automatically detects and uses your custom Python
- You can have multiple Python versions in `~/python/` (e.g., 3.10.13, 3.11.5)

