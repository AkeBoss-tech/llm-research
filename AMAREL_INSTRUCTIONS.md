# Running GSM8K Evaluation on Amarel

This guide explains how to run the GSM8K benchmark evaluation on the Amarel HPC cluster.

## Files Created

1. **`run_gsm8k_eval_amarel.sh`** - SLURM batch script for submitting the job
2. **`generate_eval_summary.py`** - Script to generate text summaries from results
3. **`eval_gsm8k_all_models.py`** - Main evaluation script (already exists)
4. **`visualize_gsm8k_results.py`** - HTML visualization generator (already exists)

## Quick Start

### 1. Transfer files to Amarel

```bash
# From your local machine, transfer the project to Amarel
scp -r /path/to/llm-research username@amarel.rutgers.edu:~/llm-research
```

### 2. SSH into Amarel

```bash
ssh username@amarel.rutgers.edu
cd ~/llm-research
```

### 3. Submit the batch job

```bash
sbatch run_gsm8k_eval_amarel.sh
```

### 4. Monitor the job

```bash
# Check job status
squeue -u $USER

# View output in real-time (replace JOBID with your job ID)
tail -f gsm8k_eval_JOBID.out

# Or check the error log
tail -f gsm8k_eval_JOBID.err
```

### 5. Check results

After the job completes, results will be in:
```
~/gsm8k_eval_results/job_JOBID/
```

This directory contains:
- `gsm8k_results_TIMESTAMP.json` - Full JSON results
- `report.html` - Interactive HTML visualization
- `summary.txt` - Text summary report
- `eval_log.txt` - Full evaluation log

## Customizing the Job

### Edit the batch script

You can modify `run_gsm8k_eval_amarel.sh` to:

1. **Change resource requirements:**
   ```bash
   #SBATCH --time=12:00:00      # Shorter time limit
   #SBATCH --mem=128G           # More memory
   #SBATCH --gres=gpu:2        # More GPUs
   ```

2. **Change evaluation parameters:**
   ```bash
   python eval_gsm8k_all_models.py \
       --models d32 d34 \              # Only test specific models
       --max-problems 500 \            # Limit problems
       --num-samples 5 \               # Multiple samples per problem
       --temperature 0.7 \             # Higher temperature
   ```

3. **Adjust environment setup:**
   - Uncomment/modify module loads based on Amarel's available modules
   - Change Python environment setup (uv/conda/venv)

### Quick test run

For a quick test, modify the script to use fewer problems:

```bash
python eval_gsm8k_all_models.py \
    --models default \
    --max-problems 10 \
    --output-dir "$OUTPUT_DIR"
```

## Environment Setup

The script tries three methods in order:

1. **uv** (recommended) - If `uv` is available, it will use it
2. **conda** - If conda is available
3. **venv** - Falls back to system Python with venv

### Using uv (recommended)

If `uv` is not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Using conda

If you prefer conda:
```bash
# Create environment
conda create -n nanochat python=3.10 -y
conda activate nanochat

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Module Setup (Amarel-specific)

You may need to load modules. Uncomment and adjust in the script:

```bash
module load python/3.10
module load cuda/12.1
module load cudnn/8.9
```

Check available modules:
```bash
module avail
```

## Results Location

Results are saved to:
```
~/gsm8k_eval_results/job_JOBID/
```

To access results:
```bash
cd ~/gsm8k_eval_results/job_JOBID/
ls -lh

# View summary
cat summary.txt

# Open HTML report (if you have a web server or transfer it locally)
# Or download to local machine:
scp username@amarel.rutgers.edu:~/gsm8k_eval_results/job_JOBID/report.html ./
```

## Downloading Results

To download results to your local machine:

```bash
# From your local machine
scp -r username@amarel.rutgers.edu:~/gsm8k_eval_results/job_JOBID ./

# Or just the important files
scp username@amarel.rutgers.edu:~/gsm8k_eval_results/job_JOBID/*.{json,html,txt} ./
```

## Troubleshooting

### Job fails immediately

1. Check the error log: `cat gsm8k_eval_JOBID.err`
2. Verify GPU availability: The script checks this automatically
3. Check module availability: `module avail`

### Out of memory

Increase memory in the batch script:
```bash
#SBATCH --mem=128G
```

### Timeout

Increase time limit:
```bash
#SBATCH --time=48:00:00
```

### Python environment issues

Try using conda instead:
```bash
# Comment out uv section, uncomment conda section
```

### CUDA not available

Check if CUDA modules are loaded and PyTorch was installed with CUDA support.

## Generating Additional Reports

After the job completes, you can generate additional summaries:

```bash
python generate_eval_summary.py ~/gsm8k_eval_results/job_JOBID/gsm8k_results_*.json
```

## Email Notifications

The script is configured to send email on job completion/failure. Make sure your email is set correctly in the script:

```bash
#SBATCH --mail-user=$USER@rutgers.edu
```

## Example Workflow

```bash
# 1. Submit job
sbatch run_gsm8k_eval_amarel.sh
# Note the job ID, e.g., 12345678

# 2. Monitor (in another terminal)
watch -n 30 'squeue -u $USER'

# 3. Check progress
tail -f gsm8k_eval_12345678.out

# 4. When complete, check results
cd ~/gsm8k_eval_results/job_12345678/
cat summary.txt
ls -lh

# 5. Download to local machine
# From local machine:
scp -r username@amarel.rutgers.edu:~/gsm8k_eval_results/job_12345678 ./
```

## Notes

- The first run will download model files (~2-8GB per model), which may take time
- Model files are cached in `~/.cache/nanochat/` for subsequent runs
- Evaluation time depends on number of problems and models
- Full GSM8K test set has ~1319 problems
- Each model evaluation may take 1-4 hours depending on GPU

# Setting Up and Running on Amarel

## Directory Structure on Amarel

Based on the [Amarel documentation](https://sites.google.com/view/cluster-user-guide/amarel), here's the recommended setup:

### Storage Locations:
- **`/home/$USER/`** - Your home directory (persistent, 100GB quota) - **Use for code**
- **`/scratch/$USER/`** - Temporary storage (1TB quota, auto-purged after 90 days) - **Use for results**
- **`/projects/f_projectname/`** - Project storage (if you have access) - **Alternative for code/results**

## Step 1: Clone Repository

**Recommended: Clone to your home directory**

```bash
cd ~
git clone <your-repo-url> llm-research
cd llm-research
```

**Alternative: Clone to projects directory (if you have one)**

```bash
cd /projects/f_yourproject/
git clone <your-repo-url> llm-research
cd llm-research
```

## Step 2: Verify Files

Make sure these files are present:
```bash
ls -la run_gsm8k_eval_amarel.sh
ls -la eval_gsm8k_all_models.py
ls -la visualize_gsm8k_results.py
```

## Step 3: Submit Job

From the repository directory:

```bash
sbatch run_gsm8k_eval_amarel.sh
```

## Step 4: Monitor Job

```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with your actual job ID)
tail -f gsm8k_eval_JOBID.out

# View errors
tail -f gsm8k_eval_JOBID.err
```

## Step 5: Find Results

Results will be saved to:
- **Primary location**: `/scratch/$USER/gsm8k_eval_results/job_JOBID/`
- **Fallback location**: `~/gsm8k_eval_results/job_JOBID/` (if /scratch not available)

```bash
# Check if /scratch exists
ls -ld /scratch/$USER

# View results
ls -lh /scratch/$USER/gsm8k_eval_results/job_JOBID/
# OR
ls -lh ~/gsm8k_eval_results/job_JOBID/
```

## Important Notes

### Storage Quotas

- **Home directory (`/home`)**: 100GB quota
  - Model cache goes here: `~/.cache/nanochat/` (~2-8GB per model)
  - Code goes here: `~/llm-research/`
  
- **Scratch directory (`/scratch`)**: 1TB quota, auto-purged after 90 days
  - Results go here: `/scratch/$USER/gsm8k_eval_results/`
  - **Important**: Download results before 90 days or they'll be deleted!

### Checking Storage Usage

```bash
# Check home directory usage
mmlsquota --block-size=auto home

# Check scratch directory usage
mmlsquota --block-size=auto scratch
```

### Downloading Results

Before the 90-day purge, download your results:

```bash
# From your local machine
scp -r username@amarel.rutgers.edu:/scratch/username/gsm8k_eval_results/job_JOBID ./

# Or just the important files
scp username@amarel.rutgers.edu:/scratch/username/gsm8k_eval_results/job_JOBID/*.{json,html,txt} ./
```

## Customizing the Script

The script automatically:
- Detects the repository location (wherever you cloned it)
- Uses `/scratch` for results if available, falls back to `~/` if not
- Uses `~/.cache/nanochat/` for model cache (persistent across jobs)

### If you cloned to a different location:

The script will work from any location as long as you run `sbatch` from within the repository directory.

### If you want to use /projects instead:

Edit the script and change:
```bash
OUTPUT_DIR="/projects/f_yourproject/gsm8k_eval_results/job_${SLURM_JOB_ID}"
```

## Troubleshooting

### "Cannot change to repository directory"
- Make sure you're running `sbatch` from within the cloned repository
- Or edit the script to set `REPO_DIR` to the absolute path

### "/scratch not available"
- The script will automatically fall back to `~/gsm8k_eval_results/`
- This is fine, but watch your home directory quota (100GB limit)

### "Out of space"
- Check quotas: `mmlsquota --block-size=auto home` and `mmlsquota --block-size=auto scratch`
- Clean up old results: `rm -rf /scratch/$USER/gsm8k_eval_results/job_OLDJOBID/`
- Or move results to /projects if you have access

### Module issues
- Check available modules: `module avail`
- Uncomment and adjust module loads in the script if needed

