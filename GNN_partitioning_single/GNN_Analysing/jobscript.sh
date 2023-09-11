#!/usr/bin/zsh

### Ask for 10 GB memory
#SBATCH --mem=10G
#SBATCH --time=3:00:00

### Name the job
#SBATCH --job-name=GNN-Data-Generation

### Declare the merged STDOUT/STDERR file
#SBATCH --output=output1.%J.txt

# Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

# Now you can activate your configured conda environments
conda activate base

### Begin of executable commands
python -u ./comparison_synthetic_data.py