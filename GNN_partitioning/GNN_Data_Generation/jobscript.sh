#!/usr/bin/zsh

### Ask for 10 GB memory
#SBATCH --mem=10G
#SBATCH --time=6:00:00

### Name the job
#SBATCH --job-name=GNN-Data-Generation

### Declare the merged STDOUT/STDERR file
#SBATCH --output=output3.%J.txt

# Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

# Now you can activate your configured conda environments
conda activate base

### Begin of executable commands
python ./gnn_generation.py run3 20 "[2,3,4,5,6,7,8]"