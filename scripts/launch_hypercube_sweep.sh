#!/bin/bash
#
## launch with: sbatch birdcall_hpc.sh config-file.yaml --export PATH
#
# Note that this script relies on "yq" (YAML command line tool) being in the path.
#

#SBATCH --mem=60G               # max 82GB
##SBATCH --partition=gpu
##SBATCH --qos=gpu
##SBATCH --gres=gpu:1
#SBATCH --time=00:30:00         # 30 minutes
#SBATCH -e slurm-%j.err.txt
#SBATCH -o slurm-%j.out.txt
#SBATCH --array=1-1000

./birdcall_hpc.sh temp/$((SLURM_ARRAY_TASK_ID)).yaml

