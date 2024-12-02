#!/bin/bash
#
## launch with: sbatch --export=ALL,PATH=PATH launch_hypercube_sweep.sh
#
# Note that this script relies on "yq" (YAML command line tool) being in the path.
#

#SBATCH --mem=60G               # max 82GB
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-1000
#SBATCH --time=00:30:00 	# 30 minutes
#SBATCH -e logs/slurm-%j.err.txt
#SBATCH -o logs/slurm-%j.out.txt

./birdcall_hpc.sh temp/$((SLURM_ARRAY_TASK_ID)).yaml

