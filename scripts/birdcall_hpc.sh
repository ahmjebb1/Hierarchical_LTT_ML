#!/bin/bash
#
## launch with: sbatch birdcall_hpc.sh config-file.yaml --export PATH
#
# Note that this script relies on "yq" (YAML command line tool) being in the path.
#

#SBATCH --mem=60G		# max 82GB
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00 	# 30 minutes
#SBATCH -e logs/slurm-%j.err.txt
#SBATCH -o logs/slurm-%j.out.txt

# Check if the first parameter is provided (the YAML config)
if [ -z "$1" ]; then
  echo "Error: No parameter provided."
  echo "Usage: $0 <parameter1>"
  exit 1
fi

# Check if the YAML exists
if [ ! -e "$1" ]; then
  echo "Error: provided configuration file does not exist."
  exit 1
fi

# load modules and environment
module load CUDA
module load Python/3.11.3
source venv/bin/activate


export MLFLOW_TRACKING_URI=$(~/bin/yq e .mlflow_uri credentials.yaml)
export MLFLOW_TRACKING_USERNAME=$(~/bin/yq e .mlflow_user credentials.yaml)
export MLFLOW_TRACKING_PASSWORD=$(~/bin/yq e .mlflow_pw credentials.yaml)

# ensure output artifacts exist:
#touch slurm-$SLURM_JOB_ID.err.txt
#touch slurm-$SLURM_JOB_ID.out.txt

best_model_filename=$(~/bin/yq -e '.best_model' $1)
touch $best_model_filename

which python

# run the training script
python ../train.py "$1"

rm $best_model_filename

# send outputs to mlflow
mlflow_id=$(cat $SLURM_JOB_ID.mlflow_run)
#mlflow artifacts log-artifact --local-file slurm-$SLURM_JOB_ID.err.txt --run-id $mlflow_id
#mlflow artifacts log-artifact --local-file slurm-$SLURM_JOB_ID.out.txt --run-id $mlflow_id
#mlflow artifacts log-artifact --local-file $best_model_filename        --run-id $mlflow_id

