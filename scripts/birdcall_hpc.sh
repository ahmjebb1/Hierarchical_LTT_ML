#!/bin/bash
#
## launch with: sbatch birdcall_hpc.sh config-file.yaml --export PATH
#
# Note that this script relies on "yq" (YAML command line tool) being in the path.
#

# memory request (max 82GB)
#SBATCH --mem=80G		
#SBATCH --partition=gpu
#SBATCH --qos=gpu
# 1 GPU requested
#SBATCH --gres=gpu:1 
# 30 minute max runtime
#SBATCH --time=00:30:00 	

# Means this script runs 50 separate jobs automatically (a key HPC parallelisation mechanism)
#SBATCH --array=0-49              # Array range 0-9, equivalent to 10 runs  

##SBATCH -e "logs/%x.slurm-%j_%A_%a.err.txt"
##SBATCH -o "logs/%x.slurm-%j_%A_%a.out.txt"

#SBATCH -e "logs/%x.%a.err.txt"
#SBATCH -o "logs/%x.%a.out.txt"

##SBATCH --job-name=$expname


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

# extract experiment name
expname=$(~/bin/yq e .expname $1)

# load modules and environment
module load CUDA
#module load Python/3.11.3
module load GCCcore/12.3.0
module load ncurses/6.4-GCCcore-12.3.0
module load XZ/5.4.2-GCCcore-12.3.0
module load zlib/1.2.13-GCCcore-12.3.0
module load libreadline/8.2-GCCcore-12.3.0
module load libffi/3.4.4-GCCcore-12.3.0
module load binutils/2.40-GCCcore-12.3.0
module load Tcl/8.6.13-GCCcore-12.3.0
module load OpenSSL/1.1
module load bzip2/1.0.8-GCCcore-12.3.0
module load SQLite/3.42.0-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
source venv-hpc/bin/activate

# Uses MLFLOW for experiment logging
export MLFLOW_TRACKING_URI=$(~/bin/yq e .mlflow_uri credentials.yaml)
export MLFLOW_TRACKING_USERNAME=$(~/bin/yq e .mlflow_user credentials.yaml)
export MLFLOW_TRACKING_PASSWORD=$(~/bin/yq e .mlflow_pw credentials.yaml)

# ensure output artifacts exist:
#touch slurm-$SLURM_JOB_ID.err.txt
#touch slurm-$SLURM_JOB_ID.out.txt
#extract the name of the best model
best_model_filename=$(~/bin/yq -e '.best_model' $1)
touch $best_model_filename

echo -n "Which Python: "
which python

# run the training script
python train.py "$1"

rm $best_model_filename

# send outputs to mlflow
#mlflow_id=$(cat $SLURM_JOB_ID.mlflow_run)
#mlflow artifacts log-artifact --local-file slurm-$SLURM_JOB_ID.err.txt --run-id $mlflow_id
#mlflow artifacts log-artifact --local-file slurm-$SLURM_JOB_ID.out.txt --run-id $mlflow_id
#mlflow artifacts log-artifact --local-file $best_model_filename        --run-id $mlflow_id

