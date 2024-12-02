#!/bin/bash
# This example launches 20 HPC runs for 20 different parameter values for the
# 'max_examples_per_class' parameter. mlflow credentials are copied into the
# config from "credentials.yaml"
#
# Note that this script relies on "yq" (YAML command line tool) being in the path.
#

mlflow_credentials_file=credentials.yaml
job_script=birdcall_hpc.sh
template_config=hypercube_sweep.yaml # Edit the template file to configure common parameters!

MLFLOW_TRACKING_URI=$(yq e .mlflow_uri $mlflow_credentials_file)
MLFLOW_TRACKING_USERNAME=$(yq e .mlflow_user $mlflow_credentials_file)
MLFLOW_TRACKING_PASSWORD=$(yq e .mlflow_pw $mlflow_credentials_file)

# Set mlflow credentials in the configuration:
yq e ".mlflow_user=\"$MLFLOW_TRACKING_USERNAME\"" -i $template_config
yq e ".mlflow_pw=\"$MLFLOW_TRACKING_PASSWORD\"" -i $template_config
yq e ".mlflow_uri=\"$MLFLOW_TRACKING_URI\"" -i $template_config

# Create mlflow experiment:
expname=$(yq e .mlflow_expname $template_config)
mlflow experiments create -n "$expname"

sleep 10
counter=0

for x in part_yaml/* ; do

  #echo $x
  cp $template_config temp/$counter.yaml
  cat $x >> temp/$counter.yaml
  # sbatch --export ALL,PATH=$PATH $job_script temp/$counter.yaml
  ((counter++)) 

done
