#!/bin/bash
default_dir=exp-`date +"%Y%m%d%H%M%S"`
dir="${1:-$default_dir}"
echo "Generating experiment folder '$dir' ... "
rm -rf $dir
mkdir $dir
mkdir $dir/temp
mkdir $dir/part_yaml
mkdir $dir/logs
cp -v train.py $dir
cp -v dataset.py $dir
cp -v util.py $dir
cp -v models.py $dir
cp -v cnn.py $dir
cp -v config.py $dir
cp -v example.yaml $dir/run.yaml
cp -v scripts/birdcall_hpc.sh $dir
cp -v scripts/credentials.yaml $dir
cp -v scripts/create_sweep_yamls.sh $dir
cp -v scripts/latin_hypercube.py $dir
cp -v scripts/launch_hypercube_sweep.sh $dir
cp -v scripts/hypercube_sweep.yaml $dir
cp -v scripts/launch_jobs.sh $dir
cp -v scripts/aggregate_data.sh $dir
cp -v --no-dereference venv-hpc $dir/venv-hpc
echo "Complete."
