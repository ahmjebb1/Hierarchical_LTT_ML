#!/bin/bash

# Assuming we have yamls in yaml/ called $exp.run.yaml, this script
# will launch HPC jobs that load up each yaml. Useful for running a batch
# without specifically controlling which paramset to use within the job array.
for exp in ltt10 ltt20 ltt30 ltt40 ltt50 wwc10 wwc20 wwc30 wwc40 wwc50 ; do sbatch --job-name="$exp" birdcall_hpc.sh yaml/$exp.run.yaml ; done

