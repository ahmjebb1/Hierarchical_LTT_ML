## Latin Hypercube Parameter Sweep

Make sure that you have created a virtual environment in the "runs" directory configured for the GPU Python (i.e. installed in an HPC interactive session).

1. Create `logs`, `part_yaml`, and `temp` directories.
1. Run the `latin_hypercube.py` file, which will produce the partial yaml configs for the sweep in the `part_yaml` directory. Edit this file to set up different parameter sets.
1. Run the `create_sweep_yamls.sh` shell script, which will populate `temp` with numbered yamls (1..N) for each parameter set.
1. Run `launch_hypercube_sweep.sh` with sbatch: `sbatch --export ALL,PATH=PATH launch_hypercube_sweep.sh`. This will launch an array job with 1000 tasks. Each task will load a `temp/x.yaml` with x corresponding to the array task ID.

