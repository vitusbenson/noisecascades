# noisecascades

Python framework for studying dynamics in double-well potentials under alpha-stable levy noise.

## Installation

```bash
conda create -n noisecascades python=3.9
conda deactivate
conda activate noisecascades
conda install -c conda-forge mamba
mamba install -c numpy xarray zarr pyyaml tqdm scipy numba
pip install fire numcodecs
git clone https://github.com/vitusbenson/noisecascades
cd noisecascades
pip install -e .
```

## General structure

**Experiment Host**
- Parses the given config yaml
- If node_idx == -1:
    - Creates the empty zarr file
    - Creates the slurm file if given
- Else:
    - Starts multiprocessing for the given node idx, i.e. distributes chunks that are assigned to this node among different client processes.

**Experiment Client**
- Runs within one process, handles 1 chunk of data
- Runs simulations and stores them

**Integrator**
- Does one simulation, handles different simulation types for maximum efficiancy

## Usage

With just one node
```bash
cd noisecascades
python noisecascades/host.py configs/your_experiment.yaml -1
python noisecascades/host.py configs/your_experiment.yaml 0
```

Using a slurm cluster (do this on login node)
```bash
cd noisecascades
python noisecascades/host.py configs/your_experiment.yaml -1
sbatch experiments/your_experiment/your_experiment.slurm
```

