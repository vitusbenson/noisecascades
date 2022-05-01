"""noisecascades
Python framework for studying dynamics in double-well potentials under alpha-stable levy noise
"""
__version__ = "0.0.1"
__author__ = 'Vitus Benson'
__credits__ = '2022, Potsdam Institute for Climate Impact Research and Leipzig University'

from . import host, client, simulate

from noisecascades.host import ExperimentHost
from noisecascades.client import ExperimentClient
from noisecascades.simulate.prepare import Integrator


# CURRENT TODO:
# Write code in host.py that translates physical parameters into reduced parameters (i.e. GMT -> c... etc.)
# Write code in host.py that sets up the slurm script, see https://vsoch.github.io/lessons/sherlock-jobs/
# Set up python packaging stuff..

# Experiment configs as yamls

# Experiment host:
#   Python class
#   Reads config yaml
#   Configs have two sections: Setup & Network configs
#   The former is for setting up the zarr, the multiprocessing & slurm
#   The latter sets up network configs
#   Build list of all configs in experiment
#   Splits List into chunksafe Parts of n processes over N nodes. Can infer N for nodes automatically.
#   Builds a dummy zarr array with proper axis and chunking
#   Builds a slurm script if N > 1
#   Starts call_clients method of experiment, via python if N == 1 else via slurm.
#   call_clients runs clients in parallel over n processes for given node idx

# Slurm script:
#   Starts Experiment host but with specific node idx

# Experiment client:
#   Python class
#   Gets list of configs
#   For each config runs simulations
#   Compute only what is necessary (i.e. low memory usage, stops compute at first passage, etc.)
#   Stores what shall be stored: Timeseries, FPT, Oktant, Residence fractions, Residence periodicity,...

# Simulate functions:
#   All the below have it for general x^4 potential
#   1 low-low-level numba function that computes just 1 step
#   Then need 1 low-level numba function per type of experiment
#       - Timeseries
#       - FPT / ExitOktant
#       - Residence fractions & periodicity
#   Need 1 high-level function that prepares parameters etc and chooses right low-level function