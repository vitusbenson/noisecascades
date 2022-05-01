

import numpy as np
import xarray as xr
import zarr

from pathlib import Path

from noisecascades.simulate.prepare import Integrator

class ExperimentClient:

    def __init__(self, config):
        self.config = config
        self.mode = config["mode"]
        self.zarrpath = Path(config["zarrpath"])
        self.zarrgroup = zarr.open_group(str(self.zarrpath))

    
    def get_coord_idxs(self, network_config):

        idxs = tuple()
        for axis in self.config["axes"]:
            if axis in network_config:
                idx = np.where(self.zarrgroup[axis][:] == network_config[axis])[0][0]
                idxs += (idx,)
            else:
                idxs += (slice(None),)
        
        return idxs

    def save_results(self, network_config, results):

        coord_idxs = self.get_coord_idxs(network_config)
        vars = self.config["dimension_names"]

        if self.mode == "timeseries":

            _, xs = results

            for i, var in enumerate(vars):
                self.zarrgroup[var].set_orthogonal_selection(coord_idxs, xs[:,i])

        elif self.mode == "fpt_orthant":
            
            fpt, orthant_entered = results

            orthant = ''.join(list((orthant_entered * 1.).astype(int).astype(str)))

            if "fpt" in vars:
                self.zarrgroup["fpt"].set_orthogonal_selection(coord_idxs, fpt)

            if "orthant" in vars:
                self.zarrgroup["orthant"].set_orthogonal_selection(coord_idxs, orthant)


    @classmethod
    def run_simulations(cls, config):

        self = cls(config["setup"])

        for network_config in config["network_configs"]:

            results = Integrator.integrate_networkconfig(network_config)
            
            self.save_results(network_config, results)
