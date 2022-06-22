

from http.client import NETWORK_AUTHENTICATION_REQUIRED
import os
import numpy as np
import xarray as xr
import yaml
import zarr

from numcodecs import Blosc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from copy import deepcopy

from noisecascades.client import ExperimentClient

class ExperimentHost:


    def __init__(self, configpath, node_idx):
        
        self.configpath = Path(configpath)

        self.config = self.parse_config(node_idx)
        
    def generate_network_params(self, network_config):
        # HERE Translate network params into dt, Tend, x_init, c, A, alphas, sigmas, taos and add inplace
        
        if network_config["type"] == "wunderling":

            if "dt" not in network_config:
                network_config["dt"] = 10.0
            if "Tend" not in network_config:
                network_config["Tend"] = 100000
            
            network_config["x_init"] = -np.ones_like(np.array(network_config["Tlim"])) if "x_init" not in network_config else np.array(network_config["x_init"])
            network_config["taos"] = np.array([1865.4339212188884, 114.21024007462583, 913.6819205970066, 19.035040012437637])[:len(network_config["Tlim"])] if "taos" not in network_config else np.array(network_config["taos"])
                
            GMT = network_config["GMT"]
            d = network_config["strength"]
            if "pf_thc_to_gis" in network_config:
                if len(network_config["Tlim"]) == 4:
                    coupl = [
                        [0.0, -network_config["pf_thc_to_gis"], network_config["pf_wais_to_gis"], 0.0],
                        [network_config["pf_gis_to_thc"], 0.0, network_config["k_wais_to_thc"] * network_config["pf_wais_to_thc"], 0.0],
                        [network_config["pf_gis_to_wais"], network_config["pf_thc_to_wais"], 0.0, 0.0],
                        [0.0, network_config["k_thc_to_amaz"] * network_config["pf_thc_to_amaz"], 0.0, 0.0]
                    
                    ]
                else:
                    coupl = [
                        [0.0, -network_config["pf_thc_to_gis"], network_config["pf_wais_to_gis"]],
                        [network_config["pf_gis_to_thc"], 0.0, network_config["k_wais_to_thc"] * network_config["pf_wais_to_thc"]],
                        [network_config["pf_gis_to_wais"], network_config["pf_thc_to_wais"], 0.0]
                    ]
            else:
                coupl = network_config["coupl"]

            Tlim = np.array(network_config["Tlim"])
            
            A = d * np.array(coupl)
            c = np.ones((len(Tlim), 4))
            c[:,0] = 0.25
            c[:,1] = 0
            c[:,2] = -0.5
            c[:,3] = -(np.sqrt(4 / 27) * GMT / Tlim) + A.sum(1)

            network_config["A"] = A
            network_config["c"] = c

        elif network_config["type"] == "ditlevsen":

            h = np.array(network_config["h"])
            delta = network_config["delta"] if "delta" in network_config else 1
            c = np.ones((len(h), 4))
            c[:,0] = 4/(delta**4)
            c[:,1] = h/(delta**3)
            c[:,2] = -8/(delta**2)
            c[:,3] = -3*h/delta

            network_config["c"] = c
            network_config["A"] = np.array(network_config["A"])

        elif network_config["type"] == "custom":
            network_config["A"] = np.array(network_config["A"])
            network_config["c"] = np.array(network_config["c"])

        elif network_config["type"] == "vannes":
            network_config["Ps"] = np.linspace(network_config["Pstart"],network_config["Pend"],network_config["Psteps"],endpoint = True)
            network_config["c"] = np.array([0.0])
            network_config["A"] = np.array([[0.0]])
            network_config["taos"] = np.array([1.0])
        else:
            pass
            
        if network_config["mode"] in ["varyforce","fpt_orthant_varyforce"]:
            if network_config["type"] == "wunderling":
                GMTs = np.array(network_config["GMTs"])
                cs = np.stack(len(GMTs)*[network_config["c"]], axis = 0)
                for i in range(len(Tlim)):
                    cs[:,i,3] = -(np.sqrt(4 / 27) * GMTs / Tlim[i]) + A.sum(1)[i]
            
            elif network_config["type"] == "ditlevsen":
                hs = np.array(network_config["hs"])
                cs = np.stack(hs.shape[0]*[network_config["c"]], axis = 0)
                cs[:,:,1] = hs/(delta**3)
                cs[:,:,3] = -3*hs/delta

            else:
                cs = network_config["cs"]
            
            network_config["cs"] = np.array(cs)

    def parse_config(self, node_idx):
        
        with open(self.configpath, 'r') as fp:
            config = yaml.load(fp, Loader = yaml.FullLoader)

        experimentname = self.configpath.name.split(".")[0]

        experimentpath = Path(config["setup"]["outpath"])/experimentname

        config["setup"]["experimentname"] = experimentname
        config["setup"]["experimentpath"] = experimentpath

        zarrpath = experimentpath/f"{experimentname}.zarr"

        config["setup"]["zarrpath"] = zarrpath

        if node_idx == -1:
            return config

        baseconfig = config["base_config"]
        baseconfig["mode"] = config["setup"]["mode"]

        n_seeds = config["param_grid"]["n_seeds"] if "n_seeds" in config["param_grid"] else 0
        zip_grid = config["param_grid"]["zip"] if "zip" in config["param_grid"] else {}
        orthogonal_grid = config["param_grid"]["orthogonal"] if "orthogonal" in config["param_grid"] else {}

        across_chunk_axes = [a for a, c in zip(config["setup"]["axes"], config["setup"]["chunks"]) if c == 1]
        within_chunk_axes = [a for a, c in zip(config["setup"]["axes"], config["setup"]["chunks"]) if c != 1]

        if n_seeds > 0:
            orthogonal_grid["seed"] = list(range(n_seeds))

        curr_configs = [baseconfig]

        for curr_ax in within_chunk_axes:

            next_configs = []

            if curr_ax == "zip_id":
                for zip_id, values in enumerate(zip(*list(zip_grid.values()))):
                    for curr_config in curr_configs:
                        curr_config = deepcopy(curr_config)
                        for param, value in zip(list(zip_grid.keys()), values):
                            curr_config[param] = value
                        curr_config[zip_id] = zip_id
                        next_configs.append(curr_configs)
            # elif curr_ax == "seed":
            #     next_configs = curr_configs
            else:
                if curr_ax in orthogonal_grid:
                    for value in orthogonal_grid[curr_ax]:
                        for curr_config in curr_configs:
                            curr_config = deepcopy(curr_config)
                            curr_config[curr_ax] = value
                            next_configs.append(curr_config)
                else:
                    next_configs = curr_configs

            curr_configs = next_configs

        # if n_seeds > 0:
        #     next_configs = []
        #     for seed in range(n_seeds):
        #         for curr_config in curr_configs:
        #             curr_config = deepcopy(curr_config)
        #             curr_config["seed"] = seed
        #             next_configs.append(curr_config)
        #     curr_configs = next_configs


        curr_configs = [curr_configs]

        for curr_ax in across_chunk_axes:

            next_configs = []

            if curr_ax == "zip_id":
                for zip_id, values in enumerate(zip(*list(zip_grid.values()))):
                    for curr_chunk_configs in curr_configs:
                        next_chunk_configs = []
                        for curr_config in curr_chunk_configs:
                            curr_config = deepcopy(curr_config)
                            for param, value in zip(list(zip_grid.keys()), values):
                                curr_config[param] = value
                            curr_config["zip_id"] = zip_id
                            next_chunk_configs.append(curr_config)
                        next_configs.append(next_chunk_configs)
                
            else:
                if curr_ax in orthogonal_grid:
                    for value in orthogonal_grid[curr_ax]:
                        for curr_chunk_configs in curr_configs:
                            next_chunk_configs = []
                            for curr_config in curr_chunk_configs:
                                curr_config = deepcopy(curr_config)
                                curr_config[curr_ax] = value
                                next_chunk_configs.append(curr_config)
                            next_configs.append(next_chunk_configs)
                else:
                    next_configs = curr_configs

            curr_configs = next_configs

        for curr_chunk_configs in curr_configs:
            for curr_config in curr_chunk_configs:
                self.generate_network_params(curr_config)
        
        n_chunks = len(curr_configs)
        n_processes = config["setup"]["n_processes"]
        n_nodes = config["setup"]["n_nodes"]

        chunks_per_node = int(np.ceil(n_chunks / n_nodes))

        # configs = []
        # for i in range(n_nodes):
        #     configs.append(curr_configs[i*chunks_per_node:(i+1)*chunks_per_node])
        # configs is now a List of List of List, 1st depth is per node, 2nd per chunk 3rd list of configs

        config["network_configs"] = curr_configs[node_idx*chunks_per_node:(node_idx+1)*chunks_per_node]
        
        return config

    def build_zarr_array(self):
        zarrpath = self.config["setup"]["zarrpath"]
        zarrpath.parents[0].mkdir(exist_ok=True, parents=True)
        if zarrpath.exists():
            print(f"Warning! Zarr file under {zarrpath} already exists! Not writing again.")
            return 

        n_seeds = self.config["param_grid"]["n_seeds"] if "n_seeds" in self.config["param_grid"] else 0
        zip_grid = self.config["param_grid"]["zip"] if "zip" in self.config["param_grid"] else {}
        orthogonal_grid = self.config["param_grid"]["orthogonal"] if "orthogonal" in self.config["param_grid"] else {}

        coords = {}
        shape = tuple()
        for axis in self.config["setup"]["axes"]:
            if axis == "zip_id":
                if zip_grid:
                    coords[axis] = range(len(list(zip(*list(zip_grid.values())))))
                    for zip_axis in zip_grid:
                        # coords[zip_axis] = ("zip_id", zip_grid[zip_axis])
                        coords[zip_axis] = ("zip_id", [str(i) if not isinstance(i, float) else i for i in zip_grid[zip_axis]])
            elif axis == "seed":
                if n_seeds > 0:
                    coords[axis] = range(n_seeds)
            else:
                if axis in orthogonal_grid:
                    coords[axis] = np.array(sorted(orthogonal_grid[axis]))
                elif axis in ["t", "time"] and self.config["setup"]["mode"] in ["timeseries", "varyforce", "vannes"]:
                    dt = self.config["base_config"]["dt"] if "dt" in self.config["base_config"] else 10.
                    Tend = self.config["base_config"]["Tend"] if "Tend" in self.config["base_config"] else 100000
                    if isinstance(dt, list):
                        ts = np.insert(np.logspace(start = dt[0], stop = dt[1], num = dt[2], base = dt[3]), 0, 0.)
                    else:
                        N = int(Tend // dt) + 1
                        ts = np.arange(0, (N+3)*dt, dt)[:N]
                    
                    coords[axis] = ts
            
            if axis in coords:
                shape += (len(coords[axis]), )
            else:
                print(f"Warning: {axis} not in coords.")
        
        print("building dummy")
        ds = xr.Dataset(coords = coords)
        
        ds = ds.chunk(chunks={k: v for k, v in zip(self.config["setup"]["axes"], self.config["setup"]["chunks"])})

        ds.to_zarr(zarrpath)

        zarrgroup = zarr.open_group(str(zarrpath))

        compressor = Blosc(cname='lz4', clevel=1)

        for var in self.config["setup"]["dimension_names"]:
            newds = zarrgroup.create_dataset(var, shape = shape, chunks = self.config["setup"]["chunks"], dtype = 'float32', fillvalue = np.nan, compressor = compressor)
            newds.attrs['_ARRAY_DIMENSIONS'] = self.config["setup"]["axes"]

        zarr.convenience.consolidate_metadata(str(zarrpath))

    def write_slurmscript(self):

        experimentname = self.config["setup"]["experimentname"]
        experimentpath = self.config["setup"]["experimentpath"]
        slurmscript_path = experimentpath/f"{experimentname}.slurm"
        slurmout_path = str(experimentpath/f"{experimentname}-%A-%a.out")
        with open(slurmscript_path, "w+") as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --array=0-{int(self.config['setup']['n_nodes'])-1}\n")
            fh.writelines(f"#SBATCH --job-name {experimentname}\n")
            fh.writelines(f"#SBATCH -o {slurmout_path}\n")
            fh.writelines(f"#SBATCH -p standard\n")
            fh.writelines(f"#SBATCH --nodes=1\n")
            fh.writelines(f"#SBATCH --ntasks=1\n")
            fh.writelines(f"#SBATCH --cpus-per-task={int(self.config['setup']['n_processes']) if self.config['setup']['mode'] == 'fpt_orthant' else int(4*self.config['setup']['n_processes'])}\n")
            fh.writelines(f"source ~/.bashrc\n")
            fh.writelines(f"conda activate {self.config['setup']['condaenv']}\n")
            fh.writelines(f"cd {os.getcwd()}\n")
            fh.writelines(f"python {os.path.relpath(os.path.realpath(__file__), start = os.getcwd())} {str(self.configpath.resolve())} $SLURM_ARRAY_TASK_ID\n")

    @classmethod
    def run_experiment(cls, configpath, node_idx):

        self = cls(configpath, node_idx)

        if node_idx == -1:
            self.build_zarr_array()

            if self.config["setup"]["n_nodes"] > 0:
                self.write_slurmscript()
        else:

            print(f"Running experiment {self.config['setup']['experimentname']} on Node {node_idx}")

            network_configs = [{"setup": self.config["setup"], "network_configs": network_config, "process_id": i} for i, network_config in enumerate(self.config["network_configs"])]
            
            if self.config["setup"]["n_processes"] <= 1:
                for network_config in network_configs:
                    ExperimentClient.run_simulations(network_config)

            else:
                with ProcessPoolExecutor(max_workers = self.config["setup"]["n_processes"]) as pool:
                    _ = list(pool.map(ExperimentClient.run_simulations, network_configs))


if __name__ == "__main__":

    import fire

    fire.Fire(ExperimentHost.run_experiment)