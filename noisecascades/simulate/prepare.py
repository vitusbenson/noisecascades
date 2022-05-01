
import numpy as np
from scipy.stats import levy_stable

from noisecascades.simulate.timeseries import simulate_timeseries
from noisecascades.simulate.fpt import simulate_fpt_orthant

class Integrator:

    def __init__(self, seed = None):
        self.rng = np.random.default_rng(seed)
        

    @staticmethod
    def parse_inputs(dt, Tend, x_init, c, A, alphas, sigmas, taos):

        x_init = np.array(x_init).astype(np.float64)
        taos = np.array(taos).astype(np.float64)
        c = np.array(c).astype(np.float64)
        A = np.array(A).astype(np.float64)

        n = len(x_init)
        N = int(Tend // dt) + 1
        dtao = dt / taos

        if not isinstance(alphas, np.ndarray):
            if not isinstance(alphas, list):
                alphas = [alphas]
            alphas = np.array(alphas)
        if len(alphas) < n:
            alphas = np.repeat(alphas, n//len(alphas) + 1)[:n]
        
        alphas = alphas.astype(np.float64)

        if not isinstance(sigmas, np.ndarray):
            if not isinstance(sigmas, list):
                sigmas = [sigmas]
            sigmas = np.array(sigmas)
        if len(sigmas) < n:
            sigmas = np.repeat(sigmas, n//len(sigmas) + 1)[:n]

        sigmas = sigmas.astype(np.float64)

        return dt, Tend, x_init, c, A, alphas, sigmas, taos, n, N, dtao
    
    def generate_noise(self, n, N, dtao, alphas, sigmas):

        L = np.zeros((N,n))
        for i in range(n):
            if sigmas[i] > 0:
                L[:,i] = sigmas[i] * (dtao[i] ** (1/alphas[i])) * levy_stable.rvs(alphas[i], 0.0, size=N, random_state = self.rng)
        
        L[np.abs(L) > 1e12] = (np.sign(L) * 1e12)[np.abs(L) > 1e12]
        L[np.isinf(L)] = (np.sign(L) * 1e12)[np.isinf(L)]
        L[np.isnan(L)] = 0.0

        return L

    @classmethod
    def integrate_networkconfig(cls, config):

        return cls.integrate_network(**config)

    @classmethod
    def integrate_network(cls, mode, dt, Tend, x_init, c, A, alphas, sigmas, taos, seed = None, **kwargs):

        self = cls(seed)

        dt, Tend, x_init, c, A, alphas, sigmas, taos, n, N, dtao = self.parse_inputs(dt, Tend, x_init, c, A, alphas, sigmas, taos)

        if mode == "timeseries":
            
            L = self.generate_noise(n, N, dtao, alphas, sigmas)

            xs = np.zeros((N,n))
            t = 0.0
            ts = np.zeros(N)

            ts, xs = simulate_timeseries(N, x_init, xs, t, ts, dt, dtao, c, A, L)

            return ts, xs

        elif mode == "fpt_orthant":

            # TODO: Calculate boundary from c (i.e. get middle sattlepoint from c)
            
            x = x_init
            t = 0.0

            n_chunks = int(np.ceil(N/1000))

            for i_chunk in range(n_chunks):

                N_chunk = min(1000, N - i_chunk * 1000)

                L_chunk = self.generate_noise(n, N_chunk, dtao, alphas, sigmas)

                t, orthant_entered, x = simulate_fpt_orthant(N_chunk, x, t, dt, dtao, c, A, L_chunk, boundary = 0.0)

                if np.any(orthant_entered[:3]):
                    return t, orthant_entered
            
            return np.NaN, orthant_entered
                





