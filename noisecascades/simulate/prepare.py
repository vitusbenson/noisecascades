
import numpy as np
from scipy.stats import levy_stable

from noisecascades.simulate.timeseries import simulate_timeseries, simulate_timeseries_logdt, simulate_timeseries_noinit
from noisecascades.simulate.fpt import simulate_fpt_orthant, simulate_fpt_orthant_logdt

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

        if isinstance(dt, list):
            dt = np.diff(np.logspace(start = dt[0], stop = dt[1], num = dt[2], base = dt[3]),prepend = [0,0])
            N = len(dt)
            dtao = (dt[:,None] / taos).T
        else:
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
                if alphas[i] == 2.0:
                    L[:,i] = sigmas[i] * (dtao[i] ** (1/2)) * np.sqrt(2) * self.rng.standard_normal(N)
                else:
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

            if len(dtao.shape) > 1:
                ts, xs = simulate_timeseries_logdt(N, x_init, xs, t, ts, dt, dtao, c, A, L, method = kwargs["int_method"] if "int_method" in kwargs else "semi_impl_cased")
            else:
                ts, xs = simulate_timeseries(N, x_init, xs, t, ts, dt, dtao, c, A, L, method = kwargs["int_method"] if "int_method" in kwargs else "semi_impl_cased")

            return ts, xs

        elif mode == "varyforce":

            cs = kwargs["cs"]
            n_chunks = cs.shape[0]
            N_chunk = N // n_chunks

            x = x_init
            xs = np.full((N,n), np.NaN)
            #xs[0,:] = x
            t = 0.0
            ts = np.zeros(N)

            for i_chunk in range(n_chunks):
                L_chunk = self.generate_noise(n, N_chunk, dtao, alphas, sigmas)
                
                #breakpoint()

                #ts[i_chunk*N_chunk+1:(i_chunk+1)*N_chunk+1], xs[i_chunk*N_chunk+1:(i_chunk+1)*N_chunk+1] = simulate_timeseries_euler_noinit(N_chunk, x, xs[i_chunk*N_chunk+1:(i_chunk+1)*N_chunk+1], t, ts[i_chunk*N_chunk+1:(i_chunk+1)*N_chunk+1], dt, dtao, cs[i_chunk,:,:], A, L_chunk)

                ts[i_chunk*N_chunk:(i_chunk+1)*N_chunk], xs[i_chunk*N_chunk:(i_chunk+1)*N_chunk] = simulate_timeseries(N_chunk, x, xs[i_chunk*N_chunk:(i_chunk+1)*N_chunk], t, ts[i_chunk*N_chunk:(i_chunk+1)*N_chunk], dt, dtao, cs[i_chunk,:,:], A, L_chunk, method = kwargs["int_method"] if "int_method" in kwargs else "semi_impl_cased")

                x = xs[(i_chunk+1)*N_chunk -1]

            return ts, xs

        elif mode == "fpt_orthant":

            # TODO: Calculate boundary from c (i.e. get middle sattlepoint from c)
            
            x = x_init
            t = 0.0

            n_chunks = int(np.ceil(N/1000))

            for i_chunk in range(n_chunks):

                N_chunk = min(1000, N - i_chunk * 1000)
                
                if len(dtao.shape) > 1:
                    L_chunk = self.generate_noise(n, N_chunk, dtao[:,i_chunk*1000:(i_chunk+1)*1000], alphas, sigmas)
                    
                    t, orthant_entered, x = simulate_fpt_orthant_logdt(N_chunk, x, t, dt[i_chunk*1000:(i_chunk+1)*1000], dtao[:,i_chunk*1000:(i_chunk+1)*1000], c, A, L_chunk, boundary = 0.0, method = kwargs["int_method"] if "int_method" in kwargs else "semi_impl_cased")
                else:
                    L_chunk = self.generate_noise(n, N_chunk, dtao, alphas, sigmas)
                    t, orthant_entered, x = simulate_fpt_orthant(N_chunk, x, t, dt, dtao, c, A, L_chunk, boundary = 0.0, method = kwargs["int_method"] if "int_method" in kwargs else "semi_impl_cased")

                if np.any(orthant_entered):
                    return t, orthant_entered
            
            return np.NaN, orthant_entered
                





