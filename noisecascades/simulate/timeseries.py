
from numba import jit
import numpy as np

from noisecascades.simulate.step import semi_impl_euler_maruyama_coupled_doublewell_alphastable_step, euler_step, heun_step, semi_impl_cased_step

@jit(nopython = True)
def simulate_timeseries_logdt(N, x, xs, t, ts, dt, dtao, c, A, L, method = "semi_impl_cased"):

    xs[0,:] = x

    for i in range(1,N):
        
        if method == "euler":
            x = euler_step(x, dtao[:,i], c, A, L[i,:])
        elif method == "heun":
            x = heun_step(x, dtao[:,i], c, A, L[i,:])
        elif method == "semi_impl":
            x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao[:,i], c, A, L[i,:])
        else:
            x = semi_impl_cased_step(x, dtao[:,i], c, A, L[i,:])

        xs[i,:] = x
        t += dt[i]
        ts[i] = t

    return ts, xs

@jit(nopython = True)
def simulate_timeseries(N, x, xs, t, ts, dt, dtao, c, A, L, method = "semi_impl_cased"):

    xs[0,:] = x

    for i in range(1,N):

        if method == "euler":
            x = euler_step(x, dtao, c, A, L[i,:])
        elif method == "heun":
            x = heun_step(x, dtao, c, A, L[i,:])
        elif method == "semi_impl":
            x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao, c, A, L[i,:])
        else:
            x = semi_impl_cased_step(x, dtao, c, A, L[i,:])
            

        xs[i,:] = x
        t += dt
        ts[i] = t

    return ts, xs


@jit(nopython = True)
def simulate_timeseries_noinit(N, x, xs, t, ts, dt, dtao, c, A, L, method = "semi_impl_cased"):

    #xs[0,:] = x

    for i in range(N):

        if method == "euler":
            x = euler_step(x, dtao, c, A, L[i,:])
        elif method == "heun":
            x = heun_step(x, dtao, c, A, L[i,:])
        elif method == "semi_impl":
            x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao, c, A, L[i,:])
        else:
            x = semi_impl_cased_step(x, dtao, c, A, L[i,:])

        xs[i,:] = x
        t += dt
        ts[i] = t

    return ts, xs
