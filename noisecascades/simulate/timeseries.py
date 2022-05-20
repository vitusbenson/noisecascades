
from numba import jit
import numpy as np

from noisecascades.simulate.step import semi_impl_euler_maruyama_coupled_doublewell_alphastable_step, euler_step, heun_step

@jit(nopython = True)
def simulate_timeseries_logdt(N, x, xs, t, ts, dt, dtao, c, A, L):

    xs[0,:] = x

    for i in range(1,N):

        x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao[:,i], c, A, L[i,:])

        xs[i,:] = x
        t += dt[i]
        ts[i] = t

    return ts, xs

@jit(nopython = True)
def simulate_timeseries(N, x, xs, t, ts, dt, dtao, c, A, L, method = "semi_impl"):

    xs[0,:] = x

    for i in range(1,N):

        if method == "euler":
            x = euler_step(x, dtao, c, A, L[i,:])
        elif method == "heun":
            x = heun_step(x, dtao, c, A, L[i,:])
        else:
            x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao, c, A, L[i,:])

        xs[i,:] = x
        t += dt
        ts[i] = t

    return ts, xs


@jit(nopython = True)
def simulate_timeseries_euler_noinit(N, x, xs, t, ts, dt, dtao, c, A, L):

    #xs[0,:] = x

    for i in range(N):

        x = euler_step(x, dtao, c, A, L[i,:])

        xs[i,:] = x
        t += dt
        ts[i] = t

    return ts, xs
