
from numba import jit

from noisecascades.simulate.step import semi_impl_euler_maruyama_coupled_doublewell_alphastable_step

@jit(nopython = True)
def simulate_timeseries(N, x, xs, t, ts, dt, dtao, c, A, L):

    xs[0,:] = x

    for i in range(1,N):

        x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao, c, A, L[i,:])

        xs[i,:] = x
        t += dt
        ts[i] = t

    return ts, xs
