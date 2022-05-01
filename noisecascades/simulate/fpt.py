
import numpy as np
from numba import jit

from noisecascades.simulate.step import semi_impl_euler_maruyama_coupled_doublewell_alphastable_step

@jit(nopython = True)
def simulate_fpt_orthant(N, x, t, dt, dtao, c, A, L, boundary = 0.0):

    for i in range(1,N):

        x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao, c, A, L[i,:])

        t += dt

        if np.any(x >= boundary):
            return t, x >= boundary, x

    return t, np.full_like(x, False, dtype = 'bool'), x
