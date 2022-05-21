
import numpy as np
from numba import jit

from noisecascades.simulate.step import semi_impl_euler_maruyama_coupled_doublewell_alphastable_step, euler_step, heun_step, semi_impl_cased_step

@jit(nopython = True)
def simulate_fpt_orthant_logdt(N, x, t, dt, dtao, c, A, L, boundary = 0.0, method = "semi_impl_cased"):
    
    
    for i in range(1,N):

        if method == "euler":
            x = euler_step(x, dtao[:,i], c, A, L[i,:])
        elif method == "heun":
            x = heun_step(x, dtao[:,i], c, A, L[i,:])
        elif method == "semi_impl":
            x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao[:,i], c, A, L[i,:])
        else:
            x = semi_impl_cased_step(x, dtao[:,i], c, A, L[i,:])

        t += dt[i]

        if np.any(x >= boundary):
            return t, x >= boundary, x

    return t, np.full_like(x, False, dtype = 'bool'), x


@jit(nopython = True)
def simulate_fpt_orthant(N, x, t, dt, dtao, c, A, L, boundary = 0.0, method = "semi_impl_cased"):
    
    for i in range(1,N):

        if method == "euler":
            x = euler_step(x, dtao, c, A, L[i,:])
        elif method == "heun":
            x = heun_step(x, dtao, c, A, L[i,:])
        elif method == "semi_impl":
            x = semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao, c, A, L[i,:])
        else:
            x = semi_impl_cased_step(x, dtao, c, A, L[i,:])

        t += dt

        if np.any(x >= boundary):
            return t, x >= boundary, x

    return t, np.full_like(x, False, dtype = 'bool'), x
