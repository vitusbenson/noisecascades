

import numpy as np
from numba import jit

@jit(nopython = True)
def vannes_euler_step(T, P, L, dt):
    dT = dt * (P/(0.5+P)*0.3*T*(1-T/90) - 0.15*T*10/(T+10) - 0.11*T*64**7/(64**7+T**7))
    xsol = T + dT
    nextT = xsol + L
    if nextT < 0.0:
        nextT = 0.0
    elif nextT > 100.0:
        nextT = 100.0
    return nextT

@jit(nopython = True)
def euler_step(x, dtao, c, A, L):
    x_new = x.copy()        

    if A.sum() != 0.0:
        transp = (A @ x)
    else:
        transp = 0*x
        
    for j in range(len(x)):
        if (not np.isnan(x[j])) and (not np.isinf(x[j])):           
            drift = x[j] - (4*c[j,0]*x[j]**3 + 3*c[j,1]*x[j]**2 + 2*c[j,2]*x[j] + (c[j,3] - transp[j]))*dtao[j]
        else:
            drift = x[j]

        x_new[j] = drift + L[j]
    
    return x_new

@jit(nopython = True)
def heun_step(x, dtao, c, A, L):

    drift_first = x.copy()
    x_first = x.copy()
    drift_second = x.copy()
    x_new = x.copy()

    if A.sum() != 0.0:
        transp_first = (A @ x)
    else:
        transp_first = 0*x
    
    for j in range(len(x)):
        if (not np.isnan(x[j])) and (not np.isinf(x[j])):
            drift_first[j] = (- (4*c[j,0]*x[j]**3 + 3 * c[j,1] * x[j]**2 + 2*c[j,2]*x[j] + c[j,3]) + transp_first[j])*dtao[j]
            x_first[j] = x[j] + drift_first[j]
        else:
            drift_first[j] = 0
            x_first[j] = x[j]
    
    if A.sum() != 0.0:
        transp_second = (A @ x_first)
    else:
        transp_second = 0*x

    for j in range(len(x)):
        if (not np.isnan(x[j])) and (not np.isinf(x[j])):
            drift_second[j] = (- (4*c[j,0]*x_first[j]**3 + 3 * c[j,1] * x_first[j]**2 + 2*c[j,2]*x_first[j] + c[j,3]) + transp_second[j])*dtao[j]
        else:
            drift_second[j] = 0
        
        x_new[j] = x[j] + (drift_first[j] + drift_second[j])/2 + L[j]
    
    return x_new

@jit(nopython = True)
def semi_impl_euler_maruyama_coupled_doublewell_alphastable_step(x, dtao, c, A, L):
    """
    V_i(x_i) = c_{i0}*x_i^4 + c_{i1}*x_i^3 + c_{i2}*x_i^2 + c_{i3}*x_i
    dx_i = (- V_i'(x_i) + A@x)*dt/t_i + s_i * dL_{dt/t_i}^{a_i}
    V_i'(x_i) = 4c_{i0}*x_i^3 + 3c_{i1}*x_i^2 + 2*c_{i2}*x_i + c_{i3}
    in 1D semi implicit part for potential:
    x_{n+1} = x_{n} - (4c_{i0}*x_{n+1}^3 + 3c_{i1}*x_{n+1}^2 + 2*c_{i2}*x_{n+1} + c_{i3} + A@x)dt/t_i
    <=> roots(4c_{i0}dt/t_i, 3c_{i1}dt/t_i, 1 + 2*c_{i2}dt/t_i, -x_{n} + (c_{i3} + A@x)dt/t_i)
    """
    x_new = x.copy()        

    if A.sum() != 0.0:
        transp = (A @ x)
    else:
        transp = 0*x
        
    for j in range(len(x)):
        if (not np.isnan(x[j])) and (not np.isinf(x[j])):           
            candidates = np.roots(np.array([4*c[j,0]*dtao[j], 3*c[j,1]*dtao[j], 1.0 + 2*c[j,2]*dtao[j], - x[j] + (c[j,3] - transp[j])*dtao[j]]).astype(np.complex128))
            if (np.abs(candidates.imag)<1e-5).sum() == 1:
                drift = (candidates[np.abs(candidates.imag)<1e-5][0]).real 
            else:
                best_idx = np.argmin(np.abs(candidates - x[j]))
                drift = (candidates[best_idx]).real
        else:
            drift = x[j]

        x_new[j] = drift + L[j]
    
    return x_new


@jit(nopython = True)
def estimate_cased_stability(c, dtao):
    limits = np.zeros((c.shape[0], 2))
    for i in range(c.shape[0]):
        dt = dtao[i]
        dV = np.array([-4*c[i,0], -3*c[i,1], -2*c[i,2], -c[i,3]]).astype(np.complex128)
        roots_cand = np.roots(dV)
        roots_cand = roots_cand[(np.abs(roots_cand.imag)<1e-5)]
        roots_cand = np.sort(np.unique(roots_cand.real))
        if len(roots_cand) == 3:
            lower_lim = np.roots(dt * dV + np.array([0,0,1,-roots_cand[0]])).real
            if len(lower_lim[(lower_lim>roots_cand[0]+1e-5) & (lower_lim<roots_cand[1]-1e-5)]) > 0:
                lower_lim = lower_lim[(lower_lim>roots_cand[0]+1e-5) & (lower_lim<roots_cand[1]-1e-5)].max()
                lower_lim = roots_cand[1] + 0.95 * (lower_lim - roots_cand[1])#min(roots_cand[1], 0.95*lower_lim)
            elif len(lower_lim[lower_lim<roots_cand[0]-1e-5]) > 0:
                lower_lim = lower_lim[lower_lim<roots_cand[0]-1e-5].max()
                lower_lim = roots_cand[0] + 0.95 * (lower_lim - roots_cand[0])#min(roots_cand[0], 0.95*lower_lim)
            else:
                lower_lim = roots_cand[0]
                
            upper_lim = np.roots(dt * dV + np.array([0,0,1,-roots_cand[2]])).real
            if len(upper_lim[(upper_lim > roots_cand[1]+1e-5) & (upper_lim < roots_cand[2]-1e-5)]) > 0:
                upper_lim = upper_lim[(upper_lim > roots_cand[1]+1e-5) & (upper_lim < roots_cand[2]-1e-5)].min()
                upper_lim = roots_cand[1] + 0.95 * (upper_lim - roots_cand[1]) #max(roots_cand[1], 0.95*upper_lim)
            elif len(upper_lim[upper_lim>roots_cand[2]+1e-5]) > 0:
                upper_lim = upper_lim[upper_lim>roots_cand[2]+1e-5].min()
                upper_lim = roots_cand[2] + 0.95 * (upper_lim - roots_cand[2]) #max(roots_cand[2], 0.95*upper_lim)
            else:
                upper_lim = roots_cand[2]
        else:
            lower_lim = np.roots(dt * dV + np.array([0,0,1,-roots_cand[0]])).real
            lower_lim = lower_lim[lower_lim < roots_cand[0]-1e-5]
            if len(lower_lim) > 0:
                lower_lim = lower_lim.max()
                lower_lim = roots_cand[0] + 0.95 * (lower_lim - roots_cand[0]) #min(roots_cand[0], 0.95*lower_lim)
            else:
                lower_lim = roots_cand[0]
            upper_lim = np.roots(dt * dV + np.array([0,0,1,-roots_cand[0]])).real
            upper_lim = upper_lim[upper_lim > roots_cand[0]+1e-5]
            if len(upper_lim) > 0:
                upper_lim = upper_lim.min()
                upper_lim = roots_cand[0] + 0.95 * (upper_lim - roots_cand[0]) #max(roots_cand[0], 0.95*upper_lim)
            else:
                upper_lim = roots_cand[0]

        limits[i,0] = lower_lim
        limits[i,1] = upper_lim
    
    return limits

@jit(nopython = True)
def semi_impl_cased_step(x, dtao, c, A, L, limits):
    """
    V_i(x_i) = c_{i0}*x_i^4 + c_{i1}*x_i^3 + c_{i2}*x_i^2 + c_{i3}*x_i
    dx_i = (- V_i'(x_i) + A@x)*dt/t_i + s_i * dL_{dt/t_i}^{a_i}
    V_i'(x_i) = 4c_{i0}*x_i^3 + 3c_{i1}*x_i^2 + 2*c_{i2}*x_i + c_{i3}
    in 1D semi implicit part for potential:
    x_{n+1} = x_{n} - (4c_{i0}*x_{n+1}^3 + 3c_{i1}*x_{n+1}^2 + 2*c_{i2}*x_{n+1} + c_{i3} + A@x)dt/t_i
    <=> roots(4c_{i0}dt/t_i, 3c_{i1}dt/t_i, 1 + 2*c_{i2}dt/t_i, -x_{n} + (c_{i3} + A@x)dt/t_i)
    """
    x_new = x.copy()        

    if A.sum() != 0.0:
        transp = (A @ x)
    else:
        transp = 0*x
        
    for j in range(len(x)):
        # if dtao[j] > 0.1:
        #     case_boundary = 1.0
        # else:
        #     case_boundary = 0.9*(-dtao[j] + np.sqrt(dtao[j]**2 + dtao[j]))/(2*dtao[j])
        # if np.abs(x[j]) <= case_boundary:
        if (x[j] > limits[j,0]) and (x[j] < limits[j,1]):
            drift = x[j] - (4*c[j,0]*x[j]**3 + 3*c[j,1]*x[j]**2 + 2*c[j,2]*x[j] + (c[j,3] - transp[j]))*dtao[j]
        elif (not np.isnan(x[j])) and (not np.isinf(x[j])):           
            candidates = np.roots(np.array([4*c[j,0]*dtao[j], 3*c[j,1]*dtao[j], 1.0 + 2*c[j,2]*dtao[j], - x[j] + (c[j,3] - transp[j])*dtao[j]]).astype(np.complex128))
            if (np.abs(candidates.imag)<1e-5).sum() == 1:
                drift = (candidates[np.abs(candidates.imag)<1e-5][0]).real 
            else:
                best_idx = np.argmin(np.abs(candidates - x[j]))
                drift = (candidates[best_idx]).real
        else:
            drift = x[j]

        x_new[j] = drift + L[j]
    
    return x_new


