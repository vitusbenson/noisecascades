
# Plan:
# Start this with Nicos Potential...
# But later: Repeat with my new custom piecewise double-well... this has the benefit that the potential well does not get smaller when approaching the bifurcation. Instead it just gets flatter...
# 1) Repeat Boers AMOC Methodology on double-well potential /w gauss noise, i.e.:
#   - Simulate timeseries /w slow increase in parameter c
#   - Calculate Var, AR1, lambda until short before tipping
#   - Create Fourier surrogates + get significance levels
#   - ACHTUNG HIER HABE ICH DAS MISSVERSTANDEN! smooth ews (fill small gaps) + label: iff up until end of timeseries existed EWS -> TP. iff up until end of timeseries no EWS -> FN. Iff ews earlier -> FP 
# -> Recreate Boers Fig.1: window-size T=2000, stddev_G = 0.2 (i.e. sigma_G = 0.2/sqrt(2)), Tipping at T=7000.
#   - repeat this 10k times or so
#   - plot precision & recall. And of course also timeseries themselves...
# 2) Repeat the above for levy:
#   - For each levy alpha: take levy sigma that is consistent with 1 gauss sigma in terms of FPT.
#   - Take extra look at levy alphas close to 2.
#   - Repeat: take levy sigma such that gaussian part of noise keeps same strength (see Imkeller&P....)
# 3) Repeat the above for autocorrelated noise

import numpy as np
from numba import jit
import xarray as xr
import scipy.stats as st

# Variance:
# 1. Remove trend + offset over time in window
# 2. Compute StdDev
# AR1:
# 1. Remove trend + offset over time in window
# 2. Compute Autocorrelation of t and t-1 in window
# lambda:
# 1. Remove trend + offset over time in window
# 2. Compute dx/dt = np.diff(x)
# 3. Regress dx/dt on x, get slope

# Note: Boers removes nonlinear x^3 fit (aka the "deterministic" evolution).. Can actually do the same if we want to, but should not matter much

# Next steps:
# - use boers setting
# - only look at EWS until shortly before transition
# - significance test: analyse linear trend from t=0 to shortly before first passage time essentially, repeat this many times (aka many seeds). Then count number of times it worked... -> resulting fraction will be 1 at alpha=2.0 and probably decrease for lower alpha...
# - significance test: take not first passage, but deterministic tipping time... check again. 
# - Think about a way to get false positive rate...

# Have versions of functions that do not handle the extra dimensions, then use apply_ufunc + dask = parallelized.... this might help with memory requirements
# Write script + save fpts, ews, pvals for the gauss zarr into extra zarrs. 
# Redo simulations for gauss noise with sigma_G = 0.2/sqrt(2) (can do this together with the levy noise sims below)
# Make simulations for levy noise, with sigma_L matched to sigma_G as above.
# Plot: pvals over alpha
# Plot: Fig1 of Boers for different alphas, seeds

def detrend_nonvec(x):
    t = np.linspace(0,1,len(x))
    return x - np.polyval(np.polyfit(t[~np.isnan(x)], x[~np.isnan(x)], 1),t)

def std_window_detrended_nonvec(x):
    x_detrend = detrend_nonvec(x)
    return np.nanstd(x_detrend)

def ar1_window_detrended_nonvec(x):
    x_detrend = detrend_nonvec(x)
    x_t_centered = x_detrend[1:] - x_detrend[1:].mean()
    x_tm1_centered = x_detrend[:-1] - x_detrend[:-1].mean()
    return np.ma.corrcoef(np.ma.array(x_t_centered, mask = np.isnan(x_t_centered)), np.ma.array(x_tm1_centered, mask = np.isnan(x_tm1_centered)))[0,1]

def lambda_window_detrended_nonvec(x):
    x_detrend = detrend_nonvec(x)
    dxdt = np.diff(x_detrend)
    return np.polyfit(x_detrend[:-1][~np.isnan(dxdt)], dxdt[~np.isnan(dxdt)], 1)[0]


def detrend(x, axis):
    if not isinstance(axis, int):
        axis = axis[0]
    t = np.linspace(0, 1, x.shape[axis])
    x_moved = np.moveaxis(x, axis, 0)

    mask = np.isnan(x_moved)
    n = np.maximum(np.count_nonzero(~mask, axis = 0), 1)
    x_moved[mask] = 0.0

    #n = x.shape[axis]
    Sab = np.einsum('i,i...->...',t, x_moved)
    #Sab = np.dot(x_detrend_moved[:, None,...].T, dxdt_moved[None, :,...].T)
    Saa = np.einsum('i,i->...',t, t)
    #Saa = np.dot(x_detrend_moved[:, None,...].T, x_detrend_moved[None, :,...].T)
    Sa = t.sum(0)
    Sb = x_moved.sum(0)

    beta = (n*Sab - Sa*Sb)/(n*Saa - Sa**2  + 1e-6)
    alpha = Sb/n - beta * Sa/n

    x_moved[mask] = np.NaN

    return np.moveaxis(x_moved - (beta[None,:].T * t).T - alpha, 0, axis)
    
    # s_moved = x_moved.shape
    # x_reshaped = x_moved.reshape(-1,s_moved[-1]).T
    # p = np.polyfit(t, x_reshaped, 1)
    # return np.moveaxis((x_reshaped - (p[0][:,None]*t).T - p[1]).T.reshape(*s_moved), -1, axis)

def std_window_detrended(x, axis):
    if not isinstance(axis, int):
        axis = axis[0]
    x_detrend = detrend(x, axis)
    return np.where(np.count_nonzero(np.isnan(x), axis = axis) == 0, np.nanstd(x_detrend, axis = axis), np.NaN)

def ar1_window_detrended(x, axis):
    if not isinstance(axis, int):
        axis = axis[0]
    x_detrend = detrend(x, axis)
    x_detrend_moved = np.moveaxis(x_detrend, axis, 0)
    x_t_centered = x_detrend_moved[1:] - x_detrend_moved[1:].mean(axis = 0)
    x_tm1_centered = x_detrend_moved[:-1] - x_detrend_moved[:-1].mean(axis = 0)

    var_t = np.nanvar(x_t_centered, axis = 0)
    var_tm1 = np.nanvar(x_tm1_centered, axis = 0)

    x_t_centered[np.isnan(x_t_centered)] = 0.0
    x_tm1_centered[np.isnan(x_tm1_centered)] = 0.0
    N = np.count_nonzero((~np.isnan(x_t_centered)) & (~np.isnan(x_t_centered)), axis = 0)
    covar = np.einsum('i...,i...->...', x_t_centered, x_tm1_centered)/(N - 1)
    #covar = np.dot(x_t_centered[:, None,...].T, x_t_centered[None, :,...].T).T/(x.shape[axis[0]] - 2)
    return np.where(np.count_nonzero(np.isnan(x), axis = axis) == 0, covar/(np.sqrt(var_t*var_tm1) + 1e-6), np.NaN)
    #np.corrcoef(x_detrend[1:], x_detrend[:-1])[0,1]

def lambda_window_detrended(x, axis):
    if not isinstance(axis, int):
        axis = axis[0]
    x_detrend = detrend(x, axis)
    dxdt = np.diff(x_detrend, axis = axis)
    x_detrend_moved = np.moveaxis(x_detrend, axis, 0)[:-1]
    dxdt_moved = np.moveaxis(dxdt, axis, 0)
    n = np.count_nonzero((~np.isnan(x_detrend_moved)) & (~np.isnan(dxdt_moved)), axis = 0)
    x_detrend_moved[np.isnan(x_detrend_moved)] = 0.0
    dxdt_moved[np.isnan(dxdt_moved)] = 0.0

    Sab = np.einsum('i...,i...->...',x_detrend_moved, dxdt_moved)
    #Sab = np.dot(x_detrend_moved[:, None,...].T, dxdt_moved[None, :,...].T)
    Saa = np.einsum('i...,i...->...',x_detrend_moved, x_detrend_moved)
    #Saa = np.dot(x_detrend_moved[:, None,...].T, x_detrend_moved[None, :,...].T)
    Sa = x_detrend_moved.sum(0)
    Sb = dxdt_moved.sum(0)

    return np.where(np.count_nonzero(np.isnan(x), axis = axis) == 0,  (n*Sab - Sa*Sb)/(n*Saa - Sa**2 + 1e-6), np.NaN)

    # p = np.polyfit(x_detrend[:-1], dxdt)
    # return p[0]


@jit(nopython=True)
def first_passage_index(vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if vec[i] >= 0:
            return i
    return -1 

def get_fpt(cube, time_axis = "t"):

    first_idx = xr.apply_ufunc(first_passage_index, cube, input_core_dims=[[time_axis]],vectorize = True, dask = 'parallelized').compute()

    first_idx_sel = xr.where(first_idx == -1, len(cube[time_axis])-1, first_idx)

    FPT = cube[time_axis].isel({time_axis: (first_idx_sel.to_array("vars").min("vars"))}).drop([time_axis])#.rename("FPT").to_dataset()

    FPT = xr.where(first_idx.to_array("vars").max("vars") == -1, np.nan, FPT).to_dataset(name = "FPT")

    FPOktant = (((first_idx_sel.to_array("vars") - first_idx_sel.to_array("vars").min("vars")) <= 0)*1).astype(str).str.join("vars", "")#.to_dataset(name = "Oktant")

    FPOktant = xr.where(first_idx.to_array("vars").max("vars") == -1, '000', FPOktant).to_dataset(name = "Oktant")


    return xr.merge([FPT, FPOktant]).drop_vars(["vars"], errors = "ignore")



def fourrier_surrogates(ts, ns):
    # From https://github.com/niklasboers/AMOC_EWS/blob/main/EWS_functions.py
    ts_fourier  = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts

def kendall_tau_test(ts, n_surrogates, surrogate_type = 'fourier', statistic_type = 'linear'):
    # From https://github.com/niklasboers/AMOC_EWS/blob/main/EWS_functions.py
    ts = ts[~np.isnan(ts)]
    tlen = ts.shape[0]

    if surrogate_type == 'fourier':
        tsf = ts - ts.mean()
        nts = fourrier_surrogates(tsf, n_surrogates)
    # elif mode1 == 'shuffle':
    #     nts = shuffle_surrogates(ts, ns)
    stat = np.zeros(n_surrogates)
    tlen = nts.shape[1]
    if statistic_type == 'linear':
        tau = st.linregress(np.arange(ts.shape[0]), ts)[0]
        for i in range(n_surrogates):
            stat[i] = st.linregress(np.arange(tlen), nts[i])[0]
    elif statistic_type == 'kt':
        tau = st.kendalltau(np.arange(ts.shape[0]), ts)[0]
        for i in range(n_surrogates):
            stat[i] = st.kendalltau(np.arange(tlen), nts[i])[0]
    p = 1 - st.percentileofscore(stat, tau) / 100.
    return p

def get_ews(cube, window_size = 500000, downsample_factor = 10000, var_axis = "x", time_axis = "t", use_vectorized = True):

    rolling_cube = cube[var_axis].rolling({time_axis: window_size}).construct("window").isel({time_axis: slice(window_size,-window_size,downsample_factor)}).chunk({time_axis:10, "window":-1})
    
    if use_vectorized:
        std = rolling_cube.reduce(std_window_detrended, dim = "window").to_dataset(name = "StdDev")
        ar1 = rolling_cube.reduce(ar1_window_detrended, dim = "window").to_dataset(name = "AR1")
        l = rolling_cube.reduce(lambda_window_detrended, dim = "window").to_dataset(name = "Lambda")
    else:
        std = xr.apply_ufunc(std_window_detrended_nonvec, rolling_cube, input_core_dims=[["window"]], vectorize = True, dask = "parallelized", output_dtypes=["float64"])
        ar1 = xr.apply_ufunc(ar1_window_detrended_nonvec, rolling_cube, input_core_dims=[["window"]], vectorize = True, dask = "parallelized", output_dtypes=["float64"])
        l = xr.apply_ufunc(lambda_window_detrended_nonvec, rolling_cube, input_core_dims=[["window"]], vectorize = True, dask = "parallelized", output_dtypes=["float64"])

    return xr.merge([std, ar1, l])

def get_significance_for_ews(ews, n_surrogates = 10000, time_axis = "t"):

    pvals = xr.apply_ufunc(kendall_tau_test, ews.chunk({time_axis:-1}), input_core_dims=[[time_axis]], vectorize = True, dask = "parallelized", kwargs = {"n_surrogates": n_surrogates}, output_dtypes = ["float64"])

    pvals = pvals.to_array("vars")
    pvals["vars"] = [f"p_{b}" for b in pvals.vars.values]
    pvals = pvals.to_dataset("vars")

    return pvals