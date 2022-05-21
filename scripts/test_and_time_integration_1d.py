import numpy as np
import time
import noisecascades as nc
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from matplotlib.colors import LogNorm
import os

def integrate(method, c, dt, alpha, sigma, seed):
    starttime = time.perf_counter()
    ts, xs = nc.Integrator.integrate_network("timeseries", dt, 100, [-1.], [c], [[0.0]], [alpha], [sigma], [1.], seed, int_method = method)
    elapsed_time = time.perf_counter() - starttime
    was_stable = ~np.any(np.isnan(xs) | np.isinf(xs))
    return elapsed_time, was_stable


# all_data = []
# for dt in [0.01]:#[1.0, 0.1, 0.01, 0.001]:
#     for alpha in [0.01, 0.1, 0.25, 0.5, 1.5]:#[0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 1.9, 1.99]:
#         for sigma in [1., 1e-2, 1e-6]:#np.logspace(1, -8, 5, base = 10):#np.logspace(1, -8, 10, base = 10):
#             print(f"dt: {dt}, alpha: {alpha}, sigma: {sigma}")
#             for seed in tqdm(range(20)):
#                 for method in ["euler", "heun", "semi_impl", "semi_impl_cased"]:
#                     for c in [0.0, -0.5]:#np.linspace(0.0, -0.6, 7):
#                         elapsed_time, was_stable = integrate(method, [0.25, 0.0, -0.5, c], dt, alpha, sigma, seed)
#                         all_data.append({
#                             "dt": dt,
#                             "method": method,
#                             "alpha": alpha,
#                             "sigma": sigma,
#                             "seed": seed,
#                             "param": c,
#                             "potential": "wunderling",
#                             "stable": was_stable,
#                             "time": elapsed_time
#                         })
#                     for h in [0.0, 6.0]:#np.linspace(0.0, 7., 8):
#                         elapsed_time, was_stable = integrate(method, [4., h, -8., -3*h], dt, alpha, sigma, seed)
#                         all_data.append({
#                             "dt": dt,
#                             "method": method,
#                             "alpha": alpha,
#                             "sigma": sigma,
#                             "seed": seed,
#                             "param": h,
#                             "potential": "ditlevsen",
#                             "stable": was_stable,
#                             "time": elapsed_time
#                         })

def process_one_alpha(alpha):
    all_data = []
    for dt in [1.0, 0.1, 0.01, 0.001]:
        for sigma in [1., 1e-2, 1e-4, 1e-6, 1e-8]:#np.logspace(1, -8, 10, base = 10):
            #print(f"dt: {dt}, alpha: {alpha}, sigma: {sigma}")
            for seed in range(100):
                for method in ["euler", "heun", "semi_impl", "semi_impl_cased"]:
                    for c in [0.0, -0.5]:
                        elapsed_time, was_stable = integrate(method, [0.25, 0.0, -0.5, c], dt, alpha, sigma, seed)
                        all_data.append({
                            "dt": dt,
                            "method": method,
                            "alpha": alpha,
                            "sigma": sigma,
                            "seed": seed,
                            "param": c,
                            "potential": "wunderling",
                            "stable": was_stable,
                            "time": elapsed_time
                        })
                    for h in [0.0, 6.0]:
                        elapsed_time, was_stable = integrate(method, [4., h, -8., -3*h], dt, alpha, sigma, seed)
                        all_data.append({
                            "dt": dt,
                            "method": method,
                            "alpha": alpha,
                            "sigma": sigma,
                            "seed": seed,
                            "param": h,
                            "potential": "ditlevsen",
                            "stable": was_stable,
                            "time": elapsed_time
                        })
    return all_data

if __name__ == '__main__':
    alphas = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 1.9, 1.99]
    with ProcessPoolExecutor(max_workers = 4) as pool:
        processed_data = list(tqdm(pool.map(process_one_alpha, alphas), total = len(alphas)))

    all_data = []
    for section in processed_data:
        all_data += section

    df = pd.DataFrame.from_records(all_data)

    os.makedirs("/Users/vbenson/Coding/noisecascades/experiments/stability/",exist_ok=True)
    df.to_csv("/Users/vbenson/Coding/noisecascades/experiments/stability/test_and_time_integration_1d.csv")

    # df = pd.read_csv("/Users/vbenson/Coding/noisecascades/experiments/stability/test_and_time_integration_1d.csv")

    plt.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(4, 4, dpi = 250, sharex = True, sharey = True, constrained_layout=True)
    for i, dt in enumerate(df.dt.unique()):
        for j, method in enumerate(df.method.unique()):
            curr_df = df[(df.method == method) & (df.dt == dt)].groupby(by = ["alpha", "sigma"]).mean().reset_index()[["alpha", "sigma", "stable"]].astype("float")
            hm = sns.heatmap(curr_df.pivot("alpha", "sigma", "stable"), ax = axs[i,j], cmap = "crest_r", vmin = 0.0, vmax = 1.0, cbar = False,yticklabels=1, xticklabels=1)#.plot(x = "alpha", y = "sigma" logx = True)
            axs[i,j].set_title(f"{method} at dt={dt}", loc='left')
            if i != 3:
                axs[i,j].set_xlabel("")
            if j != 0:
                axs[i,j].set_ylabel("")

    cbar = fig.colorbar(hm.get_children()[0], ax = axs[:,-1], location = "right", shrink = 0.6)
    cbar.set_label(r'% of stable runs', rotation=90)
    fig.suptitle("Stability limits for different methods and stepsizes.")
    plt.savefig("/Users/vbenson/Coding/noisecascades/experiments/stability/stability_of_methods_1d.png", bbox_inches = "tight")
    plt.close("all")

    curr_df = df.groupby(["method", "dt"]).mean().reset_index()#[["time"]]
    curr_df["speedup"] = curr_df["time"].max()/curr_df["time"]
    fig = plt.figure(dpi = 300)
    sns.heatmap(curr_df.pivot("method", "dt", "speedup"), norm = LogNorm(), cmap = "viridis", cbar_kws={"label": "Relative speed increase"})
    plt.title("Relative speed comparison for different methods and stepsizes (higher is better)")
    plt.savefig("/Users/vbenson/Coding/noisecascades/experiments/stability/speedup_of_methods_1d.png", bbox_inches = "tight")
    print("Latex code: \n \n")
    print(curr_df.pivot("method", "dt", "speedup").to_latex(float_format="%.1f"))
    plt.close("all")

    def integrate_ts(method, c, dt, alpha, sigma, seed):
        return nc.Integrator.integrate_network("timeseries", dt, 100, [-1.], [c], [[0.0]], [alpha], [sigma], [1.], seed, int_method = method)
    
    plt.figure(dpi =300)
    plt.plot(*integrate_ts("semi_impl", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "semi_impl", c = "#262d42")
    plt.plot(*integrate_ts("semi_impl_cased", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "semi_impl_cased", ls = (0, (3, 3)), c = "#8b8b99")
    plt.plot(*integrate_ts("euler", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "euler", ls = (1, (3, 3)), c = "#f7c267")
    plt.plot(*integrate_ts("heun", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "heun", ls = (2, (3, 3)), c = "#9b332b")
    ax = plt.gca()
    small_ax = ax.inset_axes([0.25, 0.1, 0.5, 0.4])
    small_ax.plot(*integrate_ts("semi_impl", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "semi_impl", c = "#262d42")
    small_ax.plot(*integrate_ts("semi_impl_cased", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "semi_impl", ls = (0, (3, 3)), c = "#8b8b99")
    small_ax.plot(*integrate_ts("euler", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "euler", ls = (1, (3, 3)), c = "#f7c267")
    small_ax.plot(*integrate_ts("heun", [0.25, 0.0, -0.5, 0.0], 0.01, 0.5, 0.01, 42), label = "heun", ls = (2, (3, 3)), c = "#9b332b")
    plt.legend()
    ax.set_ylim(-3, 3)
    small_ax.set_yscale("symlog")
    small_ax.set_ylim(-1e12, 1e12)
    small_ax.set_yticks([-1e12, -1e6, 0, 1e6, 1e12])
    plt.title("Numerical instabilites for dynamics under strong 0.5-stable Levy noise.")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.savefig("/Users/vbenson/Coding/noisecascades/experiments/stability/example_stability_1d.png", bbox_inches = "tight")