import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import arviz as az
import pickle
import os


def plot_average(x, y, param_name, name, log_scale, bins=10):

    x, y = np.array(x), np.array(y)
    plotx = np.mean(x.reshape((-1, bins)), axis=1)
    ploty = np.mean(y.reshape((-1, bins)), axis=1)

    if log_scale:
        plt.plot(np.log10(plotx), ploty)
    else:
        plt.plot(plotx, ploty)

    plt.title(name)
    plt.ylabel("prediction D(x)")
    plt.xlabel(param_name)
    plt.ylim((0, 1))
    plt.savefig(f"./results/{name}.png")
    plt.clf()


def mcmc_diagnostic_plots(posterior, sample_stats, it):

    az_trace = az.from_dict(posterior=posterior, sample_stats=sample_stats)

    """
    # 2 parameters or more for these pair plots
    if len(az_trace.posterior.data_vars) > 1:
        ax = az.plot_pair(az_trace, kind="hexbin", gridsize=30, marginals=True)
        fig = ax.ravel()[0].figure
        plt.ylim((5000, 30000))
        plt.xlim((1e-10, 1e-7))
        fig.savefig(f"./results/pair_plot_it{it}.png")
        plt.clf()

        ax = az.plot_pair(
            az_trace,
            kind=["scatter", "kde"],
            kde_kwargs={"fill_last": False},
            point_estimate="mean",
            marginals=True,
        )
        fig = ax.ravel()[0].figure
        fig.savefig(f"./results/point_estimate_plot_it{it}.png")
        plt.clf()
    """

    ax = az.plot_trace(az_trace, divergences=False)
    fig = ax.ravel()[0].figure
    fig.savefig(f"./results/trace_plot_it{it}.png")
    plt.clf()

    ax = az.plot_posterior(az_trace)
    fig = ax.ravel()[0].figure
    fig.savefig(f"./results/posterior_plot_it{it}.png")
    plt.clf()

    lag = np.minimum(len(list(posterior.values())[0]), 100)
    ax = az.plot_autocorr(az_trace, max_lag=lag)
    fig = ax.ravel()[0].figure
    fig.savefig(f"./results/autocorr_plot_it{it}.png")
    plt.clf()

    ax = az.plot_ess(az_trace, kind="evolution")
    fig = ax.ravel()[0].figure
    fig.savefig(f"./results/ess_evolution_plot_it{it}.png")
    plt.clf()
    plt.close()


def plot_disc_acc(accs, it):
    plt.plot(list(range(1, it + 1)), accs)
    plt.title("Discriminator accuracy evolution")
    plt.ylabel("D training accuracy")
    plt.xlabel("training epoch")
    plt.savefig("./results/disc_acc_evolution")
    plt.clf()


def plot_pair_evolution(params, mcmc_kernel):

    files = []
    for file in os.listdir("./results"):
        if file.startswith("output_it"):
            files.append(file)
    files = sorted(files, key=lambda x: int(x[9:-4]))
    arvzs, cs = [], []
    for i, f in enumerate(files):
        with open(f"./results/{f}", "rb") as obj:
            i += 1
            samples, stats = pickle.load(obj)
            if mcmc_kernel == "hmc":
                stats_names = ["logprob", "diverging", "acceptance", "step_size"]
            elif mcmc_kernel == "nuts":
                stats_names = [
                    "logprob",
                    "tree_size",
                    "diverging",
                    "energy",
                    "acceptance",
                    "mean_tree_accept",
                ]
            sample_stats = {k: v for k, v in zip(stats_names, stats)}
            var_names = [p.name for p in params]
            posterior = {k: v for k, v in zip(var_names, samples)}
            arvzs.append(az.from_dict(posterior=posterior, sample_stats=sample_stats))
            cs.append(i / len(files))

    ax = az.plot_pair(
        arvzs[0],
        kind="scatter",
        marginals=True,
        marginal_kwargs={"color": cm.hot_r(cs[0])},
        scatter_kwargs={"c": cm.hot_r(cs[0])},
    )
    for arvz, c in zip(arvzs[0:], cs[0:]):
        az.plot_pair(
            arvz,
            kind="scatter",
            marginals=True,
            marginal_kwargs={"color": cm.hot_r(c)},
            scatter_kwargs={"c": cm.hot_r(c)},
            ax=ax,
        )

    fig = ax.ravel()[0].figure
    fig.savefig("./results/pair_plot_evo.png")
