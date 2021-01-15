import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az
from tensorflow import keras
from sklearn.metrics import confusion_matrix


class DMonitor(keras.callbacks.Callback):
    """Class of keras callback to use during model training. This callback
    monitors the cnn statistical power for paramter inference, saving the
    training results as a GIF"""

    def __init__(self, param_values, testdata, nmod, genobuilder, it, bins=100):
        self.testdata = testdata
        self.param_values = param_values
        self.nmod = nmod
        self.genob = genobuilder
        self.bins = bins
        self.img_paths = []
        self.iteration = it

    def on_epoch_end(self, epoch, logs=None):
        # Get discriminator prediction function over a range of
        # values lin parameter space
        predictions = self.model.predict(self.testdata)

        # Plot the discriminator prediction function
        name = (
            f"D{self.nmod}test_{self.genob.param_name}_" f"{epoch}e_it{self.iteration}"
        )

        plot_average(
            self.param_values,
            predictions,
            self.genob.param_name,
            name,
            self.genob.log_scale,
            self.bins,
        )

        self.img_paths.append(f"./results_joint/{name}.png")

    def on_train_end(self, logs=None):
        # Save the sequence of images as a gif
        images = [imageio.imread(filename) for filename in self.img_paths]
        imageio.mimsave(
            f"./results_joint/D{self.nmod}_{self.genob.param_name}_"
            f"{self.genob.source}_it{self.iteration}.gif",
            images,
            format="GIF",
            fps=5,
        )


class DMonitor2(keras.callbacks.Callback):
    """Class of keras callback to use during model training. This callback
    monitors the cnn statistical power for paramter inference, saving the
    training results as a .png"""

    def __init__(self, param_vals, testdata, genobuilder, bins=100):

        self.testdata = testdata
        self.param_vals = param_vals
        self.genob = genobuilder
        self.bins = bins

    def on_epoch_end(self, epoch, logs=None):
        # Get discriminator prediction function over a range of
        # values lin parameter space
        predictions = self.model.predict(self.testdata)

        x, y = np.array(self.param_vals), np.array(predictions)

        plotx = np.mean(x.reshape((-1, self.bins)), axis=1)
        ploty = np.mean(y.reshape((-1, self.bins)), axis=1)

        if self.genob.log_scale:
            plt.plot(np.log10(plotx), ploty)
        else:
            plt.plot(plotx, ploty)

        sns.set_style("darkgrid")
        plt.ylabel("prediction D(x)")
        plt.xlabel(f"{self.genob.param_name}")
        plt.ylim((0, 1))


class ConfusionMatrix(keras.callbacks.Callback):
    """Class of keras callback to use during model training. This callback
    monitors the cnn statistical power by generating a Confusion Matrix
    at the end of each epoch"""

    def __init__(self, X, y, classes, cmap=plt.cm.Blues):

        self.X = X
        self.y = y
        self.classes = classes
        self.cmap = cmap
        sns.set_style("white")

    def on_epoch_end(self, epoch, logs={}):

        plt.clf()
        pred = self.model.predict(self.X)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        cm = confusion_matrix(self.y, pred, normalize="all")

        plt.imshow(cm, interpolation="nearest", cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{cm[i, j]*100:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()
        plt.xlim(-0.5, len(np.unique(self.y)) - 0.5)
        plt.ylim(len(np.unique(self.y)) - 0.5, -0.5)

        plt.colorbar()
        plt.grid(False)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title("Confusion Matrix recombination rate")
        plt.show()
        plt.pause(0.001)


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

    ax = az.plot_trace(az_trace, divergences=False)
    fig = ax.ravel()[0].figure
    fig.savefig(f"./results/trace_plot_it{it}.png")
    plt.clf()

    # 2 parameters or more for these pair plots
    if len(az_trace.posterior.data_vars) > 1:
        ax = az.plot_pair(az_trace, kind="hexbin", gridsize=30, marginals=True)
        fig = ax.ravel()[0].figure
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
