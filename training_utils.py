import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix


class DMonitor(keras.callbacks.Callback):
    """Class of keras callback to use during model training. This callback
    monitors the cnn statistical power for paramter inference, saving the
    training results as a GIF"""

    def __init__(self, testdata, nmod, genobuilder, it, bins=100):
        self.testdata = testdata
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
        name = (f"D{self.nmod}test_{self.genob.param_name}_" \
                f"{epoch}e_it{self.iteration}")

        plot_average(
            param_values,
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
        fmt = ".2f"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, f"{cm[i, j]*100:.2f}%", ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > thresh else "black",
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
