import copy
import concurrent.futures
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import littlemcmc as lmc
from scipy import optimize

from symmetric import Symmetric
from training_utils import DMonitor, DMonitor2, ConfusionMatrix


def _discriminator_build(args):
    import tensorflow as tf
    import tensorflow_addons as tfa
    from tensorflow import keras
    model_filename, model, in_shape = args

    """Build different Convnet models with permutation variance property"""

    cnn = keras.models.Sequential(name="discriminator")

    if model == 17:
        """Model 16 with no BN and with Weight Normalization.
        Paper: https://arxiv.org/pdf/1704.03971.pdf"""

        cnn.add(keras.layers.BatchNormalization())
        # None in input_shape for dimensions with variable size.
        cnn.add(
            tfa.layers.WeightNormalization(
                keras.layers.Conv2D(
                    filters=8,
                    kernel_size=(1, 5),
                    padding="same",
                    strides=(1, 2),
                    input_shape=in_shape,
                )
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3))

        cnn.add(Symmetric("max", axis=1))

        cnn.add(
            tfa.layers.WeightNormalization(
                keras.layers.Conv2D(
                    filters=16, kernel_size=(1, 5), padding="same", strides=(1, 2)
                )
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.Dropout(0.5))

        cnn.add(Symmetric("max", axis=2))

    elif model == 18:

        cnn.add(keras.layers.BatchNormalization(name="BatchNorm_1"))
        # None in input_shape for dimensions with variable size.
        cnn.add(
            tfa.layers.WeightNormalization(
                keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(1, 5),
                    padding="same",
                    strides=(1, 2),
                    input_shape=in_shape,
                ),
                name="Conv2D_WeightNorm_1",
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3, name="LeakyReLU_1"))
        cnn.add(keras.layers.BatchNormalization(name="BatchNorm_2"))

        cnn.add(Symmetric("sum", axis=1, name="Symmetric_1"))

        cnn.add(
            tfa.layers.WeightNormalization(
                keras.layers.Conv2D(
                    filters=64, kernel_size=(1, 5), padding="same", strides=(1, 2)
                ),
                name="Conv2D_WeightNorm_2",
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3, name="LeakyReLU_2"))
        cnn.add(keras.layers.BatchNormalization(name="BatchNorm_3"))
        cnn.add(keras.layers.Dropout(0.5, name="Dropout"))

        cnn.add(Symmetric("sum", axis=2, name="Symmetric_2"))

    elif model == 19:

        cnn.add(keras.layers.BatchNormalization(name="BatchNorm_1"))
        # None in input_shape for dimensions with variable size.
        cnn.add(
            keras.layers.Conv2D(
                filters=32,
                kernel_size=(1, 5),
                padding="same",
                strides=(1, 2),
                input_shape=in_shape,
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())

        cnn.add(Symmetric("sum", axis=1))

        cnn.add(
            keras.layers.Conv2D(
                filters=64, kernel_size=(1, 5), padding="same", strides=(1, 2)
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Dropout(0.5))

        cnn.add(Symmetric("sum", axis=2))

    elif model == 20:

        # None in input_shape for dimensions with variable size.
        cnn.add(
            tfa.layers.WeightNormalization(
                keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(1, 5),
                    padding="same",
                    strides=(1, 2),
                    input_shape=in_shape,
                )
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())

        cnn.add(Symmetric("sum", axis=1))

        cnn.add(
            tfa.layers.WeightNormalization(
                keras.layers.Conv2D(
                    filters=64, kernel_size=(1, 5), padding="same", strides=(1, 2)
                )
            )
        )
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Dropout(0.5))

        cnn.add(Symmetric("sum", axis=2))

    cnn.add(keras.layers.Flatten(name="Flatten"))
    cnn.add(keras.layers.Dense(128, activation="relu", name="Dense"))
    cnn.add(keras.layers.Dense(1, activation="sigmoid", name="Output_dense"))

    cnn.build(input_shape=(None, *in_shape))
    cnn.save(model_filename)


def _discriminator_load(model_filename):
    import tensorflow_addons as tfa
    from tensorflow import keras
    return keras.models.load_model(
        model_filename,
        custom_objects={
            "Symmetric": Symmetric,
            "Addons>WeightNormalization": tfa.layers.WeightNormalization,
        },
    )


def _discriminator_fit(args):
    import tensorflow as tf
    from tensorflow import keras
    model_filename, xtrain, xval, ytrain, yval, epochs = args
    cnn = _discriminator_load(model_filename)

    # Prepare the optimizer and loss function
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    cnn.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

    # Prepare the training and validation datasets
    batch_size = 32
    prefetch = 2
    train_data = tf.data.Dataset.from_tensor_slices((xtrain.astype("float32"), ytrain))
    train_data = train_data.cache().batch(batch_size).prefetch(prefetch)

    val_data = tf.data.Dataset.from_tensor_slices((xval.astype("float32"), yval))
    val_data = val_data.cache().batch(batch_size).prefetch(prefetch)

    training = cnn.fit(
        train_data, None, batch_size, epochs, validation_data=val_data, shuffle=True
    )

    # Save the keras model
    cnn.summary(line_length=75, positions=[0.58, 0.86, 0.99, 0.1])
    cnn.save(model_filename)


def _discriminator_predict(args):
    model_filename, inputs = args
    cnn = _discriminator_load(model_filename)
    return cnn.predict(inputs)


class Discriminator:
    def __init__(self, model_filename):
        self.model_filename = model_filename

    def build(self, model, in_shape):
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
            next(ex.map(_discriminator_build, [(self.model_filename, model, in_shape)]))

    def fit(self, xtrain, xval, ytrain, yval, epochs):
        args = (self.model_filename, xtrain, xval, ytrain, yval, epochs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
            next(ex.map(_discriminator_fit, [args]))

    def predict(self, inputs):
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
            preds = next(
                ex.map(_discriminator_predict, [(self.model_filename, inputs)])
            )
            print("here!")
        return preds


class MCMCGAN:
    """Class for building the coupled MCMC-Discriminator architecture"""

    def __init__(self, genob, kernel_name, seed=None, discriminator=None):
        super(MCMCGAN, self).__init__()
        self.genob = genob
        self.discriminator = discriminator
        self.kernel_name = kernel_name
        self.seed = seed

    def D(self, x, num_reps=64):
        """
        Simulate with parameters `x`, then classify the simulations with the
        discriminator. Returns the average over `num_replicates` simulations.
        """

        self.genob.num_reps = num_reps

        return np.mean(self.discriminator.predict(self.genob.simulate_msprime(x)))

    # Where `D(x)` is the average discriminator output from n independent
    # simulations (which are simulated with parameters `x`).
    def log_prob(self, x):

        proposals = copy.deepcopy(self.genob.params)
        print(x)

        i = 0
        proposals["r"].val = x
        print(f"{proposals['r'].name}: {proposals['r'].val}")
        i += 1

        score = self.D(proposals)
        print(score)
        return np.log(score)

    def dlog_prob(self, x):
        eps = x * np.sqrt(np.finfo(float).eps)
        return optimize.approx_fprime(x, self.log_prob, eps)

    def logp_dlogp(self, x):
        return self.log_prob(x), self.dlog_prob(x)

    def setup_mcmc(
        self,
        num_mcmc_results,
        num_burnin_steps,
        initial_guess,
        step_sizes,
        steps_between_results,
    ):

        # Initialize the HMC transition kernel.
        self.num_mcmc_results = num_mcmc_results
        self.num_burnin_steps = num_burnin_steps
        self.initial_guess = initial_guess
        self.step_sizes = step_sizes
        self.steps_between_results = steps_between_results
        self.samples = None
        print(self.step_sizes)

        if self.kernel_name not in ["hmc", "nuts"]:
            raise NameError("kernel value must be either hmc or nuts")

        elif self.kernel_name == "hmc":
            self.mcmc_kernel = lmc.HamiltonianMC(
                logp_dlogp_func=self.logp_dlogp,
                model_ndim=1,
                target_accept=0.75,
                adapt_step_size=True,
            )

        # Good NUTS tutorial: https://adamhaber.github.io/post/nuts/
        elif self.kernel_name == "nuts":
            self.mcmc_kernel = lmc.NUTS(
                logp_dlogp_func=self.logp_dlogp,
                model_ndim=1,
                target_accept=0.75,
                adapt_step_size=True,
            )

    # Run the chain (with burn-in).
    # autograph=False is recommended by the TFP team. It is related to how
    # control-flow statements are handled.
    # experimental_compile=True for higher efficiency in XLA_GPUs, TPUs, etc...
    # @tf.function(autograph=False, experimental_compile=True)
    def run_chain(self):

        print(f"Selected mcmc kernel is {self.kernel_name}")

        trace, stats = lmc.sample(
            logp_dlogp_func=self.logp_dlogp,
            model_ndim=1,
            step=self.mcmc_kernel,
            draws=self.num_mcmc_results,
            tune=self.num_burnin_steps,
            start=self.initial_guess,
            chains=1,
            progressbar=True,
            random_seed=self.seed,
        )

        self.samples = trace

        return stats

    def hist_samples(self, params, it, bins=10):

        colors = ["b", "g", "r"]
        for i, p in enumerate(params):
            sns.distplot(self.samples[:, i], color=colors[i])
            ymax = plt.ylim()[1]
            plt.vlines(p.val, 0, ymax, color=colors[i])
            plt.ylim(0, ymax)
            plt.legend([p.name])
            plt.xlabel("Values")
            plt.ylabel("Density")
            plt.title(f"{self.kernel_name} samples for {p.name} at iteration {it}")
            plt.savefig(
                f"./results/mcmcgan_{self.kernel_name}_histogram_it{it}"
                f"_{p.name}.png"
            )
            plt.clf()

    def traceplot_samples(self, params, it):

        # EXPAND COLORS FOR MORE PARAMETERS
        colors = ["b", "g", "r"]
        sns.set_style("darkgrid")
        for i, p in enumerate(params):
            plt.plot(self.samples[:, i], c=colors[i], alpha=0.3)
            plt.hlines(
                p.val,
                0,
                len(self.samples),
                zorder=4,
                color=colors[i],
                label="${}$".format(i),
            )
            plt.legend([p.name])
            plt.xlabel("Accepted samples")
            plt.ylabel("Values")
            plt.title(
                f"Trace plot of {self.kernel_name} samples for {p.name}"
                f" at {it}. Acc. rate: {self.acceptance:.3f}"
            )
            plt.savefig(
                f"./results/mcmcgan_{self.kernel_name}_traceplot_it{it}"
                f"_{p.name}.png"
            )
            plt.clf()

    def jointplot(self, it):

        g = sns.jointplot(self.samples[:, 0], self.samples[:, 1], kind="kde")
        g.plot_joint(sns.kdeplot, color="b", zorder=0, levels=6)
        g.plot_marginals(sns.rugplot, color="r", height=-0.15, clip_on=False)
        plt.xlabel("P1")
        plt.ylabel("P2")
        plt.title(f"Jointplot at iteration {it}")
        plt.savefig(f"./results/jointplot_{it}.png")
        plt.clf()
