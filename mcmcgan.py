import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from collections import OrderedDict

from symmetric import Symmetric
from training_utils import DMonitor, DMonitor2, ConfusionMatrix


class MCMCGAN:
    """Class for building the coupled MCMC-Discriminator architecture"""

    def __init__(self, genob, kernel_name, seed=None, discriminator=None):
        super(MCMCGAN, self).__init__()
        self.genob = genob
        self.discriminator = discriminator
        self.kernel_name = kernel_name
        self.seed = seed

    def set_discriminator(self, cnn):
        self.discriminator = cnn

    def load_discriminator(self, file):
        self.discriminator = keras.models.load_model(
            file,
            custom_objects={
                "Symmetric": Symmetric,
                "Addons>WeightNormalization": tfa.layers.WeightNormalization,
            },
        )

    def build_discriminator(self, model, in_shape):
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

        elif model == "pop_gen_cnn":
            """Convolutional neural network used in
            https://github.com/flag0010/pop_gen_cnn/"""

            cnn.add(keras.layers.Conv2D(128, 2, activation="relu"))
            cnn.add(keras.layers.BatchNormalization())
            cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
            cnn.add(keras.layers.Conv2D(128, 2, activation="relu"))
            cnn.add(keras.layers.BatchNormalization())
            cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
            cnn.add(keras.layers.Conv2D(128, 2, activation="relu"))
            cnn.add(keras.layers.BatchNormalization())
            cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
            cnn.add(keras.layers.Conv2D(128, 2, activation="relu"))
            cnn.add(keras.layers.BatchNormalization())
            cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
            cnn.add(keras.layers.Flatten())
            cnn.add(
                keras.layers.Dense(256, activation="relu", kernel_initializer="normal")
            )
            cnn.add(keras.layers.Dense(1, activation="sigmoid"))

            self.discriminator = cnn
            return

        elif model == "keras":
            """Discriminator used in the GAN implementation example in keras"""

            cnn.add(
                keras.layers.Conv2D(
                    64, (1, 7), strides=(1, 2), padding="same", input_shape=in_shape
                )
            )
            cnn.add(keras.layers.LeakyReLU(alpha=0.2))
            cnn.add(keras.layers.Conv2D(128, (1, 7), strides=(1, 2), padding="same"))
            cnn.add(keras.layers.LeakyReLU(alpha=0.2))
            cnn.add(keras.layers.GlobalMaxPooling2D())

        cnn.add(keras.layers.Flatten(name="Flatten"))
        cnn.add(keras.layers.Dense(128, activation="relu", name="Dense"))
        cnn.add(keras.layers.Dense(1, activation="sigmoid", name="Output_dense"))

        self.discriminator = cnn

    def D(self, x, num_reps=32):
        """
        Simulate with parameters `x`, then classify the simulations with the
        discriminator. Returns the average over `num_replicates` simulations.
        """

        self.genob.num_reps = num_reps

        return tf.reduce_mean(
            self.discriminator.predict(self.genob.simulate_msprime(x).astype("float32"))
        )

    # Where `D(x)` is the average discriminator output from n independent
    # simulations (which are simulated with parameters `x`).
    def _unnormalized_log_prob(self, x):

        import copy
        proposals = copy.deepcopy(self.genob.params)

        i = 0
        for p in proposals:
            if proposals[p].inferable:
                if tf.math.less(x[i], proposals[p].bounds[0]) or tf.math.greater(
                    x[i], proposals[p].bounds[1]
                ):
                    # We reject these parameter values by returning probability 0.
                    return -np.inf
                else:
                    proposals[p].val = x[i]
                    print(f"{proposals[p].name}: {proposals[p].val}")

                i += 1

        score = self.D(proposals)
        tf.print(score)
        return tf.math.log(score)

    def unnormalized_log_prob(self, x):
        return tf.py_function(self._unnormalized_log_prob, inp=[x], Tout=tf.float32)

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
        tf.print(self.step_sizes)

        if self.kernel_name not in ["random walk", "hmc", "nuts"]:
            raise NameError("kernel value must be either random walk, hmc or nuts")

        if self.kernel_name == "random walk":
            self.mcmc_kernel = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=self.unnormalized_log_prob
            )

        elif self.kernel_name == "hmc":
            mcmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.unnormalized_log_prob,
                num_leapfrog_steps=6,
                step_size=self.step_sizes,
            )

            self.mcmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                mcmc,
                num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.70,
            )

        elif self.kernel_name == "nuts":
            mcmc = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.unnormalized_log_prob,
                step_size=self.step_sizes,
                max_tree_depth=10,
                max_energy_diff=1000.0,
            )

            self.mcmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                mcmc,
                num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.75,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    step_size=new_step_size
                ),
                step_size_getter_fn=lambda pkr: pkr.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
            )

    # Run the chain (with burn-in).
    # autograph=False is recommended by the TFP team. It is related to how
    # control-flow statements are handled.
    # experimental_compile=True for higher efficiency in XLA_GPUs, TPUs, etc...
    @tf.function(autograph=False, experimental_compile=True)
    def run_chain(self):

        is_accepted = None
        log_acc_r = None
        tf_seed = tf.constant(self.seed)
        print(f'Selected mcmc kernel is {self.kernel_name}')

        if self.kernel_name == "random walk":
            samples = tfp.mcmc.sample_chain(
                num_results=self.num_mcmc_results,
                num_burnin_steps=self.num_burnin_steps,
                current_state=self.initial_guess,
                kernel=self.mcmc_kernel,
                seed=tf_seed,
                trace_fn=None,
                steps_between_results=self.steps_between_results,
            )

        elif self.kernel_name in ["hmc", "nuts"]:
            # Run the chain (with burn-in).
            samples, [is_accepted, log_acc_rat] = tfp.mcmc.sample_chain(
                num_results=self.num_mcmc_results,
                num_burnin_steps=self.num_burnin_steps,
                current_state=self.initial_guess,
                kernel=self.mcmc_kernel,
                seed=tf_seed,
                num_steps_between_results=self.steps_between_results,
                trace_fn=lambda _, pkr: [
                    pkr.inner_results.is_accepted,
                    pkr.inner_results.log_accept_ratio,
                ],
            )

            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            log_acc_r = tf.reduce_mean(tf.cast(log_acc_rat, dtype=tf.float32))

        self.samples = samples
        self.acceptance = is_accepted
        sample_mean = tf.reduce_mean(samples)
        sample_stddev = tf.math.reduce_std(samples)

        return sample_mean, sample_stddev, is_accepted, log_acc_r

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
            # plt.show()
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
            # plt.show()
            plt.clf()
