import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import torch
import os

# Silence tensorflow
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_probability as tfp


class MCMCGAN:
    """Class for building the coupled MCMC-Discriminator architecture"""

    def __init__(self, genob, kernel_name, seed=None, discriminator=None):
        super(MCMCGAN, self).__init__()
        self.genob = genob
        self.discriminator = discriminator
        self.kernel_name = kernel_name
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.iter = 0

    def D(self):
        """
        Simulate with parameters `x`, then classify the simulations with the
        discriminator. Returns the average over `num_replicates` simulations.
        """

        genmats = torch.Tensor(self.genob.simulate_msprime(self.proposals)).to(self.device)
        out = self.discriminator.module.predict(genmats).cpu().numpy()
        return np.mean(out.astype(np.float32))

    # Where `D(x)` is the average discriminator output from n independent
    # simulations (which are simulated with parameters `x`).
    def _target_log_prob(self, *x):

        tf.print(x)
        for i, p in enumerate(self.genob.inferable_params):
            self.proposals[p.name].val = x[i].numpy()

        score = tf.cast(self.D(), tf.float32)
        #tf.print(score)
        return tf.math.log(score)

    def target_log_prob(self, *x):
        return tf.py_function(self._target_log_prob, inp=x, Tout=tf.float32)

    def setup_mcmc(
        self,
        num_mcmc_results,
        num_burnin_steps,
        thinning,
        num_reps_Dx,
    ):

        tf.config.run_functions_eagerly(True)
        self.proposals = copy.deepcopy(self.genob.params)
        self.num_mcmc_results = num_mcmc_results
        self.num_burnin_steps = num_burnin_steps
        self.thinning = thinning
        self.genob.num_reps = num_reps_Dx
        self.samples = None
        self.stats = None
        tfb = tfp.bijectors

        self.bijs, self.inits, self.step_sizes = [], [], []
        for p in self.genob.inferable_params:
            self.bijs.append(tfb.Sigmoid(
                low=tf.cast(p.bounds[0], tf.float32),
                high=tf.cast(p.bounds[1], tf.float32)
                ),
            )
            self.inits.append(tf.cast(p.init, tf.float32))
            self.step_sizes.append(tf.cast(p.step_size, tf.float32))

        if self.kernel_name not in ["hmc", "nuts"]:
            raise NameError("kernel value must be either hmc or nuts")

        # Create and set up the HMC sampler
        elif self.kernel_name == "hmc":
            sampler = tfp.mcmc.TransformedTransitionKernel(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.target_log_prob,
                    num_leapfrog_steps=6,
                    step_size=self.step_sizes,
                ),
                bijector=self.bijs
                )

            # Step size adaptation to target_acc_prob during the burn-in stage
            self.mcmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=sampler,
                num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.75,
            )

        # Create and set up the NUTS sampler
        # Good NUTS tutorial: https://adamhaber.github.io/post/nuts/
        elif self.kernel_name == "nuts":
            sampler = tfp.mcmc.TransformedTransitionKernel(
                tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn=self.target_log_prob,
                    step_size=self.step_sizes,
                    max_tree_depth=8,
            ),
            bijector=self.bijs
            )

            # Step size adaptation to target_acc_prob during the burn-in stage
            self.mcmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=sampler,
                num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.75,
                # NUTS inside of a TTK requires custom getter/setter functions.
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    inner_results=pkr.inner_results._replace(step_size=new_step_size)
                    ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )

    def trace_fn_nuts(self, _, pkr):
        """Trace function to collect stats during NUTS sampling"""
        results = pkr.inner_results.inner_results.inner_results
        return (
            results.target_log_prob,
            results.leapfrogs_taken,
            results.has_divergence,
            results.energy,
            results.is_accepted,
            results.log_accept_ratio,
        )

    def trace_fn_hmc(self, _, pkr):
        """Trace function to collect stats during HMC sampling"""
        results = pkr.inner_results.inner_results.inner_results
        return (
            results.accepted_results.target_log_prob,
            ~(results.log_accept_ratio > -1000.0),
            results.is_accepted,
            results.accepted_results.step_size,
        )

    # autograph=False is recommended by the TFP team. It is related to how
    # control-flow statements are handled.
    # experimental_compile=True for higher efficiency in XLA_GPUs, TPUs, etc...
    @tf.function(autograph=False, experimental_compile=True)
    def run_chain(self):
        """Run the MCMC chain with the specified kernel"""

        # Stats to collect during the chain simulation
        trace_fn = (
            self.trace_fn_nuts if self.kernel_name == "nuts" else self.trace_fn_hmc
        )

        tf_seed = tf.constant(self.seed) if self.seed else None
        print(f"Selected mcmc kernel is {self.kernel_name}")

        # Add a progress bar for the chain sampling iterations
        t = self.thinning + 1
        pbar = tfp.experimental.mcmc.ProgressBarReducer(
            self.num_mcmc_results * t + self.num_burnin_steps - t
        )
        self.mcmc_kernel = tfp.experimental.mcmc.WithReductions(self.mcmc_kernel, pbar)

        # Run the chain
        samples, stats = tfp.mcmc.sample_chain(
            num_results=self.num_mcmc_results,
            num_burnin_steps=self.num_burnin_steps,
            current_state=self.inits,
            kernel=self.mcmc_kernel,
            seed=tf_seed,
            num_steps_between_results=self.thinning,
            trace_fn=trace_fn,
        )
        pbar.bar.close()
        print("sampling finished")

        # Collect the samples and stats. Download as a pickle file
        #self.samples = [s[stats[2]].numpy() for s in samples]
        self.samples = [s.numpy() for s in samples]
        self.stats = stats
        pack = [self.samples, self.stats]
        with open(f"./results/output_it{self.iter}.pkl", "wb") as obj:
            pickle.dump(pack, obj, protocol=pickle.HIGHEST_PROTOCOL)

    def result_to_stats(self):
        """Convert results from running the MCMC chain into a posterior list
        and sample_stats list that can be given to Arviz for visualizations"""

        # Stats for HMC kernel
        if self.kernel_name == "hmc":
            stats_names = ["logprob", "diverging", "acceptance", "step_size"]
            sample_stats = {k: v for k, v in zip(stats_names, self.stats)}

        # Stats for NUTS kernel
        elif self.kernel_name == "nuts":
            stats_names = [
                "logprob",
                "tree_size",
                "diverging",
                "energy",
                "acceptance",
                "mean_tree_accept",
            ]
            sample_stats = {k: v for k, v in zip(stats_names, self.stats)}
            # sample_stats['tree_size'] = np.diff(sample_stats['tree_size'], axis=1)

        # Samples from the posterior distribution
        var_names = [p.name for p in self.genob.inferable_params]
        posterior = {k: v for k, v in zip(var_names, self.samples)}
        print(f'Acceptance probability is: {np.mean(sample_stats["acceptance"])}')

        return posterior, sample_stats

    def hist_samples(self, bins=10):
        """Plot a histogram of the collected samples for a given parameter"""

        colors = ["red", "blue", "green", "black", "gold", "chocolate", "teal"]
        sns.set_style("darkgrid")
        plt.clf()
        for i, p in enumerate(self.genob.inferable_params):
            sns.distplot(self.samples[i], color=colors[i])
            ymax = plt.ylim()[1]
            if self.genob.source == "msprime":
                plt.vlines(p.val, 0, ymax, color=colors[i])
            plt.ylim(0, ymax)
            plt.legend([p.name])
            plt.xlabel("Values")
            plt.ylabel("Density")
            plt.title(
                f"{self.kernel_name} samples for {p.name} at iteration {self.iter}"
            )
            plt.savefig(
                f"./results/mcmcgan_{self.kernel_name}_histogram_it{self.iter}"
                f"_{p.name}.png"
            )
            plt.clf()

    def traceplot_samples(self):
        """Plot a traceplot of the MCMC chain states for a given parameter"""

        # EXPAND COLORS FOR MORE PARAMETERS
        colors = ["red", "blue", "green", "black", "gold", "chocolate", "teal"]
        sns.set_style("darkgrid")
        plt.clf()
        for i, p in enumerate(self.genob.inferable_params):
            plt.plot(self.samples[i], c=colors[i], alpha=0.3)
            if self.genob.source == "msprime":
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
                f"Trace plot of {self.kernel_name} samples for {p.name} at {self.iter}."
            )
            plt.savefig(
                f"./results/mcmcgan_{self.kernel_name}_traceplot_it{self.iter}"
                f"_{p.name}.png"
            )
            plt.clf()

    def jointplot_samples(self):
        """Plot a jointplot of two variables with marginal density histograms"""

        params = self.genob.inferable_params
        log = [p.plotlog for p in params]
        plt.clf()

        g = sns.jointplot(
            x=self.samples[0],
            y=self.samples[1],
            kind="hist",
            bins=30,
            log_scale=log[:2],
            height=10,
            ratio=3,
            space=0,
        )
        g.plot_joint(sns.kdeplot, color="k", zorder=1, levels=6, alpha=0.75)
        g.plot_marginals(sns.rugplot, color="r", height=-0.1, clip_on=False, alpha=0.1)
        plt.xlabel(params[0].name)
        plt.ylabel(params[1].name)
        plt.title(f"Jointplot at iteration {self.iter}")
        plt.savefig(f"./results/jointplot_{self.iter}.png")
        plt.clf()
