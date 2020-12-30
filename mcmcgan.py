import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    # Stop tensorflow from vomitting all over the console!
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_probability as tfp


class Symmetric(nn.Module):
    """Class for permutation invariant cnn. This layer collapses
    the dimension specified in the given axis using a summary statistic"""

    def __init__(self, function, axis, **kwargs):
        self.function = function
        self.axis = axis
        super(Symmetric, self).__init__(**kwargs)

    def forward(self, x):
        if self.function == "sum":
            out = torch.sum(x, dim=self.axis, keepdim=True)
        elif self.function == "mean":
            out = torch.mean(x, dim=self.axis, keepdim=True)
        elif self.function == "min":
            out = torch.min(x, dim=self.axis, keepdim=True)
        elif self.function == "max":
            out = torch.max(x, dim=self.axis, keepdim=True)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 1 because it is only 1 channel in a tensor (N, C, H, W)
        self.batch1 = nn.BatchNorm2d(1, eps=0.001, momentum=0.99)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(1, 7),
            stride=(1, 2),
            padding=(0, 2),
        )
        self.batch2 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)

        self.symm1 = Symmetric("sum", 2)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(1, 7),
            stride=(1, 2),
            padding=(0, 2),
        )
        self.batch3 = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.dropout1 = nn.Dropout2d(0.5)

        self.symm2 = Symmetric("sum", 3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    # x represents our data
    def forward(self, x):

        x = self.batch1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch2(x)

        x = self.symm1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.batch3(x)
        x = self.dropout1(x)

        x = self.symm2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = torch.sigmoid(x)

        return output

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_accuracy(self, y_true, y_prob):

        y_true = y_true.squeeze()
        y_prob = y_prob.squeeze()
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item() / y_true.size(0)

    def fit(self, trainflow, valflow, epochs, lr):

        optimizer = torch.optim.Adam(self.parameters(), lr)
        lossf = nn.BCELoss()
        best_val_loss = 1.

        print("Initializing weights of the model")
        self.apply(self.weights_init)
        self.train()

        for epoch in range(epochs):  # loop over the dataset multiple times

            train_loss, val_loss, acc_train, acc_val = [0.0, 0.0, 0.0, 0.0]

            for i, (inputs, labels) in enumerate(trainflow, 1):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out = self(inputs)
                loss = lossf(out, labels)
                loss.mean().backward()
                optimizer.step()

                # print statistics
                train_loss += loss.item()
                acc_train += self.get_accuracy(labels, out)
                if i % 20 == 0:  # print every 20 mini-batches
                    print(
                        "[%d | %d] TRAINING: loss: %.3f | acc: %.3f"
                        % (
                            epoch + 1,
                            i,
                            train_loss / i,
                            acc_train / i,
                        ),
                        end="\r",
                    )

            print("")
            with torch.no_grad():
                for j, (genmats, labels) in enumerate(valflow, 1):
                    preds = self(genmats)
                    val_loss += lossf(preds, labels).item()
                    acc_val += self.get_accuracy(labels, preds)
                    print(
                        "        VALIDATION: loss: %.3f - acc: %.3f"
                        % (
                            val_loss / j,
                            acc_val / j,
                        ),
                        end="\r",
                    )
                if (val_loss / j) < best_val_loss:
                    best_val_loss = val_loss / j
                    best_epoch = epoch + 1
                    best_model = copy.deepcopy(self.state_dict())

            print("")

        self.load_state_dict(best_model)
        print(f'Best model has validation loss {best_val_loss:.3f} from {best_epoch}')

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            preds = self(inputs)
        return preds


class MCMCGAN:
    """Class for building the coupled MCMC-Discriminator architecture"""

    def __init__(self, genob, kernel_name, seed=None, discriminator=None):
        super(MCMCGAN, self).__init__()
        self.genob = genob
        self.discriminator = discriminator
        self.kernel_name = kernel_name
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def D(self, x, num_reps=32):
        """
        Simulate with parameters `x`, then classify the simulations with the
        discriminator. Returns the average over `num_replicates` simulations.
        """

        self.genob.num_reps = num_reps
        genmats = torch.Tensor(self.genob.simulate_msprime(x)).to(self.device)
        out = self.discriminator.module.predict(genmats).cpu().numpy()
        return np.mean(out.astype(np.float32))

    # Where `D(x)` is the average discriminator output from n independent
    # simulations (which are simulated with parameters `x`).
    def _unnormalized_log_prob(self, x):

        proposals = copy.deepcopy(self.genob.params)
        i = 0
        x = x.numpy()
        for p in proposals:
            if proposals[p].inferable:
                if not (proposals[p].bounds[0] < x[i] < proposals[p].bounds[1]):
                    # We reject these parameter values by returning probability 0.
                    return -np.inf
                proposals[p].val = x[i]
                #print(f"{proposals[p].name}: {proposals[p].val}")

                i += 1

        score = tf.convert_to_tensor(self.D(proposals), tf.float32)
        #tf.print(score)
        return tf.math.log(score)

    def unnormalized_log_prob(self, x):
        return tf.py_function(self._unnormalized_log_prob, inp=[x], Tout=tf.float32)

    def setup_mcmc(
        self,
        num_mcmc_results,
        num_burnin_steps,
        inits,
        step_sizes,
        steps_between_results,
    ):

        tf.config.run_functions_eagerly(True)
        # Initialize the HMC transition kernel.
        self.num_mcmc_results = num_mcmc_results
        self.num_burnin_steps = num_burnin_steps
        self.inits = tf.constant(inits, tf.float32)
        self.step_sizes = tf.constant(step_sizes, tf.float32)
        self.steps_between_results = steps_between_results
        self.samples = None
        self.stats = None

        if self.kernel_name not in ["hmc", "nuts"]:
            raise NameError("kernel value must be either hmc or nuts")

        elif self.kernel_name == "hmc":
            mcmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.unnormalized_log_prob,
                num_leapfrog_steps=10,
                step_size=self.step_sizes,
            )

            self.mcmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                mcmc,
                num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.70,
            )

        # Good NUTS tutorial: https://adamhaber.github.io/post/nuts/
        elif self.kernel_name == "nuts":
            mcmc = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=self.unnormalized_log_prob,
                step_size=self.step_sizes,
                max_tree_depth=8,
                max_energy_diff=1000.0,
            )

            self.mcmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mcmc,
                num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.70,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    step_size=new_step_size
                ),
                step_size_getter_fn=lambda pkr: pkr.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
            )


    def trace_fn_nuts(self, _, pkr):
        return (
            #pkr.inner_results.accepted_results.target_log_prob,
            pkr.inner_results.target_log_prob,
            pkr.inner_results.leapfrogs_taken,
            pkr.inner_results.has_divergence,
            pkr.inner_results.energy,
            pkr.inner_results.is_accepted,
            pkr.inner_results.log_accept_ratio
            )

    def trace_fn_hmc(self, _, pkr):
        return (
            pkr.inner_results.accepted_results.target_log_prob,
            ~(pkr.inner_results.log_accept_ratio > -1000.),
            pkr.inner_results.is_accepted,
            pkr.inner_results.accepted_results.step_size,
            )


    # Run the chain (with burn-in).
    # autograph=False is recommended by the TFP team. It is related to how
    # control-flow statements are handled.
    # experimental_compile=True for higher efficiency in XLA_GPUs, TPUs, etc...
    @tf.function(autograph=False, experimental_compile=True)
    def run_chain(self):

        trace_fn = self.trace_fn_nuts if self.kernel_name=='nuts' else self.trace_fn_hmc
        tf_seed = tf.constant(self.seed)
        print(f"Selected mcmc kernel is {self.kernel_name}")

        samples, stats = tfp.mcmc.sample_chain(
            num_results=self.num_mcmc_results,
            num_burnin_steps=self.num_burnin_steps,
            current_state=self.inits,
            kernel=self.mcmc_kernel,
            seed=tf_seed,
            num_steps_between_results=self.steps_between_results,
            trace_fn=trace_fn
        )

        self.samples = samples.numpy()
        self.stats = [s.numpy() for s in stats]
        pack = [self.samples, self.stats]
        with open("./results/output.pkl", "wb") as obj:
            pickle.dump(pack, obj, protocol=pickle.HIGHEST_PROTOCOL)



    def result_to_stats(self, params):

        if self.kernel_name=='hmc':
            stats_names = ['logprob', 'diverging', 'acceptance', 'step_size']
            sample_stats = {k:v.T for k, v in zip(stats_names, self.stats)}

        elif self.kernel_name=='nuts':
            stats_names = ['logprob', 'tree_size', 'diverging', 'energy', 'acceptance', 'mean_tree_accept']
            sample_stats = {k:v.T for k, v in zip(stats_names, self.stats)}
            #sample_stats['tree_size'] = np.diff(sample_stats['tree_size'], axis=1)

        var_names = [p.name for p in params]
        posterior = {k:v for k, v in zip(var_names, self.samples.T)}

        return posterior, sample_stats


    def hist_samples(self, params, it, bins=10):

        colors = ["red", "blue", "green", "black", "gold", "chocolate", "teal"]
        sns.set_style("darkgrid")
        plt.clf()
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
        colors = ["red", "blue", "green", "black", "gold", "chocolate", "teal"]
        sns.set_style("darkgrid")
        plt.clf()
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
                f"Trace plot of {self.kernel_name} samples for {p.name} at {it}."
            )
            plt.savefig(
                f"./results/mcmcgan_{self.kernel_name}_traceplot_it{it}"
                f"_{p.name}.png"
            )
            plt.clf()

    def jointplot_samples(self, params, it):

        log = [p.plotlog for p in params]
        plt.clf()

        g = sns.jointplot(
            x=self.samples[:, 0],
            y=self.samples[:, 1],
            kind="hist",
            bins=30,
            log_scale=log,
            height=10,
            ratio=3,
            space=0,
        )
        g.plot_joint(sns.kdeplot, color="k", zorder=1, levels=6, alpha=0.75)
        g.plot_marginals(sns.rugplot, color="r", height=-0.1, clip_on=False, alpha=0.1)
        plt.xlabel(params[0].name)
        plt.ylabel(params[1].name)
        plt.title(f"Jointplot at iteration {it}")
        plt.savefig(f"./results/jointplot_{it}.png")
        plt.clf()
