import copy
import concurrent.futures
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import littlemcmc as lmc
from scipy import optimize


import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.batch1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 5),
                            stride=(1, 2), padding=(0, 2))
        self.batch2 = nn.BatchNorm2d(32)

        self.symm1 = Symmetric("sum", 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5),
                            stride=(1, 2), padding=(0, 2))
        self.batch3 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.5)

        self.symm2 = Symmetric("sum", 3)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)


    # x represents our data
    def forward(self, x):

        x = self.batch1(x)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.3)
        x = self.batch2(x)

        x = self.symm1(x)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.3)

        x = self.batch3(x)
        x = self.dropout1(x)

        x = self.symm2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = torch.sigmoid(x)
        return output


    def get_accuracy(self, y_true, y_prob):

        y_true = y_true.squeeze()
        y_prob = y_prob.squeeze()
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item() / y_true.size(0)


    def fit(self, trainflow, valflow, epochs, lr):

        optimizer = torch.optim.Adam(self.parameters(), lr)
        lossf = nn.BCELoss()

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainflow, 0):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out = self(inputs)
                loss = lossf(out, labels)
                loss.mean().backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f training acc: %.3f' %
                          (epoch+1, i+1, running_loss/20, self.get_accuracy(labels, out)), end= "\r")
                    running_loss = 0.0

            print('')
            with torch.no_grad():
                for genmats, labels in valflow:
                    preds = self(genmats)
                    print('[%d] validation acc: %.3f' % (epoch+1, self.get_accuracy(labels, preds)), end= "\r")
            print('')

    def predict(self, inputs):
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

    def D(self, x, num_reps=64):
        """
        Simulate with parameters `x`, then classify the simulations with the
        discriminator. Returns the average over `num_replicates` simulations.
        """

        self.genob.num_reps = num_reps
        genmats = torch.Tensor(self.genob.simulate_msprime(x)).to(self.device)
        out = self.discriminator.module.predict(genmats).cpu().numpy()
        return np.mean(out)

    # Where `D(x)` is the average discriminator output from n independent
    # simulations (which are simulated with parameters `x`).
    def log_prob(self, x):

        proposals = copy.deepcopy(self.genob.params)
        i = 0
        for p in proposals:
            if proposals[p].inferable:
                if not (proposals[p].bounds[0] < x[i] < proposals[p].bounds[1]):
                    # We reject these parameter values by returning probability 0.
                    return -np.inf
                proposals[p].val = x[i]
                #print(f"{proposals[p].name}: {proposals[p].val}")

                i += 1

        score = self.D(proposals)
        #print(score)
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
        model_ndim = len(initial_guess)

        if self.kernel_name not in ["hmc", "nuts"]:
            raise NameError("kernel value must be either hmc or nuts")

        elif self.kernel_name == "hmc":
            self.mcmc_kernel = lmc.HamiltonianMC(
                logp_dlogp_func=self.logp_dlogp,
                model_ndim=model_ndim,
                target_accept=0.75,
                adapt_step_size=True,
            )

        # Good NUTS tutorial: https://adamhaber.github.io/post/nuts/
        elif self.kernel_name == "nuts":
            self.mcmc_kernel = lmc.NUTS(
                logp_dlogp_func=self.logp_dlogp,
                model_ndim=model_ndim,
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
                f"Trace plot of {self.kernel_name} samples for {p.name} at {it}."
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
