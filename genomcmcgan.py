# -*- coding: utf-8 -*-

# # Installing required libraries
# !apt-get install python-dev libgsl0-dev
#
# # The latest version of tskit 0.3 gives problem with msprime
# !pip install tskit==0.2.3 zarr msprime stdpopsim tensorflow

# Importing libraries and modules
import os
import pickle
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from mcmcgantorch import MCMCGAN, Discriminator
from genobuilder import Genobuilder


def run_genomcmcgan(
    genobuilder,
    kernel_name,
    data_path,
    discriminator_model,
    epochs,
    num_mcmc_samples,
    num_mcmc_burnin,
    seed,
    parallelism,
):

    np.random.seed(seed)


    # Check if folder with results exists, and create it otherwise
    if not os.path.exists("./results"):
        os.makedirs("./results")

    with open(genobuilder, "rb") as obj:
        genob = pickle.load(obj)
    genob.parallelism = parallelism

    # Generate the training and validation datasets
    if data_path:
        with open(data_path, "rb") as obj:
            xtrain, ytrain, xval, yval = pickle.load(obj)
    else:
        xtrain, xval, ytrain, yval = genob.generate_data(num_reps=1000)

    mcmcgan = MCMCGAN(genob, kernel_name, seed)
    mcmcgan.discriminator = Discriminator()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        mcmcgan.discriminator = nn.DataParallel(mcmcgan.discriminator)

    mcmcgan.discriminator.to(device)
    #summary(my_nn, (1, 99, 128))

    xtrain = torch.Tensor(xtrain).float().to(device)
    ytrain = torch.Tensor(ytrain).float().unsqueeze(-1).to(device)
    trainset = torch.utils.data.TensorDataset(xtrain, ytrain)
    trainflow = torch.utils.data.DataLoader(trainset, 32, True)
    xval = torch.Tensor(xval).float().to(device)
    yval = torch.Tensor(yval).float().unsqueeze(-1).to(device)
    valset = torch.utils.data.TensorDataset(xval, yval)
    valflow = torch.utils.data.DataLoader(valset, 32, True)

    # After wrappinf the cnn model with DataParallel, -.module.- is necessary
    mcmcgan.discriminator.module.fit(trainflow, valflow, epochs, lr=0.0002)

    # Initial guess must always be a float, otherwise with an int there are errors
    inferable_params = []
    for p in mcmcgan.genob.params.values():
        print(f"{p.name} inferable: {p.inferable}")
        if p.inferable:
            inferable_params.append(p)

    initial_guesses = np.array([float(p.initial_guess) for p in inferable_params])
    step_sizes = np.array([float(p.initial_guess * 0.1) for p in inferable_params])
    mcmcgan.setup_mcmc(
        num_mcmc_samples, num_mcmc_burnin, initial_guesses, step_sizes, 1
    )

    max_num_iters = 4
    convergence = False
    it = 1

    while not convergence and max_num_iters != it:

        print("Starting the MCMC sampling chain")
        start_t = time.time()

        samples = mcmcgan.run_chain()

        # Draw traceplot and histogram of collected samples
        mcmcgan.traceplot_samples(inferable_params, it)
        mcmcgan.hist_samples(inferable_params, it)
        if mcmcgan.samples.shape[1] == 2:
            mcmcgan.jointplot_samples(inferable_params, it)

        for i, p in enumerate(inferable_params):
            p.proposals = mcmcgan.samples[:, i]

        means = np.mean(mcmcgan.samples, axis=0)
        stds = np.std(mcmcgan.samples, axis=0)
        for j, p in enumerate(inferable_params):
            print(f"{p.name} samples with mean {means[j]} and std {stds[j]}")
        initial_guesses = means
        step_sizes = stds
        # mcmcgan.step_sizes = tf.constant(np.sqrt(stds))
        mcmcgan.setup_mcmc(
            num_mcmc_samples, num_mcmc_burnin, initial_guesses, step_sizes, 1
        )

        xtrain, xval, ytrain, yval = mcmcgan.genob.generate_data(
            num_mcmc_samples, proposals=True
        )

        xtrain = torch.Tensor(xtrain).float().to(device)
        ytrain = torch.Tensor(ytrain).float().unsqueeze(-1).to(device)
        trainset = torch.utils.data.TensorDataset(xtrain, ytrain)
        trainflow = torch.utils.data.DataLoader(trainset, 32, True)
        xval = torch.Tensor(xval).float().to(device)
        yval = torch.Tensor(yval).float().unsqueeze(-1).to(device)
        valset = torch.utils.data.TensorDataset(xval, yval)
        valflow = torch.utils.data.DataLoader(valset, 32, True)

        mcmcgan.discriminator.module.fit(trainflow, valflow, epochs, lr=0.0002)

        it += 1
        t = time.time() - start_t
        print(f"A single iteration of the MCMC-GAN took {t} seconds")

    print(f"The estimates obtained after {it} iterations are:")
    print(means)


if __name__ == "__main__":

    # Parser object to collect user input from terminal
    parser = argparse.ArgumentParser(
        description="Markov Chain Monte Carlo-coupled GAN that works with"
        "genotype matrices lodaded with the Genobuilder() class"
    )

    parser.add_argument(
        "genobuilder",
        help="Genobuilder object to use for genotype matrix generation",
        type=str,
    )

    parser.add_argument(
        "-k",
        "--kernel-name",
        help="Type of MCMC kernel to run. See choices for options. Default set to hmc",
        type=str,
        choices=["hmc", "nuts", "random walk"],
        default="hmc",
    )

    parser.add_argument(
        "-d",
        "--data-path",
        help="Path to genotype matrices data stored as a pickle object",
        type=str,
    )

    parser.add_argument(
        "-m",
        "--discriminator-model",
        help="Path to a cnn model to load as the discriminator of the MCMC-GAN as an .hdf5 file",
        type=str,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs to train the discriminator on real and fake data on each iteration of MCMCGAN",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-n",
        "--num-mcmc-samples",
        help="Number of MCMC samples to collect in each training iteration of MCMCGAN",
        type=int,
        default=10,
    )

    parser.add_argument(
        "-b",
        "--num-mcmc-burnin",
        help="Number of MCMC burn-in steps in each training iteration of MCMCGAN",
        type=int,
        default=10,
    )

    parser.add_argument(
        "-se",
        "--seed",
        help="Seed for stochastic parts of the algorithm for reproducibility",
        default=None,
        type=int,
    )

    parser.add_argument(
        "-p",
        "--parallelism",
        help="Number of cores to use for simulation. If set to zero, os.cpu_count() is used.",
        default=0,
        type=int,
    )

    # Get argument values from parser
    args = parser.parse_args()

    run_genomcmcgan(
        args.genobuilder,
        args.kernel_name,
        args.data_path,
        args.discriminator_model,
        args.epochs,
        args.num_mcmc_samples,
        args.num_mcmc_burnin,
        args.seed,
        args.parallelism,
    )

    # Command example:
    # python genomcmcgan.py geno.pkl -d geno_genmats.pkl -k hmc -e 3 -n 10 -b 5 -se 2020
