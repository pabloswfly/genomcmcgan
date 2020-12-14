# -*- coding: utf-8 -*-

# # Installing required libraries
# !apt-get install python-dev libgsl0-dev
#
# # The latest version of tskit 0.3 gives problem with msprime
# !pip install tskit==0.2.3 zarr msprime stdpopsim tensorflow

# Importing libraries and modules
import pickle
import time
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mcmcgan import MCMCGAN
from genobuilder import Genobuilder

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Check if folder with results exists, and create it otherwise
    if not os.path.exists("./results"):
        os.makedirs("./results")

    with open(genobuilder, "rb") as obj:
        genob = pickle.load(obj)

    genob.parallelism = parallelism

    if data_path:
        with open(data_path, "rb") as obj:
            xtrain, ytrain, xval, yval = pickle.load(obj)
    else:
        xtrain, xval, ytrain, yval = genob.generate_data(num_reps=1000)

    # Prepare the training and validation datasets
    batch_size = 32
    train_data = tf.data.Dataset.from_tensor_slices((xtrain.astype("float32"), ytrain))
    train_data = train_data.cache().batch(batch_size).prefetch(2)

    val_data = tf.data.Dataset.from_tensor_slices((xval.astype("float32"), yval))
    val_data = val_data.cache().batch(batch_size).prefetch(2)

    # Prepare a list of genotype matrices from a range of parameter values
    # from msprime for testing
    """
    xtest = genob.generate_fakedata(num_reps=1000)
    test_data = tf.data.Dataset.from_tensor_slices((xtest.astype("float32")))
    test_data = test_data.cache().batch(batch_size).prefetch(2)
    """

    print("Data simulation finished")

    mcmcgan = MCMCGAN(genob, kernel_name, seed)

    # Load a given discriminator or build one of the implemented ones
    if discriminator_model:
        mcmcgan.load_discriminator(discriminator_model)
    else:
        model = 18
        mcmcgan.build_discriminator(
            model, in_shape=(genob.num_samples, genob.fixed_dim, 1)
        )

    # Prepare the optimizer and loss function
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    mcmcgan.discriminator.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

    training = mcmcgan.discriminator.fit(
        train_data, None, batch_size, epochs, validation_data=val_data, shuffle=True
    )

    # Save the keras model
    mcmcgan.discriminator.summary(line_length=75, positions=[0.58, 0.86, 0.99, 0.1])
    filename = f"D{model}_trained_{epochs}e.h5"
    mcmcgan.discriminator.save(filename)

    # Initial guess must always be a float, otherwise with an int there are errors
    inferable_params = []
    for p in mcmcgan.genob.params.values():
        print(f"{p.name} inferable: {p.inferable}")
        if p.inferable:
            inferable_params.append(p)

    initial_guesses = tf.constant([float(p.initial_guess) for p in inferable_params])
    step_sizes = tf.constant([float(p.initial_guess * 0.1) for p in inferable_params])
    mcmcgan.discriminator.run_eagerly = True
    tf.config.run_functions_eagerly(True)
    mcmcgan.setup_mcmc(
        num_mcmc_samples, num_mcmc_burnin, initial_guesses, step_sizes, 1
    )

    max_num_iters = 4
    convergence = False
    it = 1

    while not convergence and max_num_iters != it:

        print(f"Starting the MCMC sampling chain for iteration {it}")
        start_t = time.time()

        is_accepted, log_acc_rate = mcmcgan.run_chain()
        print(f"Is accepted: {is_accepted}, acc_rate: {log_acc_rate}")

        # Draw traceplot and histogram of collected samples
        mcmcgan.traceplot_samples(inferable_params, it)
        mcmcgan.hist_samples(inferable_params, it)
        print(mcmcgan.samples.shape)
        if mcmcgan.samples.shape[1] == 2:
            mcmcgan.jointplot_samples(inferable_params, it)

        for i, p in enumerate(inferable_params):
            p.proposals = mcmcgan.samples[:, i].numpy()

        means = np.mean(mcmcgan.samples, axis=0)
        stds = np.std(mcmcgan.samples, axis=0)
        for j, p in enumerate(inferable_params):
            print(f"{p.name} samples with mean {means[j]} and std {stds[j]}")
        initial_guesses = tf.constant(means)
        step_sizes = tf.constant(stds)
        # mcmcgan.step_sizes = tf.constant(np.sqrt(stds))
        mcmcgan.setup_mcmc(
            num_mcmc_samples, num_mcmc_burnin, initial_guesses, step_sizes, 1
        )

        # Prepare the training and validation datasets
        # xtrain, xval, ytrain, yval = mcmcgan.genob.generate_data(n_reps)
        xtrain, xval, ytrain, yval = mcmcgan.genob.generate_data(
            num_mcmc_samples, proposals=True
        )
        train_data = tf.data.Dataset.from_tensor_slices(
            (xtrain.astype("float32"), ytrain)
        )
        train_data = train_data.cache().batch(batch_size).prefetch(16)

        val_data = tf.data.Dataset.from_tensor_slices((xval.astype("float32"), yval))
        val_data = val_data.cache().batch(batch_size).prefetch(16)

        mcmcgan.discriminator.fit(
            train_data, None, batch_size, epochs, val_data, shuffle=True
        )

        it += 1
        if training.history["accuracy"][-1] < 0.55:
            print("convergence")
            convergence = True

        t = time.time() - start_t
        print(f"A single iteration of the MCMC-GAN took {t} seconds")

    print(f"The estimates obtained after {it} iterations are:")
    print(float(means))


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
