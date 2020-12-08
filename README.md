# Geno-MCMC-GAN
Source code for my Master thesis project at the University of Copenhagen, with title 'Demographic inference using MCMC-coupled GAN'. I'm doing the project in the  Racimo lab, GeoGenetics at the GLOBE institute, under Graham Gower and Fernando Racimo supervision.

In a standard GAN, two neural networks called the Discriminator (D) and the Generator (G) compete in a zero-sum game through the training iterations. The goal of G is to synthesize fake data that is as realistic as the one coming from the real samples. The goal of D is to not be fooled by the data generated from G and to correctly classify the real and fake samples. In our MCMC-GAN, the Generator is a Markov Chain Monte Carlo algorithm coupled to msprime engine, and the Discriminator is a permutation-invariant (or exchangeable) Convolutional Neural Network.


## Usage
### genobuilder.py

The script genobuilder.py contains the tools to create a Genobuilder() object. This data object can generate genotype matrices from variant data under a demographic scenario. The data can be generated from different sources, such as msprime, stdpopsim and empirical data coming from VCF files. The script can be run in the console, with a command like:

> python genobuilder.py download_genmats -s msprime -n 1000 -nh 99 -l 1e6 -maf 0.05 -f 128 -se 2020 -o my_geno -p 16

Or using the long flags:

> python genobuilder.py download_genmats --source msprime --num-rep 1000  --number-haplotypes 99 --sequence.length 1e6 --maf-threshold 0.05 --fixed-dimension 128 --seed 2020 --output my_geno --parallelism 16

The different arguments and flags are:

- **function:** Function to perform. Choices are "init" (output is just the genobuilder object) or "download_genmats" (output are the genobuilder object and -n genotype matrices).

- **-s/--source:** Source engine for the genotype matrices from the real dataset to infer. Choices are "msprime", "stdpopsim" or "empirical". "msprime" is selected by default.

- **-nh/--number-haplotypes:** Number of haplotypes/rows that will conform the genotype matrices. Default is 99 (Number of individuals in CEU from 1,000 GP).

- **-l/--sequence-length:** Length of the randomly sampled genome region in bp. Default is 1000000 (1e6).

- **-maf/--maf-threshold:** Threshold for the minor allele frequency to filter rare variants. Default is 0.05.

- **-f/--fixed-dimension:** Number of columns to rescale the genmats after sampling the sequence. Default is 128.

- **-n/--num-rep:** Number of genotype matrices to generate.

- **-se/--seed:** Seed for stochastic parts of the algorithm for reproducibility.

- **-o/--output:** Name of the output file with the downloaded genobuilder pickle object "*\*.pkl*". In case of "download_genmats" function, the genotype matrices are stored in the "*\*_data.pkl*" file.

- **-p/--parallelism:** Number of cores to use for simulation. If set to zero, os.cpu_count() is used. Default is 0.

For simplicity, the default parameters work pretty good, so we recommend to run:

> python genobuilder.py download_genmats -n 1000 -o my_geno


### genomcmcgan.py

The script genomcmcgan.py contains the tool to run and train the MCMC-GAN model. To run the code, first the user need to generate a pickled file containing a genotype object (using genobuilder.py). Optionally, the user can also provide a pickle object containing a set of genotype matrices previously generated. The script can be run in the console, with a command like:

> python genomcmcgan.py my_geno.pkl -d geno_data.pkl -k hmc -e 10 -n 100 -b 50 -se 2020 -p 16

Or using the long flags:

> python genomcmcgan.py my_geno.pkl --data-path my_geno_data.pkl --kernel-name hmc --epochs 10 --num-mcmc-samples 100 --num-mcmc-burnin 50 --seed 2020 --parallelism 16

The different arguments and flags are:

- **genobuilder:** Genobuilder object to use for genotype matrix generation.

- **-k/--kernel-name:** Type of MCMC kernel to run. See choices for options. Default set to hmc.

- **-d/--data-path:** Path to genotype matrices data stored as a pickle object.

- **-m/--discriminator-model:** Path to a cnn model to load as the discriminator of the MCMC-GAN as an .hdf5 file.

- **-e/--epochs:** Number of epochs to train the discriminator on real and fake data on each iteration of MCMCGAN.

- **-n/--num-mcmc-samples:** Number of MCMC samples to collect in each training iteration of MCMCGAN.

- **-n/--num-mcmc-burnin:** Number of MCMC burn-in steps in each training iteration of MCMCGAN.

- **-se/--seed:** Seed for stochastic parts of the algorithm for reproducibility.

- **-p/--parallelism:** Number of cores to use for simulation. If set to zero, os.cpu_count() is used. Default is 0.


## Library requirements:

The code was tested using the following packages and versions:

Tensorflow==2.3.*

Tensorflow probability==0.11.0

Tensorflow addons==0.8.3

Zarr==2.4.0

Msprime==0.7.4

Stdpopsim==0.1.2
