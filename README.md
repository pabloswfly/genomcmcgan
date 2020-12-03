# Geno-MCMC-GAN
Source code for my Master thesis project at the University of Copenhagen, with title 'Demographic inference using MCMC-coupled GAN'. I'm doing the project in the  Racimo lab, GeoGenetics at the GLOBE institute, under Graham Gower and Fernando Racimo supervision.


## Usage
### genobuilder.py

The script genobuilder.py contains the tool to create a Genobuilder() object. This data object can generate genotype matrices from variant data under a demographic scenario. The data can be generated from different sources, such as msprime, stdpopsim and empirical data coming from VCF files. The script can be run in the console, with a command like:

> python genobuilder.py download_genmats -s msprime -n 1000 -nh 99 -l 1e6 -maf 0.05 -f 128 -o test

Or using the long flags:

> python genobuilder.py download_genmats --source msprime --num-rep 1000  --number-haplotypes 99 --sequence.length 1e6 --maf-threshold 0.05 --fixed-dimension 128 --output test

The different arguments and flags are:

- **function:** Function to perform. Choices are "init" (output is just the genobuilder object) or "download_genmats" (output are the genobuilder object and -n genotype matrices).

- **-s/--source:** Source engine for the genotype matrices from the real dataset to infer. Choices are "msprime", "stdpopsim" or "empirical". "msprime" is selected by default.

- **-nh/--number-haplotypes:** Number of haplotypes/rows that will conform the genotype matrices. Default is 99 (Number of individuals in CEU from 1,000 GP).

- **-l/--sequence-length:** Length of the randomly sampled genome region in bp. Default is 1000000 (1e6).

- **-maf/--maf-threshold:** Threshold for the minor allele frequency to filter rare variants. Default is 0.05

- **-f/--fixed-dimension:** Number of columns to rescale the genmats after sampling the sequence. Default is 128

- **-n/--num-rep:** Number of genotype matrices to generate.

- **-o/--output:** Name of the output file with the downloaded genobuilder pickle object. In case of "download_genmats" function, the genotype matrices are stored in the "*_data.pkl* file.


## Library requirements:

The code was tested using the following packages and versions:

Tensorflow==2.3.*

Tensorflow probability==0.11.0

Tensorflow addons==0.8.3

Zarr==2.4.0

Msprime==0.7.4

Stdpopsim==0.1.2
