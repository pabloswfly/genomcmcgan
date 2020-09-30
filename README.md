# mcmcgan
Source code for my Master thesis project at the University of Copenhagen, with title 'Demographic inference using MCMC-coupled GAN'. I'm doing the project in Fernando Racimo's group, GeoGenetics at the GLOBE institute.

The code is still under construction - This maybe can explain why it looks like a mess... for now!

## Scripts
- **genomcmcgan_joint.py:** Python script to run MCMC-GAN inference on simulated data to predict multiple demographic parameters jointly. The current parameters are the recombination rate, the mutation rate, and the effective size. The MCMC kernel selected is HMC.

- **genomcmcgan_uniparam_simple.py:** Python script to run MCMC-GAN inference on simulated data to predict a single parameter. The MCMC kernel selected is Random Walk Metropolis.

- **genomcmcgan_uniparam_simple.py:** Python script to run MCMC-GAN inference on simulated data to predict a single parameter. The MCMC kernel selected is Random Walk Metropolis, and the Gaussian approximation is applied.

## Library requirements:

Tensorflow 2.*
Keras
Zarr
Msprime
Stdpopsim
