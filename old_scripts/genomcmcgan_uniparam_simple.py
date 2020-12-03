# -*- coding: utf-8 -*-
"""genomcmcgan_4.ipynb"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # Installing required libraries
# !apt-get install python-dev libgsl0-dev
#
# # The latest version of tskit 0.3 gives problem with msprime
# !pip install tskit==0.2.3 zarr msprime stdpopsim tensorflow

# Importing libraries and modules
import msprime
import stdpopsim
import zarr
import random
import pickle
import imageio
import bisect
import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import tensorflow_probability as tfp
print(tfp.__version__)
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class Genobuilder():
    """Class for building genotype matrices from msprime, stdpopsim
      or empirical data read from Zarr directories, and other utilities
      relates to these"""

    def __init__(self, source, num_samples, seq_len, maf_thresh,
                 fixed_dim = 128, scale=False, **kwargs):
      self._num_samples = num_samples
      self._seq_len = seq_len
      self._maf_thresh = maf_thresh
      self._source = source
      self._fixed_dim = fixed_dim
      self._scale = scale
      self._param_bounds = None
      self._num_reps = None
      super(Genobuilder, self).__init__(**kwargs)


    def set_parameters(self, sim_source, param_name, fixed_val,
                       param_bounds, log_scale):

      self._sim_source = sim_source
      self._param_name = param_name
      self._param_bounds = param_bounds
      self._log_scale = log_scale
      if log_scale:
        # float_power allows to use negative numbers as exponent
        self._fixed_val = np.float_power(10, fixed_val)
      else:
        self._fixed_val = fixed_val


    @property
    def num_samples(self):
      return self._num_samples

    @property
    def seq_len(self):
      return self._seq_len

    @property
    def maf_thresh(self):
      return self._maf_thresh

    @property
    def source(self):
      return self._source

    @property
    def fixed_dim(self):
      return self._fixed_dim

    @property
    def num_reps(self):
      return self._num_reps

    @property
    def fixed_val(self):
      return self._fixed_val

    @property
    def param_bounds(self):
      return self._param_bounds

    @property
    def sim_source(self):
      return self._sim_source

    @property
    def param_name(self):
      return self._param_name

    @property
    def log_scale(self):
      return self._log_scale

    @property
    def scale(self):
      return self._scale


    @num_samples.setter
    def num_samples(self, n):
      if type(n) != int or n < 0:
        raise ValueError('Genobuilder num_samples must be a positive integer')
      self._num_samples = n

    @maf_thresh.setter
    def maf_thresh(self, maf):
      if maf < 0 or maf > 1:
        raise ValueError('The Minor Allele Frequency must be between 0 and 1')
      self._maf_thresh = maf

    @seq_len.setter
    def seq_len(self, l):
      self._seq_len = int(l)

    @source.setter
    def source(self, s):
      if s not in ['msprime', 'stdpopsim', 'empirical']:
        raise ValueError('Genobuilder source must be either msprime, '
                         'stdpopsim or empirical')
      self._source = s

    @fixed_dim.setter
    def fixed_dim(self, f):
      if f % 2 != 0:
        raise ValueError('We recommend the fixed dimension to be multiple of 2')
      self._fixed_dim = f

    @num_reps.setter
    def num_reps(self, n):
      self._num_reps = n

    @fixed_val.setter
    def fixed_val(self, v):
      if self._log_scale:
        self._fixed_val = np.float_power(10, v)
      else:
        self._fixed_val = v

    @param_bounds.setter
    def param_bounds(self, b):
      if type(b) is not tuple:
        raise ValueError('We recommend the fixed dimension to be multiple of 2')
      self._param_bounds = b

    @sim_source.setter
    def sim_source(self, s):
      if s not in ['msprime', 'stdpopsim']:
        raise ValueError('Genobuilder sim_source must be either',
                         'msprime or stdpopsim')
      self._sim_source = s

    @param_name.setter
    def param_name(self, n):
      if n not in ['recomb', 'mutation', 'effective', 'growth']:
        raise ValueError('Genobuilder param_name must be either',
                         'recomb, mutation, effective or growth')
      self._param_name = n

    @log_scale.setter
    def log_scale(self, l):
      if type(l) is not bool:
        raise ValueError('log_scale must be either True or False')
      self._log_scale = l



    def simulate_msprime(self, x=None, seed=None):
      """Simulate demographic data, returning a tensor with n_reps number
      of genotype matrices"""

      if x == None:
        x = self.fixed_val

      if self.param_name=='recombination rate':
        sims = msprime.simulate(
            sample_size=self.num_samples, Ne=10000, length=self.seq_len,
            mutation_rate=1e-8, recombination_rate=x,
            num_replicates=self.num_reps, random_seed=seed)

      elif self.param_name=='mutation rate':
        sims = msprime.simulate(
            sample_size=self.num_samples, Ne=10000, length=self.seq_len,
            mutation_rate=self.fixed_val, recombination_rate=1e-8,
            num_replicates=self.num_reps, random_seed=seed)

      elif self.param_name=='effective size':
        sims = msprime.simulate(
            sample_size=self.num_samples, Ne=self.fixed_val,
            length=self.seq_len, mutation_rate=1e-8, recombination_rate=1e-8,
            num_replicates=self.num_reps, random_seed=seed)

      elif self.param_name=='growth rate':
        popconfig = [msprime.PopulationConfiguration(
            sample_size=self._num_samples, growth_rate=self.fixed_val)]
        sims = msprime.simulate(
            Ne=10000, length=self.seq_len, mutation_rate=1e-8,
            recombination_rate=1e-8, num_replicates=self.num_reps,
            random_seed=seed, population_configurations=popconfig)

      elif self.param_name=='time bottleneck':
        demographic_events = [msprime.InstantaneousBottleneck(
            time=self.fixed_val, strength=1e3, population=0)]
        sims = msprime.simulate(
            sample_size=self.num_samples, Ne=10000, length=self.seq_len,
            mutation_rate=1e-8, recombination_rate=1e-8,
            num_replicates=self.num_reps, random_seed=seed,
            demographic_events=demographic_events)

      mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

      # For each tree sequence output from the simulation
      for i, ts in enumerate(sims):
        mat[i] = self._resize_from_ts(ts)

      # Scale genotype matrices from [0, 1] to [-1, 1]. If we were to use
      # a generator, this scale should be done with tanh function
      if self.scale:
        mat = scale_matrix(mat)

      # Expand dimension by 1 (add channel dim). -1 stands for last axis.
      mat = np.expand_dims(mat, axis=-1)

      return mat



    def simulate_msprime_list(self, param_vals, seed=None):
      """Simulate demographic data, returning a tensor with n_reps number of
      genotype matrices. Here, params is a list of values, not a single scalar"""

      sims = []

      if self.param_name=='recombination rate':
        for p in param_vals:
          sims.append(msprime.simulate(
              sample_size=self.num_samples, Ne=10000, length=self.seq_len,
              mutation_rate=1e-8, recombination_rate=p, random_seed=seed))

      elif self.param_name=='mutation rate':
        for p in param_vals:
          sims.append(msprime.simulate(
              sample_size=self.num_samples, Ne=10000, length=self.seq_len,
              mutation_rate=p, recombination_rate=1e-8, random_seed=seed))

      elif self.param_name=='effective size':
        for p in param_vals:
          sims.append(msprime.simulate(
              sample_size=self.num_samples, Ne=p, length=self.seq_len,
              mutation_rate=1e-8, recombination_rate=1e-8, random_seed=seed))

      elif self.param_name=='growth rate':
        for p in param_vals:
          popconfig = [msprime.PopulationConfiguration(
              sample_size=self.num_samples, growth_rate=p)]
          sims.append(msprime.simulate(
              Ne=10000, length=self.seq_len, mutation_rate=1e-8,
              recombination_rate=1e-8, random_seed=seed,
              population_configurations=popconfig))

      elif self.param_name=='time bottleneck':
        for p in param_vals:
          demographic_events = [msprime.InstantaneousBottleneck(
              time=p, strength=1e3, population=0)]
          sims.append(msprime.simulate(
            sample_size=self.num_samples, Ne=10000, length=self.seq_len,
            mutation_rate=1e-8, recombination_rate=1e-8, random_seed=seed,
            demographic_events=demographic_events))

      mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

      # For each tree sequence output from the simulation
      for i, ts in enumerate(sims):
        mat[i] = self._resize_from_ts(ts)

      # Scale genotype matrices from [0, 1] to [-1, 1]. If we were to use
      # a generator, this scale should be done with tanh function
      if self.scale:
        mat = scale_matrix(mat)

      # Expand dimension by 1 (add channel dim). -1 stands for last axis.
      mat = np.expand_dims(mat, axis=-1)

      return mat



    def _parse_empiricaldata(self, haplotype):

      # Set up some data paths
      mask_file = "/content/gdrive/My Drive/mcmcgan/20140520.pilot_mask.autosomes.bed"
      zarr_path = "/content/gdrive/My Drive/mcmcgan/zarr"

      # Locate the data contained in zarr
      callset = zarr.open_group(zarr_path, mode='r')

      num_samples = len(callset['1/samples'])

      data = np.zeros((self.num_reps, num_samples, self.fixed_dim))

      # Get lists of randomly selected chromosomes and genomic locations
      chroms, pos, slices = self._random_sampling_geno(
          callset, mask_file=mask_file)

      # For each randomly sampled genomic location
      for i, (chrom, pos, loc_region) in enumerate(zip(chroms, pos, slices)):
        print(f'it {i}  :  chromosome {chrom}  :  position {pos}')

        # Extract genotype and genomic position for the variants for all samples
        gt_zarr = np.asarray(callset[f'{chrom}/calldata/GT'][loc_region])
        pos_zarr = callset[f'{chrom}/variants/POS'][loc_region]
        alt_zarr = callset[f'{chrom}/variants/ALT'][loc_region]

        # Make sure the genome is diploid, and extract one of the haplotypes
        assert gt_zarr.shape[2] == 2, "Samples are not diploid"
        hap = self._haploidify(gt_zarr, haplotype)

        # To check the number of 0s and 1s in each gt
        # Filtering missing data by looking at -1? No -1 in 1000 genomes data.
        #unique, counts = np.unique(hap, return_counts=True)
        #print(dict(zip(unique, counts)))

        # Get the relative position in the sequence length to resize the matrix
        relative_pos = pos_zarr - pos

        data[i] = self._resize_from_zarr(hap, relative_pos, alt_zarr)

      if self.scale:
        data = scale_matrix(data)

      data = np.expand_dims(data, axis=-1)

      return data


    def simulate_stdpopsim(self, engine, species, model, pop,
                            error_prob=None, seed=None):

      stdengine = stdpopsim.get_engine(engine)
      stdspecies = stdpopsim.get_species(species)
      stdmodel = stdspecies.get_demographic_model(model)

      geno = [(i, get_chrom_size(i)) for i in range(1, 23)]
      # Sort the list by size.
      geno.sort(key=lambda a: a[1], reverse=True)
      cum_weights = []
      rng = random.Random(seed)
      for i, (chrom, size) in enumerate(geno):
          cum_weights.append(size if i == 0 else size + cum_weights[i-1])

      # The order for sampling from populations is ['YRI', 'CEU', 'CHB']
      if pop=='YRI':
        stdsamples = stdmodel.get_samples(self.num_samples, 0, 0)
      elif pop=='CEU':
        stdsamples = stdmodel.get_samples(0, self.num_samples, 0)
      elif pop=='CHB':
        stdsamples = stdmodel.get_samples(0, 0, self.num_samples)

      sims = []
      for i in range(self.num_reps):
        chrom, size = rng.choices(geno, cum_weights=cum_weights)[0]
        factor = self.seq_len/size
        stdcontig = stdspecies.get_contig(
            'chr' + str(chrom), length_multiplier=factor)
        sims.append(stdengine.simulate(stdmodel, stdcontig, stdsamples))

      mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

      # For each tree sequence output from the simulation
      for i, ts in enumerate(sims):

        if type(error_prob) is float:
          mat[i] = self._mutate_geno(ts, p=error_prob)

        elif type(error_prob) is np.ndarray:
          mat[i] = self._mutate_geno(ts, p=error_prob[i])

        # No error prob, it doesn't mutate the matrix
        else:
          mat[i] = self._resize_from_ts(ts)


      # Scale genotype matrices from [0, 1] to [-1, 1]. If we were to use
      # a generator, this scale should be done with tanh function
      if self.scale:
        mat = scale_matrix(mat)

      # Expand dimension by 1 (add channel dim). -1 stands for last axis.
      mat = np.expand_dims(mat, axis=-1)

      return mat



    def generate_data(self, num_reps, paramlist=None):
      #Generate (X, y) data from demographic simulations.

      self.num_reps = num_reps
      low, high = self.param_bounds

      print(f'generating {num_reps} genotype matrices from {self.source}')
      if self.source=='stdpopsim':
        gen1 = self.simulate_stdpopsim(engine='msprime', species='HomSap',
                            model='OutOfAfricaArchaicAdmixture_5R19', pop='CEU',
                            error_prob=None)

      elif self.source=='empirical':
        gen1 = self._parse_empiricaldata(haplotype = 0)

      elif self.source=='msprime':
        gen1 = self.simulate_msprime(seed=None)

      print(f'generating {num_reps} genotype matrices with different '
            f'{self.param_name} from {self.sim_source}')


      if type(paramlist) is not np.ndarray:
        if self.log_scale:
          exps = np.random.uniform(low, high, size=num_reps)
          paramlist = np.float_power(10, exps)
        else:
          paramlist = np.random.uniform(low, high, size=num_reps)
      else:
        if self.log_scale:
          paramlist = np.float_power(10, paramlist)

      if self.sim_source=='msprime':
        gen0 = self.simulate_msprime_list(paramlist, seed=None)
      elif self.sim_source=='stdpopsim':
        gen0 = self.simulate_stdpopsim(engine='msprime', species='HomSap',
                            model='OutOfAfricaArchaicAdmixture_5R19',
                            pop='CEU', error_prob=paramlist, seed=None)

      X = np.concatenate((gen1, gen0))
      y = np.concatenate((np.ones((num_reps)), np.zeros((num_reps))))
      print(f'X data shape is: {X.shape}')

      low, high = self.param_bounds

      draw_genmat(np.squeeze(gen1[0]), name=f'{self.param_name}_{self.fixed_val}')
      draw_genmat(np.squeeze(gen0[0]), name=f'{self.param_name}_{low}:{high}')

      #Split randomly into training and test data.
      return train_test_split(X, y, test_size=0.1)


    def generate_testdata(self, num_reps, testlist=None):

      self.num_reps = num_reps
      low, high = self.param_bounds

      print(f'generating {num_reps} genotype matrices with different'
            f' {self.param_name} from {self.sim_source} for testing')

      if type(testlist) is not np.ndarray:
          if self.log_scale:
            testlist = np.logspace(low, high, num=num_reps)
          else:
            testlist = np.linspace(low, high, num=num_reps)

      if self.sim_source=='msprime':
        x_test = self.simulate_msprime_list(testlist, seed=None)
      if self.sim_source=='stdpopsim':
        x_test = self.simulate_stdpopsim(engine='msprime', species='HomSap',
                            model='OutOfAfricaArchaicAdmixture_5R19',
                            pop='CEU', error_prob=testlist, seed=None)

      return x_test, testlist


    def generator_iterator(self, num_reps):

      self.num_reps = num_reps

      print(f'generating {num_reps} genotype matrices with different'
            f' {self.param_name} from {self.sim_source} for testing')

      if self.sim_source=='msprime':
        x = self.simulate_msprime(seed=None)
      if self.sim_source=='stdpopsim':
        x = self.simulate_stdpopsim(engine='msprime', species='HomSap',
                            model='OutOfAfricaArchaicAdmixture_5R19',
                            pop='CEU', error_prob=p_val, seed=None)

      yield x



    def _mutate_geno_old(self, ts, p=0.001):
      """Returns a genotype matrix with a fixed number of columns,
      as specified in x"""

      rows = int(self.num_samples)
      cols = int(self.seq_len)
      m = np.zeros((rows, cols), dtype=float)

      for variant in ts.variants():

          # Filter by MAF
          if self.maf_thresh is not None:
              af = np.mean(variant.genotypes)
              if af < self.maf_thresh or af > 1 - self.maf_thresh:
                  continue

          m[:, int(variant.site.position)] += variant.genotypes

      m = m.flatten()
      n = np.random.binomial(len(m), p)
      idx = np.random.randint(0, len(m), size=n)
      m[idx] = 1 - m[idx]
      m = m.reshape((rows, cols))

      f = int(cols/self.fixed_dim)
      mat = np.zeros((rows, self.fixed_dim), dtype=float)

      for i in range(self.fixed_dim):
        s = i*f
        e = s + f - 1
        mat[:, i] = np.sum(m[:, s:e], axis=1)

      return mat


    def _mutate_geno(self, ts, p=0.001, flip=True):
      """Returns a genotype matrix with a fixed number of columns,
      as specified in x"""

      rows = int(self.num_samples)
      cols = int(self.fixed_dim)
      m = np.zeros((rows, cols), dtype=float)

      for variant in ts.variants():

          # Filter by MAF
          if self.maf_thresh is not None:
              af = np.mean(variant.genotypes)
              if af < self.maf_thresh or af > 1 - self.maf_thresh:
                  continue

          # Polarise the matrix by major allele frequency.
          if flip:
              af = np.mean(variant.genotypes)
              if af > 0.5 or (af == 0.5 and random.Random() > 0.5):
                  variant.genotypes = 1 - variant.genotypes


          n = np.random.binomial(len(variant.genotypes), p)
          if n is not None:
              idx = np.random.randint(0, len(variant.genotypes), size=n)
              variant.genotypes[idx] = 1 - variant.genotypes[idx]

          j = int(variant.site.position * self.fixed_dim / ts.sequence_length)
          np.add(m[:, j], variant.genotypes, out=m[:,j],
                 where= variant.genotypes != -1)


      return m



    def _random_sampling_geno(self, callset, mask_file = None, seed = None):
      """random sampling from chromosome based on the proportional
      size and the mask"""

      # Extract chromosome number and length from stdpopsim catalog
      geno = [(i, get_chrom_size(i)) for i in range(1, 23)]

      # Sort the list by size.
      geno.sort(key=lambda a: a[1], reverse=True)

      cum_weights = []
      for i, (chrom, size) in enumerate(geno):
          cum_weights.append(size if i == 0 else size + cum_weights[i-1])

      print("Charging up the chromosomes")
      locs = [0]
      for i in range(1, 23):
        print(f"Charging chromosome {i}")
        query = f"{i}/variants/POS"
        locs.append(np.asarray(callset[query]))

      mask = load_mask(mask_file, min_len = 10000) if mask_file else None

      rng = random.Random(seed)
      chroms, slices, mask_ranges, pos = [], [], [], []

      while len(chroms) < self.num_reps:
        chrom, size = rng.choices(geno, cum_weights=cum_weights)[0]

        assert size > self.seq_len
        proposal = rng.randrange(0, size - self._seq_len)

        if mask:
          for start, end in mask[str(chrom)]:
            if start < proposal < end:
              chroms.append(chrom)
              pos.append(proposal)
              slices.append(locate(
                  locs[chrom], start=proposal, stop=proposal + self._seq_len))

        else:
          chroms.append(chrom)
          pos.append(proposal)
          slices.append(locate(
              locs[chrom], start=proposal, stop=proposal + self.seq_len))

      return chroms, pos, slices



    def _resize_from_ts(self, ts, flip=True):
      """Returns a genotype matrix with a fixed number of columns,
      as specified in size"""

      m = np.zeros((ts.num_samples, self.fixed_dim), dtype=float)
      flip = {0: 1, 1: 0, -1: -1}

      for variant in ts.variants():

          # Calculate allele frequency
          af = np.mean(variant.genotypes)

          # Filter by MAF
          if self.maf_thresh is not None:
              if af < self.maf_thresh or af > 1 - self.maf_thresh:
                  continue

          # Polarise the matrix by major allele frequency.
          if flip:
              if af > 0.5 or (af == 0.5 and random.Random() > 0.5):
                  variant.genotypes = [flip[b] for b in variant.genotypes]

          j = int(variant.site.position * self.fixed_dim / ts.sequence_length)
          np.add(m[:, j], variant.genotypes, out=m[:,j],
                 where= variant.genotypes != -1)

      return m


    def _resize_from_zarr(self, mat, pos, alts, flip=True):
      """Resizes a matrix using a sum window, given a genotype matrix,
      positions vector,sequence length and the desired fixed size
      of the new matrix"""

      # Initialize empty matrix with the new dimensions
      m = np.zeros((mat.shape[1], self.fixed_dim), dtype=mat.dtype)
      flip = {0: 1, 1: 0, -1: -1}

      # Fill in the resized matrix
      for _pos, _gt, _alt in zip(pos, mat, alts):

          """
          # Check that all the SNPs are biallelic
          if np.count_nonzero(_alt) != 1:
              print('found')
              continue
          """

          # Calculate allele frequency
          af = np.mean(_gt)

          # Filter by MAF
          if self.maf_thresh is not None:
              if af < self.maf_thresh or af > 1 - self.maf_thresh:
                  continue

          # Polarise the matrix by major allele frequency.
          if flip:
              if af > 0.5 or (af == 0.5 and random.Random() > 0.5):
                  _gt = [flip[b] for b in _gt]

          j = int(_pos * self.fixed_dim / self.seq_len) - 1
          np.add(m[:, j], _gt, out=m[:, j], where=_gt != -1)

      return m



    def _haploidify(self, genmat, h):
      """Returns the selected haplotype from a numpy array with
      a ploidy dimension. The parameter h must be either 0 or 1"""

      if h in [0, 1, 2]:
          if h == 2:
            self.num_samples *= 2
            return np.concatenate((genmat[:, :, 0], genmat[:, :, 1]))
          else:
            return genmat[:, :, h]

      print('The parameter h must be 0 or 1 for one haplotype, or 2 for both')
      return


def filter_maf(gt, pos, maf):
    """Filter a genotype matrix gt and the SNP position vector pos in base of
    the desired Minor Allele Frequency maf parameter"""

    # Filter alleles and position where af > maf_threshold i.e. 0.05
    af = np.mean(gt, axis=1)
    condition = af > maf
    gt = gt[np.array(condition), :]
    pos = pos[condition]

    # Filter alleles and position where af > 1 - maf_threshold i.e. 0.95
    af = np.mean(gt, axis=1)
    condition = af < 1 - maf
    gt = gt[np.array(condition), :]
    pos = pos[condition]

    return gt, pos


def load_mask(mask_file, min_len):
      """Given a mask file in BED format, parse the mask data and
      returns a matrix of tuples containing the permited regions,
      as (start, end) positions"""

      # Initialize empty mask dictionary
      mask = {str(k): [] for k in range(1, 23)}

      # Read through the lines and add to the dictionary for each chrom
      with open(mask_file, 'r') as file:
        for line in file:
          chrom, start, end, _ = line.split()
          start, end = int(start), int(end)

          if (end - start) > min_len:
            mask[chrom[3:]].append((int(start), int(end)))

        file.close()

      return mask



def get_chrom_size(chrom):
      """These sizes are based on the catalog for Homosapiens in stdpopsim,
      but they're exactly the same as the one given by the VCF files,
      so I use them for both real and simulated data"""

      chrom = str(chrom)
      length = {
          '1': 249250621, '2': 243199373, '3': 198022430, '4': 191154276,
          '5': 180915260, '6': 171115067, '7': 159138663, '8': 146364022,
          '9': 141213431, '10': 135534747, '11': 135006516, '12': 133851895,
          '13': 115169878, '14':107349540, '15': 102531392, '16': 90354753,
          '17': 81195210, '18': 78077248,'19': 59128983, '20': 63025520,
          '21': 48129895, '22': 51304566}

      return length[chrom]


def scale_matrix(mat):
      """Scale matrix values within [-1, 1] range"""
      return (mat*2/np.max(mat) - 1)


def draw_genmat(img, name):

    plt.imshow(img, cmap="winter")
    plt.title(f'genomat_{name}')
    plt.savefig(f'./results_nogauss/genomat_{name}.png')
    plt.show()


def locate(sorted_idx, start=None, stop=None):
    """This implementation comes from scikit-allel library.
    Change it a little for copyright lol"""

    start_idx = bisect.bisect_left(sorted_idx, start) if start is not None else 0
    stop_idx = bisect.bisect_right(sorted_idx, stop) if stop is not None else len(v)

    return slice(start_idx, stop_idx)


def vcf2zarr(vcf_files, pop_file, zarr_path):
# Two veery good tutorials:
# http://alimanfoo.github.io/2018/04/09/selecting-variants.html
# http://alimanfoo.github.io/2017/06/14/read-vcf.html

    # Get a list of the wanted samples from one population
    # which are found in the VCF files
    import pysam
    first_vcf = pysam.VariantFile(vcf_files.replace('{n}', '1'))
    wanted_samples = samples_from_population(pop_file)
    found_samples = list(set(wanted_samples).intersection(
                                              list(first_vcf.header.samples)))

    # Create one zarr folder for each chromosome
    for chrom in range(1, 23):
      vcf = vcf_files.replace('{n}', str(chrom))
      print(f"Creating zarr object for chromosome {chrom}")
      allel.vcf_to_zarr(vcf, zarr_path, group=str(chrom), region=str(chrom),
                        fields=['POS', 'ALT', 'samples', 'GT'],
                        samples=found_samples, overwrite=True)



class Symmetric(keras.layers.Layer):
    """Class of keras layer from permutation invariant cnn. This layer collapses
       the dimension specified in the given axis using a summary statistic"""

    def __init__(self, function, axis, **kwargs):
      self.function = function
      self.axis = axis
      super(Symmetric, self).__init__(**kwargs)

    def call(self, x):
      if self.function == 'sum':
        out = keras.backend.sum(x, axis=self.axis, keepdims=True)
      if self.function == 'mean':
        out = keras.backend.mean(x, axis=self.axis, keepdims=True)
      if self.function == 'min':
        out = keras.backend.min(x, axis=self.axis, keepdims=True)
      if self.function == 'max':
        out = keras.backend.max(x, axis=self.axis, keepdims=True)
      return out

    # Without this, Its not possible to load and save the model
    def get_config(self):

      config = super().get_config().copy()
      config.update({
          'function': self.function,
          'axis': self.axis,
      })
      return config


class DMonitor(keras.callbacks.Callback):
    """Class of keras callback to use during model training. This callback
      monitors the cnn statistical power for paramter inference, saving the
      training results as a GIF"""


    def __init__(self, testdata, nmod, genobuilder, it, bins=100):
        self.testdata = testdata
        self.nmod = nmod
        self.genob = genobuilder
        self.bins = bins
        self.img_paths = []
        self.iteration = it

    def on_epoch_end(self, epoch, logs=None):
        # Get discriminator prediction function over a range of
        # values lin parameter space
        predictions = self.model.predict(self.testdata)

        # Plot the discriminator prediction function
        name = f'D{self.nmod}test_{self.genob.param_name}_' \
               f'{epoch}e_it{self.iteration}'
        plot_average(param_values, predictions, self.genob.param_name,
                     name, self.genob.log_scale, self.bins)
        self.img_paths.append(f'./results_nogauss/{name}.png')

    def on_train_end(self, logs=None):
        # Save the sequence of images as a gif
        images = [imageio.imread(filename) for filename in self.img_paths]
        imageio.mimsave(
            f'./results_nogauss/D{self.nmod}_{self.genob.param_name}_' \
            f'{self.genob.source}_it{self.iteration}.gif',
            images, format='GIF', fps=5)



class DMonitor2(keras.callbacks.Callback):
    """Class of keras callback to use during model training. This callback
      monitors the cnn statistical power for paramter inference, saving the
      training results as a .png"""


    def __init__(self, param_vals, testdata, genobuilder, bins=100):

        self.testdata = testdata
        self.param_vals = param_vals
        self.genob = genobuilder
        self.bins = bins


    def on_epoch_end(self, epoch, logs=None):
        # Get discriminator prediction function over a range of
        # values lin parameter space
        predictions = self.model.predict(self.testdata)

        x, y = np.array(self.param_vals), np.array(predictions)

        plotx = np.mean(x.reshape((-1, self.bins)), axis=1)
        ploty = np.mean(y.reshape((-1, self.bins)), axis=1)

        if self.genob.log_scale:
            plt.plot(np.log10(plotx), ploty)
        else:
            plt.plot(plotx, ploty)

        sns.set_style('darkgrid')
        plt.ylabel('prediction D(x)')
        plt.xlabel(f'{self.genob.param_name}')
        plt.ylim((0, 1))


class ConfusionMatrix(keras.callbacks.Callback):
    """Class of keras callback to use during model training. This callback
       monitors the cnn statistical power by generating a Confusion Matrix
       at the end of each epoch"""

    def __init__(self, X, y, classes, cmap=plt.cm.Blues):

        self.X = X
        self.y = y
        self.classes = classes
        self.cmap = cmap
        sns.set_style("white")


    def on_epoch_end(self, epoch, logs={}):

        plt.clf()
        pred = self.model.predict(self.X)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        cm = confusion_matrix(self.y, pred, normalize='all')

        plt.imshow(cm, interpolation='nearest', cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{cm[i, j]*100:.2f}%',
                        ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.xlim(-0.5, len(np.unique(self.y))-0.5)
        plt.ylim(len(np.unique(self.y))-0.5, -0.5)

        plt.colorbar()
        plt.grid(False)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix recombination rate')
        plt.show()
        plt.pause(0.001)




class MCMCGAN():
    """Class for building the coupled MCMC-Discriminator architecture"""

    def __init__(self, genob, kernel_name, calibrated, discriminator = None):
        super(MCMCGAN, self).__init__()
        self.genob = genob
        self.discriminator = discriminator
        self.kernel_name = kernel_name
        self.calibrated = calibrated


    def set_discriminator(self, cnn):
        self.discriminator = cnn


    def load_discriminator(self, file):
        self.discriminator = keras.models.load_model(
            file, custom_objects={'Symmetric': Symmetric,
            'Addons>WeightNormalization': tfa.layers.WeightNormalization})


    def build_discriminator(self, model, in_shape):
      """Build different Convnet models with permutation variance property"""

      cnn = keras.models.Sequential(name='discriminator')

      if model==17:
        """Model 16 with no BN and with Weight Normalization.
        Paper: https://arxiv.org/pdf/1704.03971.pdf"""

        cnn.add(keras.layers.BatchNormalization())
        # None in input_shape for dimensions with variable size.
        cnn.add(tfa.layers.WeightNormalization(
            keras.layers.Conv2D(filters=8, kernel_size=(1, 5), padding='same',
                                strides=(1, 2), input_shape=in_shape)))
        cnn.add(keras.layers.LeakyReLU(0.3))

        cnn.add(Symmetric('max', axis=1))

        cnn.add(tfa.layers.WeightNormalization(
            keras.layers.Conv2D(filters=16, kernel_size=(1, 5),
                                padding='same', strides=(1, 2))))
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.Dropout(0.5))

        cnn.add(Symmetric('max', axis=2))


      elif model==18:

        cnn.add(keras.layers.BatchNormalization(name='BatchNorm_1'))
        # None in input_shape for dimensions with variable size.
        cnn.add(tfa.layers.WeightNormalization(
            keras.layers.Conv2D(filters=32, kernel_size=(1, 5), padding='same',
                                strides=(1, 2), input_shape=in_shape),
                                name='Conv2D_WeightNorm_1'))
        cnn.add(keras.layers.LeakyReLU(0.3, name='LeakyReLU_1'))
        cnn.add(keras.layers.BatchNormalization(name='BatchNorm_2'))

        cnn.add(Symmetric('sum', axis=1, name='Symmetric_1'))

        cnn.add(tfa.layers.WeightNormalization(
            keras.layers.Conv2D(filters=64, kernel_size=(1, 5),
                                padding='same', strides=(1, 2)),
                                name='Conv2D_WeightNorm_2'))
        cnn.add(keras.layers.LeakyReLU(0.3, name='LeakyReLU_2'))
        cnn.add(keras.layers.BatchNormalization(name='BatchNorm_3'))
        cnn.add(keras.layers.Dropout(0.5, name='Dropout'))

        cnn.add(Symmetric('sum', axis=2, name='Symmetric_2'))


      elif model==19:

        # None in input_shape for dimensions with variable size.
        cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(1, 5), padding='same',
                                strides=(1, 2), input_shape=in_shape))
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())

        cnn.add(Symmetric('max', axis=1))

        cnn.add(keras.layers.Conv2D(filters=64, kernel_size=(1, 5),
                                padding='same', strides=(1, 2)))
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Dropout(0.5))

        cnn.add(Symmetric('max', axis=2))


      elif model==20:

        # None in input_shape for dimensions with variable size.
        cnn.add(tfa.layers.WeightNormalization(
            keras.layers.Conv2D(filters=32, kernel_size=(1, 5), padding='same',
                                strides=(1, 2), input_shape=in_shape)))
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())

        cnn.add(Symmetric('sum', axis=1))

        cnn.add(tfa.layers.WeightNormalization(
            keras.layers.Conv2D(filters=64, kernel_size=(1, 5),
                                padding='same', strides=(1, 2))))
        cnn.add(keras.layers.LeakyReLU(0.3))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.Dropout(0.5))

        cnn.add(Symmetric('sum', axis=2))


      elif model=='pop_gen_cnn':
        """Convolutional neural network used in
           https://github.com/flag0010/pop_gen_cnn/"""

        cnn.add(keras.layers.Conv2D(128, 2, activation='relu'))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        cnn.add(keras.layers.Conv2D(128, 2, activation='relu'))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        cnn.add(keras.layers.Conv2D(128, 2, activation='relu'))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        cnn.add(keras.layers.Conv2D(128, 2, activation='relu'))
        cnn.add(keras.layers.BatchNormalization())
        cnn.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        cnn.add(keras.layers.Flatten())
        cnn.add(keras.layers.Dense(256, activation='relu',
                                   kernel_initializer='normal'))
        cnn.add(keras.layers.Dense(1, activation='sigmoid'))

        self.discriminator = cnn
        return


      elif model=='keras':
        """Discriminator used in the GAN implementation example in keras"""

        cnn.add(keras.layers.Conv2D(64, (1, 7), strides=(1, 2),
                                    padding="same", input_shape=in_shape))
        cnn.add(keras.layers.LeakyReLU(alpha=0.2))
        cnn.add(keras.layers.Conv2D(128, (1, 7), strides=(1, 2),
                                    padding="same"))
        cnn.add(keras.layers.LeakyReLU(alpha=0.2))
        cnn.add(keras.layers.GlobalMaxPooling2D())


      cnn.add(keras.layers.Flatten(name='Flatten'))
      cnn.add(keras.layers.Dense(128, activation='relu', name='Dense'))
      cnn.add(keras.layers.Dense(1, activation='sigmoid', name='Output_dense'))

      self.discriminator = cnn


    #@tf.function
    def D(self, x, num_reps=64):
        """
        Simulate with parameters `x`, then classify the simulations with the
        discriminator. Returns the average over `num_replicates` simulations.
        """

        tf.keras.backend.clear_session()
        self.genob.num_reps = num_reps

        if self.genob.log_scale:
          x = np.float_power(10, x.numpy())
        else:
          x = x.numpy()

        return tf.reduce_mean(self.discriminator(
                    self.genob.simulate_msprime(x).astype("float16")))


    # Target distribution is proportional to: 2 - 2 * D(x),
    # where `D(x)` is the average discriminator output from n independent
    # simulations (which are simulated with parameters `x`).
    def _unnormalized_log_prob(self, x):
        if tf.math.less(x, self.genob.param_bounds[0]) or \
                            tf.math.greater(x, self.genob.param_bounds[1]):
            print('out')
            # We reject these parameter values by returning probability 0.
            return -np.inf

        score = self.D(x)
        tf.print(score)
        return tf.math.log(score)

    def unnormalized_log_prob(self, x):
        return tf.py_function(self._unnormalized_log_prob,
                              inp=[x], Tout=tf.float32)


    def setup_mcmc(self, num_mcmc_results, num_burnin_steps,
                    initial_guess, step_size=np.float64(1.)):

      # Initialize the HMC transition kernel.
      self.num_mcmc_results = num_mcmc_results
      self.num_burnin_steps = num_burnin_steps
      self.initial_guess = initial_guess
      self.samples = None

      if self.kernel_name not in ['random walk', 'hmc', 'nuts']:
          raise NameError('kernel value must be either random walk, hmc or nuts')


      if self.kernel_name == 'random walk':
          if self.calibrated:
                self.mcmc_kernel = tfp.mcmc.RandomWalkMetropolis(
                  target_log_prob_fn=self.unnormalized_log_prob)
          else:
                self.mcmc_kernel = tfp.mcmc.UncalibratedRandomWalk(
                  target_log_prob_fn=self.unnormalized_log_prob)

          """
          self.mcmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                mcmc, num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.25)
          """


      elif self.kernel_name == 'hmc':
          if self.calibrated:
              mcmc = tfp.mcmc.HamiltonianMonteCarlo(
                      target_log_prob_fn=self.unnormalized_log_prob,
                      num_leapfrog_steps=3,
                      step_size=step_size)
          else:
              mcmc = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
                      target_log_prob_fn=self.unnormalized_log_prob,
                      num_leapfrog_steps=3,
                      step_size=step_size)

          self.mcmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                mcmc, num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                target_accept_prob=0.70)


      elif self.kernel_name == 'nuts':
          mcmc = tfp.mcmc.NoUTurnSampler(
                  target_log_prob_fn=self.unnormalized_log_prob,
                  step_size=step_size,
                  max_tree_depth=10,
                  max_energy_diff=1000.0)

          self.mcmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                mcmc, num_adaptation_steps=int(self.num_burnin_steps * 0.8),
                step_size_setter_fn=lambda pkr,
                new_step_size: pkr._replace(step_size=new_step_size),
                step_size_getter_fn=lambda pkr: pkr.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio)



    # Run the chain (with burn-in).
    @tf.function
    def run_chain(self):

        is_accepted = None
        log_acc_r = None

        if self.kernel_name == 'random walk':
            samples = tfp.mcmc.sample_chain(
                  num_results=self.num_mcmc_results,
                  num_burnin_steps=self.num_burnin_steps,
                  current_state=self.initial_guess,
                  kernel=self.mcmc_kernel,
                  trace_fn=None
                  )

        elif self.kernel_name in ['hmc', 'nuts']:
            # Run the chain (with burn-in).
            samples, [is_accepted, log_acc_rat] = tfp.mcmc.sample_chain(
                  num_results=self.num_mcmc_results,
                  num_burnin_steps=self.num_burnin_steps,
                  current_state=self.initial_guess,
                  kernel=self.mcmc_kernel,
                  trace_fn=lambda _, pkr: [pkr.inner_results.is_accepted,
                                          pkr.inner_results.log_accept_ratio]
                  )

            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            log_acc_r = tf.reduce_mean(tf.cast(log_acc_rat, dtype=tf.float32))

        self.samples = samples
        sample_mean = tf.reduce_mean(samples)
        sample_stddev = tf.math.reduce_std(samples)

        return sample_mean, sample_stddev, is_accepted, log_acc_r


    def hist_samples(self, it, bins=10):

        sns.distplot(self.samples, bins)
        sns.set_style('darkgrid')
        plt.title(f"{self.kernel_name} samples distribution for iteration {it}")
        plt.ylabel('Frequency')
        plt.xlabel('Parameter value')
        plt.savefig(f'./results_nogauss/mcmcgan_hmc_it{it}.png')


    def traceplot(self, it, ylim, c='r', a=.3):

        sns.set_style('darkgrid')
        plt.plot(self.samples, c=c, alpha=a)
        if self.genob.log_scale:
            plt.hlines(np.log10(genob.fixed_val), 0,
                   self.num_mcmc_results, zorder=4, color=c)
        else:
            plt.hlines(genob.fixed_val, 0, self.num_mcmc_results,
                        zorder=4, color=c)
        plt.title(f'MCMC trace plot for iteration {it}', fontsize='xx-large')
        plt.ylim(ylim)
        plt.ylabel('Parameter value', fontsize='xx-large')
        plt.xlabel('MCMC step', fontsize='xx-large')
        plt.legend(['MCMC samples', 'real value'], fontsize='x-large')
        sns.despine(bottom=True, left=False, offset= 1)
        plt.savefig(f'./results_nogauss/mcmcgan_trace_it{it}.png')
        plt.show()



def plot_average(x, y, param_name, name, log_scale, bins=10):

  x, y = np.array(x), np.array(y)
  plotx = np.mean(x.reshape((-1, bins)), axis=1)
  ploty = np.mean(y.reshape((-1, bins)), axis=1)

  if log_scale:
    plt.plot(np.log10(plotx), ploty)
  else:
    plt.plot(plotx, ploty)

  plt.title(name)
  plt.ylabel('prediction D(x)')
  plt.xlabel(param_name)
  plt.ylim((0, 1))
  plt.savefig(f'./results_nogauss/{name}.png')
  plt.clf()


genob = Genobuilder(
    source='msprime',   #Source of the data with fixed parameters to infer
    num_samples=99,       #Number of sampled haplotypes / number of rows
    seq_len=1e6,          #Length of the randomly sampled genome region in bp
    maf_thresh=0.05,      # Filter rare minor alleles by their frequency
    fixed_dim=128,         #Number of columns of the genotype matrix after rescaling
    scale=False
    )

genob.set_parameters(
    sim_source = 'msprime',
    param_name = 'mutation rate',      #The parameter to infer
    fixed_val = -8,             #The fixed value for the parameter in set A
    param_bounds = (-11, -6),   #The range for the randomly picked parameter value in set B
    log_scale = True            #True if the parameter are given in log10 scale
    )

# Prepare the training and validation datasets
batch_size = 32
xtrain, xval, ytrain, yval = genob.generate_data(num_reps=2000)
train_data = tf.data.Dataset.from_tensor_slices((xtrain.astype("float32"), ytrain))
train_data = (
      train_data.shuffle(len(ytrain))
      .cache()
      .batch(batch_size)
      .prefetch(2)
)

val_data = tf.data.Dataset.from_tensor_slices((xval.astype("float32"), yval))
val_data = (
      val_data.shuffle(len(yval))
      .cache()
      .batch(batch_size)
      .prefetch(2)
)

# Prepare a list of genotype matrices from a range of parameter values
# from msprime for testing
xtest, param_values = genob.generate_testdata(num_reps=2000)
test_data = tf.data.Dataset.from_tensor_slices((xtest.astype("float32")))
test_data = (
      test_data
      .cache()
      .batch(batch_size)
      .prefetch(2)
)

"""
xmoretest, xmoreval, ymoretest, ymoreval = genob.generate_data(num_reps = 3000)
more_data = tf.data.Dataset.from_tensor_slices((xmoretest.astype("float16"), ymoretest))
more_data = (
      more_data.shuffle(len(ymoretest))
      .cache()
      .batch(batch_size)
      .prefetch(2)
)
"""

print('Data simulation finished')

# Prepare the optimizer and loss function
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

model = 18
epochs = 20

mcmcgan = MCMCGAN(genob=genob, kernel_name='random walk', calibrated=True)

mcmcgan.build_discriminator(
    model, in_shape=(genob.num_samples, genob.fixed_dim, 1))
mcmcgan.discriminator.compile(
    optimizer=opt,
    loss=loss_fn,
    metrics = ['accuracy'])

training = mcmcgan.discriminator.fit(
    train_data, None, batch_size, epochs, validation_data=val_data,
    shuffle=True, callbacks=DMonitor2(param_values, test_data, genob))

# Save the keras model
mcmcgan.discriminator.summary(line_length=75, positions=[.58, .86, .99, .1])
filename = f'D{model}_trained_{epochs}e.h5'
mcmcgan.discriminator.save(filename)



mcmcgan.discriminator.run_eagerly = True
tf.config.run_functions_eagerly(True)
batch_size = 32
epochs = 10
n_reps = 3000
convergence = False
densities = []
steps = []
it = 1
max_num_iters = 10
# Initial guess must always be a float, otherwise with an int there are errors
mcmcgan.setup_mcmc(
                   num_mcmc_results=1000,
                   num_burnin_steps=100,
                   initial_guess=tf.constant([-10.])
                   )

original_bounds = mcmcgan.genob.param_bounds

while convergence == False and max_num_iters != it:

    start_t = time.time()

    # Uncalibrated kernels doesn't converge to the desired distribution.
    # MetropolisHastings(UncalibratedHamiltonianMonteCarlo(...)) is functionally
    # the same as HamiltonianMonteCarlo(...).
    print('Starting the MCMC sampling chain')
    sample_mean, sample_stddev, is_accepted, acc_rate = mcmcgan.run_chain()
    new_params = mcmcgan.samples
    mcmcgan.initial_guess = sample_mean

    print('mean:{:.8f}  stddev:{:.8f}'.format(
        sample_mean.numpy(), sample_stddev.numpy()))
    mcmcgan.traceplot(it, original_bounds)
    plt.clf()
    mcmcgan.hist_samples(it)

    min = np.min(new_params)
    max = np.max(new_params)
    if min > mcmcgan.genob.param_bounds[0]:
        mcmcgan.genob.param_bounds = (min, mcmcgan.genob.param_bounds[1])
    if max < mcmcgan.genob.param_bounds[1]:
        mcmcgan.genob.param_bounds = (mcmcgan.genob.param_bounds[0], max)
    print(f'New parameters bound are {mcmcgan.genob.param_bounds[0]}' \
            f' and {mcmcgan.genob.param_bounds[1]}')
    mcmcgan.initial_guess = tf.constant([sample_mean.numpy()])

    sns.distplot(new_params, bins=10)
    plt.title("MCMC samples distribution")
    plt.savefig(f'./results_nogauss/mcmcgan_{mcmcgan.kernel_name}_it{it}.png')
    plt.show()

    densities.extend(mcmcgan.samples)
    iteration = [it for n in range(len(mcmcgan.samples))]
    steps.extend(iteration)

    # Prepare the training and validation datasets

    xtrain, xval, ytrain, yval = mcmcgan.genob.generate_data(
                                      num_reps = n_reps, paramlist=new_params)
    train_data = tf.data.Dataset.from_tensor_slices(
                      (xtrain.astype("float32"), ytrain))
    train_data = (
          train_data.shuffle(len(ytrain))
          .cache()
          .batch(batch_size)
          .prefetch(2)
    )

    val_data = tf.data.Dataset.from_tensor_slices((xval.astype("float32"), yval))
    val_data = (
          val_data.shuffle(len(yval))
          .cache()
          .batch(batch_size)
          .prefetch(2)
    )


    xtest, param_values = mcmcgan.genob.generate_testdata(num_reps=2000)
    test_data = tf.data.Dataset.from_tensor_slices((xtest.astype("float32")))
    test_data = (
	   test_data
          .cache()
	   .batch(batch_size)
	   .prefetch(2)
)

    training = mcmcgan.discriminator.fit(
        train_data, None, batch_size, epochs, validation_data=val_data,
        shuffle=True, callbacks=DMonitor(test_data, model, genob, it))

    it += 1
    if training.history['accuracy'][-1] < 0.55:
    	print('convergence')
    	convergence = True

    t = time.time() - start_t
    print(f'A single iteration of the MCMC-GAN took {t} seconds')


df = pd.DataFrame(list(zip(densities, steps)), columns=['density', 'iteration'])

if mcmcgan.genob.log_scale:
  val = np.log10(mcmcgan.genob.fixed_val)
else:
  val = mcmcgan.genob.fixed_val

pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
ridge_plot = sns.FacetGrid(df, row="iteration", hue="iteration",
                           aspect=5, height=1.25, palette = pal)
# Use map function to make density plot in each element of the grid.
ridge_plot.map(sns.kdeplot, "density", clip_on=False,
               shade=True, alpha=0.6, lw=4, bw=.2)
ridge_plot.map(plt.axvline, x=val, c='r')
ridge_plot.map(plt.axhline, y=0, lw=4, clip_on=False)
plt.title(f'Evolution of the density estimated with MCMC')
plt.ylabel('D(x)')
plt.xlabel('Parameter value')
plt.legend(['MCMC samples', 'real value'])
ridge_plot.savefig('./results_nogauss/ridgeplot.png')
