import os
from collections import OrderedDict
import concurrent.futures
import msprime
import pickle
import argparse
import stdpopsim
import zarr
import random
import bisect
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from parameter import Parameter
import demography as dm

_ex = None


def executor(p):
    global _ex
    if _ex is None:
        _ex = concurrent.futures.ProcessPoolExecutor(max_workers=p)
    return _ex


def do_sim(args):

    seed = args[5]
    genob = args[0]
    rng = random.Random(seed)
    ts = dm.onepop_exp(args)

    return genob._resize_from_ts(ts, rng)


class Genobuilder:
    """Class for building genotype matrices from msprime, stdpopsim
    or empirical data read from Zarr directories, and other utilities
    relates to these"""

    def __init__(
        self,
        source,
        num_samples,
        seq_len,
        maf_thresh,
        fixed_dim=128,
        seed=None,
        parallelism=0,
        **kwargs,
    ):
        self._num_samples = num_samples
        self._seq_len = seq_len
        self._maf_thresh = maf_thresh
        self._source = source
        self._fixed_dim = fixed_dim
        self._seed = seed
        self._num_reps = None
        self.parallelism = parallelism
        super(Genobuilder, self).__init__(**kwargs)

    def set_parameters(self, sim_source, params):

        self._sim_source = sim_source
        self._params = params

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
    def seed(self):
        return self._seed

    @property
    def num_reps(self):
        return self._num_reps

    @property
    def params(self):
        return self._params

    @property
    def sim_source(self):
        return self._sim_source

    @property
    def parallelism(self):
        return self._parallelism

    @num_samples.setter
    def num_samples(self, n):
        if type(n) != int or n < 0:
            raise ValueError("Genobuilder num_samples must be a positive integer")
        self._num_samples = n

    @maf_thresh.setter
    def maf_thresh(self, maf):
        if maf < 0 or maf > 1:
            raise ValueError("The Minor Allele Frequency must be between 0 and 1")
        self._maf_thresh = maf

    @seq_len.setter
    def seq_len(self, seqlen):
        self._seq_len = int(seqlen)

    @source.setter
    def source(self, s):
        if s not in ["msprime", "stdpopsim", "empirical"]:
            raise ValueError(
                "Genobuilder source must be either msprime, " "stdpopsim or empirical"
            )
        self._source = s

    @fixed_dim.setter
    def fixed_dim(self, f):
        if f % 2 != 0:
            raise ValueError("We recommend the fixed dimension to be multiple of 2")
        self._fixed_dim = f

    @seed.setter
    def seed(self, s):
        self._seed = s

    @num_reps.setter
    def num_reps(self, n):
        self._num_reps = n

    @params.setter
    def params(self, p):
        self._params = p

    @sim_source.setter
    def sim_source(self, s):
        if s not in ["msprime", "stdpopsim"]:
            raise ValueError(
                "Genobuilder sim_source must be either", "msprime or stdpopsim"
            )
        self._sim_source = s

    @parallelism.setter
    def parallelism(self, p):
        if p == 0:
            p = os.cpu_count()
        self._parallelism = p

    def simulate_msprime_list(self, param_vals, seed=None):

        sims = []
        rng = random.Random(seed)

        for p in param_vals:
            sims.append(
                msprime.simulate(
                    sample_size=self.num_samples,
                    Ne=self.params["Ne"].val,
                    length=self.seq_len,
                    mutation_rate=self.params["mu"].val,
                    recombination_rate=p,
                    random_seed=seed,
                )
            )

        mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

        # For each tree sequence output from the simulation
        for i, ts in enumerate(sims):
            mat[i] = self._resize_from_ts(ts, rng)

        # Expand dimension by 1 (add channel dim). -1 stands for last axis.
        mat = np.expand_dims(mat, axis=1)

        return mat

    def simulate_msprime(self, params, randomize=False, proposals=False):
        """Simulate demographic data, returning a tensor with n_reps number
        of genotype matrices"""

        rng = random.Random(self.seed)

        args = [
            (self, params, randomize, i, proposals, rng.randrange(1, 2 ** 32))
            for i in range(self.num_reps)
        ]
        mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))
        ex = executor(self.parallelism)
        timeout = 0.5 * self.num_reps
        try:
            for i, m in enumerate(ex.map(do_sim, args, timeout=timeout)):
                mat[i] = m
        except concurrent.futures.TimeoutError:
            print("time out!")

        # Expand dimension by 1 (add channel dim). -1 stands for last axis.
        mat = np.expand_dims(mat, axis=1)

        return mat

    def _parse_empiricaldata(self, haplotype):

        # Set up some data paths
        mask_file = "/content/gdrive/My Drive/mcmcgan/20140520.pilot_mask.autosomes.bed"
        zarr_path = "/content/gdrive/My Drive/mcmcgan/zarr"

        # Locate the data contained in zarr
        callset = zarr.open_group(zarr_path, mode="r")

        num_samples = len(callset["1/samples"])

        data = np.zeros((self.num_reps, num_samples, self.fixed_dim))

        # Get lists of randomly selected chromosomes and genomic locations
        chroms, pos, slices = self._random_sampling_geno(callset, mask_file=mask_file)

        # For each randomly sampled genomic location
        for i, (chrom, pos, loc_region) in enumerate(zip(chroms, pos, slices)):
            print(f"it {i}  :  chromosome {chrom}  :  position {pos}")

            # Extract genotype and genomic position for the variants for all samples
            gt_zarr = np.asarray(callset[f"{chrom}/calldata/GT"][loc_region])
            pos_zarr = callset[f"{chrom}/variants/POS"][loc_region]
            alt_zarr = callset[f"{chrom}/variants/ALT"][loc_region]

            # Make sure the genome is diploid, and extract one of the haplotypes
            assert gt_zarr.shape[2] == 2, "Samples are not diploid"
            hap = self._haploidify(gt_zarr, haplotype)

            # To check the number of 0s and 1s in each gt
            # Filtering missing data by looking at -1? No -1 in 1000 genomes data.
            # unique, counts = np.unique(hap, return_counts=True)
            # print(dict(zip(unique, counts)))

            # Get the relative position in the sequence length to resize the matrix
            relative_pos = pos_zarr - pos

            data[i] = self._resize_from_zarr(hap, relative_pos, alt_zarr)

        data = np.expand_dims(data, axis=1)

        return data

    def simulate_stdpopsim(
        self, engine, species, model, pop, error_prob=None, seed=None
    ):

        stdengine = stdpopsim.get_engine(engine)
        stdspecies = stdpopsim.get_species(species)
        stdmodel = stdspecies.get_demographic_model(model)

        geno = [(i, get_chrom_size(i)) for i in range(1, 23)]
        # Sort the list by size.
        geno.sort(key=lambda a: a[1], reverse=True)
        cum_weights = []
        rng = random.Random(self.seed)
        for i, (chrom, size) in enumerate(geno):
            cum_weights.append(size if i == 0 else size + cum_weights[i - 1])

        # The order for sampling from populations is ['YRI', 'CEU', 'CHB']
        if pop == "YRI":
            stdsamples = stdmodel.get_samples(self.num_samples, 0, 0)
        elif pop == "CEU":
            stdsamples = stdmodel.get_samples(0, self.num_samples, 0)
        elif pop == "CHB":
            stdsamples = stdmodel.get_samples(0, 0, self.num_samples)

        sims = []
        for i in range(self.num_reps):
            chrom, size = rng.choices(geno, cum_weights=cum_weights)[0]
            factor = self.seq_len / size
            stdcontig = stdspecies.get_contig(
                "chr" + str(chrom), length_multiplier=factor
            )
            sims.append(
                stdengine.simulate(stdmodel, stdcontig, stdsamples, seed=self.seed)
            )

        mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

        rng = random.Random(seed)
        # For each tree sequence output from the simulation
        for i, ts in enumerate(sims):

            if type(error_prob) is float:
                mat[i] = self._mutate_geno_old(ts, p=error_prob)

            elif type(error_prob) is np.ndarray:
                mat[i] = self._mutate_geno_old(ts, p=error_prob[i])

            # No error prob, it doesn't mutate the matrix
            else:
                mat[i] = self._resize_from_ts(ts, rng)

        # Expand dimension by 1 (add channel dim). -1 stands for last axis.
        mat = np.expand_dims(mat, axis=1)

        return mat

    def generate_data(self, num_reps, proposals=False):
        # Generate (X, y) data from demographic simulations.

        self.num_reps = num_reps

        print(f"generating {num_reps} genotype matrices from {self.source}")
        if self.source == "stdpopsim":
            gen1 = self.simulate_stdpopsim(
                engine="msprime",
                species="HomSap",
                model="OutOfAfricaArchaicAdmixture_5R19",
                pop="CEU",
                error_prob=None,
            )

        elif self.source == "empirical":
            gen1 = self._parse_empiricaldata(haplotype=0)

        elif self.source == "msprime":
            gen1 = self.simulate_msprime(self.params)

        print(f"generating {num_reps} genotype matrices from msprime")
        gen0 = self.simulate_msprime(self.params, randomize=True, proposals=proposals)

        X = np.concatenate((gen1, gen0))
        y = np.concatenate((np.ones((num_reps)), np.zeros((num_reps))))
        print(f"X data shape is: {X.shape}")

        # Split randomly into training and test data.
        return train_test_split(X, y, test_size=0.1, random_state=self.seed)

    def generate_fakedata(self, num_reps, testlist=None):

        self.num_reps = num_reps

        print(f"generating {num_reps} genotype matrices from msprime for testing")
        return self.simulate_msprime(self.params, seed=None, randomize=True)

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

        f = int(cols / self.fixed_dim)
        mat = np.zeros((rows, self.fixed_dim), dtype=float)

        for i in range(self.fixed_dim):
            s = i * f
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
            np.add(
                m[:, j], variant.genotypes, out=m[:, j], where=variant.genotypes != -1
            )

        return m

    def _random_sampling_geno(self, callset, mask_file=None, seed=None):
        """random sampling from chromosome based on the proportional
        size and the mask"""

        # Extract chromosome number and length from stdpopsim catalog
        geno = [(i, get_chrom_size(i)) for i in range(1, 23)]

        # Sort the list by size.
        geno.sort(key=lambda a: a[1], reverse=True)

        cum_weights = []
        for i, (chrom, size) in enumerate(geno):
            cum_weights.append(size if i == 0 else size + cum_weights[i - 1])

        print("Charging up the chromosomes")
        locs = [0]
        for i in range(1, 23):
            print(f"Charging chromosome {i}")
            query = f"{i}/variants/POS"
            locs.append(np.asarray(callset[query]))

        mask = load_mask(mask_file, min_len=10000) if mask_file else None

        rng = random.Random(seed)
        chroms, slices, pos = [], [], []

        while len(chroms) < self.num_reps:
            chrom, size = rng.choices(geno, cum_weights=cum_weights)[0]

            assert size > self.seq_len
            proposal = rng.randrange(0, size - self._seq_len)

            if mask:
                for start, end in mask[str(chrom)]:
                    if start < proposal < end:
                        chroms.append(chrom)
                        pos.append(proposal)
                        slices.append(
                            locate(
                                locs[chrom],
                                start=proposal,
                                stop=proposal + self._seq_len,
                            )
                        )

            else:
                chroms.append(chrom)
                pos.append(proposal)
                slices.append(
                    locate(locs[chrom], start=proposal, stop=proposal + self.seq_len)
                )

        return chroms, pos, slices

    def _resize_from_ts(self, ts, rng, flip=True):
        """Returns a genotype matrix with a fixed number of columns,
        as specified in size"""

        m = np.zeros((ts.num_samples, self.fixed_dim), dtype=int)
        ac_thresh = self.maf_thresh * ts.num_samples

        for variant in ts.variants():

            # Filter by MAF
            genotypes = variant.genotypes
            ac1 = np.sum(genotypes)
            ac0 = len(genotypes) - ac1
            if min(ac0, ac1) < ac_thresh:
                continue

            # Polarise 0 and 1 in genotype matrix by major allele frequency.
            # If allele counts are the same, randomly choose a major allele.
            if flip:
                if ac1 > ac0 or (ac1 == ac0 and rng.random() > 0.5):
                    genotypes ^= 1

            j = int(variant.site.position * self.fixed_dim / ts.sequence_length)
            m[:, j] += genotypes

        return m.astype(float)

    def _resize_from_zarr(self, mat, pos, alts, rng, flip=True):
        """Resizes a matrix using a sum window, given a genotype matrix,
        positions vector,sequence length and the desired fixed size
        of the new matrix"""

        # Initialize empty matrix with the new dimensions
        m = np.zeros((mat.shape[1], self.fixed_dim), dtype=mat.dtype)
        ac_thresh = self.maf_thresh * mat.shape[1]

        # Fill in the resized matrix
        for _pos, _gt, _alt in zip(pos, mat, alts):

            """
            # Check that all the SNPs are biallelic
            if np.count_nonzero(_alt) != 1:
                print('found')
                continue
            """

            # Filter by MAF
            ac1 = np.sum(_gt)
            ac0 = len(_gt) - ac1
            if min(ac0, ac1) < ac_thresh:
                continue

            # Polarise 0 and 1 in genotype matrix by major allele frequency.
            # If allele counts are the same, randomly choose a major allele.
            if flip:
                if ac1 > ac0 or (ac1 == ac0 and rng.random() > 0.5):
                    _gt ^= 1

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

        print("The parameter h must be 0 or 1 for one haplotype, or 2 for both")
        return


def load_mask(mask_file, min_len):
    """Given a mask file in BED format, parse the mask data and
    returns a matrix of tuples containing the permited regions,
    as (start, end) positions"""

    # Initialize empty mask dictionary
    mask = {str(k): [] for k in range(1, 23)}

    # Read through the lines and add to the dictionary for each chrom
    with open(mask_file, "r") as file:
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
        "1": 249250621,
        "2": 243199373,
        "3": 198022430,
        "4": 191154276,
        "5": 180915260,
        "6": 171115067,
        "7": 159138663,
        "8": 146364022,
        "9": 141213431,
        "10": 135534747,
        "11": 135006516,
        "12": 133851895,
        "13": 115169878,
        "14": 107349540,
        "15": 102531392,
        "16": 90354753,
        "17": 81195210,
        "18": 78077248,
        "19": 59128983,
        "20": 63025520,
        "21": 48129895,
        "22": 51304566,
    }

    return length[chrom]


def draw_genmat(img, name):

    plt.imshow(img, cmap="winter")
    plt.title(f"genomat_{name}")
    plt.savefig(f"./results/genomat_{name}.png")
    plt.show()


def locate(sorted_idx, start=None, stop=None):
    """This implementation comes from scikit-allel library.
    Change it a little for copyright lol"""

    start_idx = bisect.bisect_left(sorted_idx, start) if start is not None else 0
    stop_idx = (
        bisect.bisect_right(sorted_idx, stop) if stop is not None else len(sorted_idx)
    )

    return slice(start_idx, stop_idx)


# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # Parser object to collect user input from terminal
    parser = argparse.ArgumentParser(
        description="Create a genobuilder object an work with genotype matrices"
    )

    parser.add_argument(
        "function",
        help="Function to perform",
        choices=["init", "download_genmats"],
    )

    parser.add_argument(
        "-s",
        "--source",
        help="Source engine for the genotype matrices from the real dataset to infer",
        choices=["msprime", "stdpopsim", "empirical"],
        default="msprime",
    )

    parser.add_argument(
        "-nh",
        "--number-haplotypes",
        help="Number of haplotypes/rows that will conform the genotype matrices",
        default=99,
        type=int,
    )

    parser.add_argument(
        "-l",
        "--sequence-length",
        help="Length of the randomly sampled genome region in bp",
        default=1000000,
        type=int,
    )

    parser.add_argument(
        "-maf",
        "--maf-threshold",
        help="Threshold for the minor allele frequency to filter rare variants",
        default=0.05,
        type=float,
    )

    parser.add_argument(
        "-f",
        "--fixed-dimension",
        help="Number of columns to rescale the genmats after sampling the sequence",
        default=128,
        type=int,
    )

    parser.add_argument(
        "-n",
        "--num-rep",
        help="Number of genotype matrices to generate",
        type=int,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Name of the output file with the downloaded pickle object",
        default="my_geno",
        type=str,
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
    params_dict = OrderedDict()

    params_dict["r"] = Parameter(
        "r", 1.25e-8, 1e-9, (1e-10, 1e-7), inferable=False, plotlog=True
    )
    params_dict["mu"] = Parameter(
        "mu", 1.25e-8, 1e-9, (1e-11, 1e-7), inferable=False, plotlog=True
    )
    # params_dict["Ne"] = Parameter("Ne", 10000, 14000, (5000, 15000), inferable=False)

    # offset 0 - my bound
    # For onepop_exp model:
    params_dict["T1"] = Parameter("T1", 3000, 4000, (1500, 5000), inferable=True)
    params_dict["N1"] = Parameter("N1", 10000, 20000, (1000, 30000), inferable=True)
    params_dict["T2"] = Parameter("T2", 500, 1000, (100, 1500), inferable=False)
    params_dict["N2"] = Parameter("N2", 5000, 20000, (1000, 30000), inferable=True)
    params_dict["growth"] = Parameter("growth", 0.01, 0.02, (0, 0.05), inferable=True)

    # For onepop_migration model:
    """
    params_dict["T1"] = Parameter("T1", 1000, 4000, (500, 5000), inferable=False)
    params_dict["N1"] = Parameter("N1", 5000, 18000, (1000, 20000), inferable=False)
    params_dict["N2"] = Parameter("N2", 8000, 15000, (1000, 20000), inferable=False)
    params_dict["mig"] = Parameter("mig", 0.9, 0.2, (0, 0.3), inferable=True)
    """

    genob = Genobuilder(
        source=args.source,
        num_samples=args.number_haplotypes,
        seq_len=args.sequence_length,
        maf_thresh=args.maf_threshold,
        fixed_dim=args.fixed_dimension,
        seed=args.seed,
        parallelism=args.parallelism,
    )

    if len(params_dict.keys()) >= 1:
        genob.set_parameters(sim_source="msprime", params=params_dict)
    else:
        print("No parameters detected in the parameter dictionary")
        genob = None

    if genob and args.function == "init":

        output = str(args.output) + ".pkl"
        with open(output, "wb") as obj:
            pickle.dump(genob, obj, protocol=pickle.HIGHEST_PROTOCOL)

    elif genob and args.function == "download_genmats":

        xtrain, xval, ytrain, yval = genob.generate_data(args.num_rep)
        pack = [xtrain, ytrain, xval, yval]

        geno_out = str(args.output) + ".pkl"
        with open(geno_out, "wb") as obj:
            pickle.dump(genob, obj, protocol=pickle.HIGHEST_PROTOCOL)

        data_out = str(args.output) + "_data.pkl"
        with open(data_out, "wb") as obj:
            pickle.dump(pack, obj, protocol=pickle.HIGHEST_PROTOCOL)

        print("Data simulation finished")


# Command example:
# python genobuilder.py download_genmats -n 1000 -s msprime -nh 99 -l 1e6 -maf 0.05 -f 128 -se 2020 -o test -p 16
