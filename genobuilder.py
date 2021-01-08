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
    """Perform msprime simulations with multiprocessing"""

    # Perform simulation with chosen demographic model
    genob = args[0]
    ts = dm.onepop_exp(args)

    # Return resized matrix
    return genob._resize_from_ts(ts)


def do_parsing(args):
    """Parse data from VCF files with multiprocessing"""

    genob, callset, chrom, pos, loc_region, h, i = args
    print(f"it {i}  :  chromosome {chrom}  :  position {pos}", end="\r")

    # Extract genotype and genomic position for the variants for all samples
    gt_zarr = np.asarray(callset[f"{chrom}/calldata/GT"][loc_region])
    pos_zarr = callset[f"{chrom}/variants/POS"][loc_region]
    alt_zarr = callset[f"{chrom}/variants/ALT"][loc_region]

    # Make sure the genome is diploid, and extract one of the haplotypes
    assert gt_zarr.shape[2] == 2, "Samples are not diploid"
    hap = haploidify(gt_zarr, h)

    # Get the relative position in the sequence length to resize the matrix
    relative_pos = pos_zarr - pos

    # Return resized matrix
    return genob._resize_from_zarr(hap, relative_pos, alt_zarr)


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
        zarr_path="",
        mask_file="",
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
        self._parallelism = parallelism
        self._rng = random.Random(seed)
        self._zarr_path = zarr_path
        self._mask_file = mask_file
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

    @property
    def rng(self):
        return self._rng

    @property
    def zarr_path(self):
        return self._zarr_path

    @property
    def mask_file(self):
        return self._mask_file

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

    @rng.setter
    def rng(self, r):
        self._rng = r

    @zarr_path.setter
    def zarr_path(self, z):
        assert isinstance(z, str), "zarr_path must be a string"
        assert os.path.isdir(z), "path in zarr_path does not exist"
        self._zarr_path = z

    @mask_file.setter
    def mask_file(self, m):
        assert isinstance(m, str), "mask_file must be a string"
        assert os.path.isfile(m), "file in mask_file does not exist"
        self._mask_file = m

    def simulate_msprime_list(self, param_vals):
        """This function will go away once the code is complete"""

        sims = []
        for p in param_vals:
            sims.append(
                msprime.simulate(
                    sample_size=self.num_samples,
                    Ne=self.params["Ne"].val,
                    length=self.seq_len,
                    mutation_rate=self.params["mu"].val,
                    recombination_rate=p,
                    random_seed=self.seed,
                )
            )

        mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

        # For each tree sequence output from the simulation
        for i, ts in enumerate(sims):
            mat[i] = self._resize_from_ts(ts)

        # Expand dimension by 1 (add channel dim). -1 stands for last axis.
        return np.expand_dims(mat, axis=1)

    def simulate_msprime(self, params, randomize=False, proposals=False):
        """Simulate demographic data, returning a tensor with n_reps number
        of genotype matrices.
        params: dictionary of Parameter class Values.
        randomize: True for random parameter value selection (msprime sims)
        proposals: True for proposal parameter values (calculating D(x) scores)
        """

        # Prepare arguments and empty matrix
        args = [(self, params, randomize, i, proposals) for i in range(self.num_reps)]
        mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

        # Executor for multiprocessing
        ex = executor(self.parallelism)

        # Do simulations with multiprocessing except if it takes too long
        timeout = 0.5 * self.num_reps
        try:
            for i, m in enumerate(ex.map(do_sim, args, timeout=timeout)):
                mat[i] = m
        except concurrent.futures.TimeoutError:
            print("time out!")

        # Expand dimension by 1 (add channel dim).
        return np.expand_dims(mat, axis=1)

    def parse_empirical_data(self, haplotype):
        """Parse empirical data from Zarr files previously parsed
        with vcf2zarr.py.
        haplotype: hap to extract gt data. 0 or 1 for each of them, 2 for both
        """

        assert self.zarr_path != "", "--zarr-path argument must be a path string"
        if haplotype == 2:
            self.num_samples *= 2

        # Locate the data contained in the zarr files
        callset = zarr.open_group(self.zarr_path, mode="r")
        num_samples = len(callset["1/samples"])
        mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

        # Get lists of randomly selected chromosomes and genomic locations
        chrom, pos, loc_region = self._random_sampling_geno(callset)
        idx = list(range(1, self.num_reps))
        args = [
            [self] * self.num_reps,
            [callset] * self.num_reps,
            chrom,
            pos,
            loc_region,
            [haplotype] * self.num_reps,
            idx,
        ]
        # Executor for multiprocessing
        ex = executor(self.parallelism)

        # Do simulations with multiprocessing except if it takes too long
        timeout = 0.5 * self.num_reps
        try:
            # For each randomly sampled genomic location
            for i, m in enumerate(ex.map(do_parsing, zip(*args), timeout=timeout)):
                mat[i] = m
        except concurrent.futures.TimeoutError:
            print("time out!")

        # Expand dimension by 1 (add channel dim).
        return np.expand_dims(mat, axis=1)

    def simulate_stdpopsim(self, engine, species, model, pop, error_prob=None):
        """Generate simulated data from stdpopsim"""

        # Set variables for the simulator
        stdengine = stdpopsim.get_engine(engine)
        stdspecies = stdpopsim.get_species(species)
        stdmodel = stdspecies.get_demographic_model(model)

        # Sample genotype location with odds weighted by chromosome length
        geno = [(i, get_chrom_size(i)) for i in range(1, 23)]
        geno.sort(key=lambda a: a[1], reverse=True)
        cum_weights = []
        for i, (chrom, size) in enumerate(geno):
            cum_weights.append(size if i == 0 else size + cum_weights[i - 1])

        # The order for sampling from populations is ['YRI', 'CEU', 'CHB']
        if pop == "YRI":
            stdsamples = stdmodel.get_samples(self.num_samples, 0, 0)
        elif pop == "CEU":
            stdsamples = stdmodel.get_samples(0, self.num_samples, 0)
        elif pop == "CHB":
            stdsamples = stdmodel.get_samples(0, 0, self.num_samples)

        # Perform simulations
        sims = []
        for i in range(self.num_reps):
            chrom, size = self.rng.choices(geno, cum_weights=cum_weights)[0]
            factor = self.seq_len / size
            stdcontig = stdspecies.get_contig(
                "chr" + str(chrom), length_multiplier=factor
            )
            sims.append(stdengine.simulate(stdmodel, stdcontig, stdsamples))

        mat = np.zeros((self.num_reps, self.num_samples, self.fixed_dim))

        # Resize from ts, and add sequencing errors if error_prob is given
        for i, ts in enumerate(sims):
            if type(error_prob) is float:
                mat[i] = self._mutate_geno_old(ts, p=error_prob)

            elif type(error_prob) is np.ndarray:
                mat[i] = self._mutate_geno_old(ts, p=error_prob[i])

            # No error prob, don't mutate the matrix
            else:
                mat[i] = self._resize_from_ts(ts)

        # Expand dimension by 1 (add channel dim)
        return np.expand_dims(mat, axis=1)

    def generate_data(self, num_reps, proposals=False):
        """Generate (X, y) labelled data from demographic simulations. The labels
        are y=0 for simulated data and y=1 for data with parameters to infer."""

        self.num_reps = num_reps
        # Generate genotype matrices from the data with params to infer
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
            gen1 = self.parse_empirical_data(haplotype=0)

        elif self.source == "msprime":
            gen1 = self.simulate_msprime(self.params)

        # Generate genotype matrices from the simulated data
        print(f"generating {num_reps} genotype matrices from msprime")
        gen0 = self.simulate_msprime(self.params, randomize=True, proposals=proposals)

        # Assemble data and labels
        X = np.concatenate((gen1, gen0))
        y = np.concatenate((np.ones((num_reps)), np.zeros((num_reps))))
        print(f"X data shape is: {X.shape}")

        # Split randomly into training and test data.
        return train_test_split(X, y, test_size=0.1, random_state=self.seed)

    def generate_fakedata(self, num_reps, testlist=None):
        """Generate a batch of only simulated data.
        testlist: list of parameter values that I use for debugging
        """

        self.num_reps = num_reps
        print(f"generating {num_reps} genotype matrices from msprime for testing")
        return self.simulate_msprime(self.params, randomize=True)

    def _mutate_geno(self, ts, p=0.001):
        """Mutate a tree sequence simulation introducing sequencing errors
        with probability p. Then, create and resize a genotype matrix"""

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
            if ac1 > ac0 or (ac1 == ac0 and self.rng.random() > 0.5):
                genotypes ^= 1

            n = np.random.binomial(len(variant.genotypes), p)
            if n is not None:
                idx = np.random.randint(0, len(variant.genotypes), size=n)
                variant.genotypes[idx] = 1 - variant.genotypes[idx]

            j = int(variant.site.position * self.fixed_dim / ts.sequence_length)
            np.add(
                m[:, j], variant.genotypes, out=m[:, j], where=variant.genotypes != -1
            )

        return m

    def _random_sampling_geno(self, callset):
        """Random sampling of a genomic window with the odds weighted by
        chromosome length. If a genomic mask is given, it filters regions with
        low quantity of callable regions"""

        # Extract chromosome number and length from stdpopsim catalog
        geno = [(i, get_chrom_size(i)) for i in range(1, 23)]

        # Sort the list by size.
        geno.sort(key=lambda a: a[1], reverse=True)

        # Get cumulative weights for each chromosome for sampling
        cum_weights = []
        for i, (chrom, size) in enumerate(geno):
            cum_weights.append(size if i == 0 else size + cum_weights[i - 1])

        print("Charging up the chromosomes")
        locs = [0]
        for i in range(1, 23):
            print(f"Charging chromosome {i}", end="\r")
            query = f"{i}/variants/POS"
            locs.append(np.asarray(callset[query]))

        # Load mask if given
        mask = load_mask(self.mask_file) if self.mask_file else None

        # Initialize empty variables
        chroms, slices, pos = [], [], []

        # Sample until we obtain the desired number of genomic regions
        while len(chroms) < self.num_reps:
            chrom, size = self.rng.choices(geno, cum_weights=cum_weights)[0]
            assert size > self.seq_len

            # Check if the proposed regions falls mostly inside the mask
            start_proposal = self.rng.randrange(0, size - self.seq_len)
            if mask:
                if not inside_mask(mask, start_proposal, chrom, self.seq_len):
                    continue

            # If region is OK, append the region data to the lists
            chroms.append(chrom)
            pos.append(start_proposal)
            slices.append(
                locate(
                    locs[chrom],
                    start=start_proposal,
                    stop=start_proposal + self.seq_len,
                )
            )

        return chroms, pos, slices

    def _resize_from_ts(self, ts):
        """Returns a genotype matrix with a fixed number of columns,
        as specified in size"""

        # Initialize empty matrix with the new dimensions
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
            if ac1 > ac0 or (ac1 == ac0 and self.rng.random() > 0.5):
                genotypes ^= 1

            j = int(variant.site.position * self.fixed_dim / ts.sequence_length)
            m[:, j] += genotypes

        return m.astype(float)

    def _resize_from_zarr(self, mat, pos, alts):
        """Resizes a matrix using a sum window, given a genotype matrix,
        positions vector,sequence length and the desired fixed size
        of the new matrix"""

        # Initialize empty matrix with the new dimensions
        m = np.zeros((mat.shape[1], self.fixed_dim), dtype=mat.dtype)
        ac_thresh = self.maf_thresh * mat.shape[1]

        # Fill in the resized matrix
        for _pos, _gt, _alt in zip(pos, mat, alts):

            # Filter by MAF
            ac1 = np.sum(_gt)
            ac0 = len(_gt) - ac1
            if min(ac0, ac1) < ac_thresh:
                continue

            # Polarise 0 and 1 in genotype matrix by major allele frequency.
            # If allele counts are the same, randomly choose a major allele.
            if ac1 > ac0 or (ac1 == ac0 and self.rng.random() > 0.5):
                _gt ^= 1

            j = int(_pos * self.fixed_dim / self.seq_len) - 1
            np.add(m[:, j], _gt, out=m[:, j], where=_gt != -1)

        return m.astype(float)


def haploidify(genmat, h):
    """Returns the selected haplotype from a numpy array with
    a ploidy dimension. The parameter h must be either 0, 1 or 2"""

    if h in [0, 1, 2]:
        if h == 2:
            return np.concatenate((genmat[:, :, 0], genmat[:, :, 1]))
        else:
            return genmat[:, :, h]

    print("The parameter h must be 0 or 1 for one haplotype, or 2 for both")


def load_mask(mask_file):
    """Given a mask file in BED format, parse the mask data and
    returns a matrix of tuples containing the permited regions,
    as (start, end) positions"""

    # Initialize empty mask dictionary
    mask = {str(k): [] for k in range(1, 23)}

    # Read through the lines and add to the dictionary for each chrom
    with open(mask_file, "r") as file:
        for line in file:
            chrom, start, end, _ = line.split()
            mask[chrom[3:]].append((int(start), int(end)))
        file.close()

    return mask


def inside_mask(mask, first, chrom, seq_len, threshold=0.7):
    """Check whether a proposal with starting position 'first' falls inside
    the given mask, with a number of bp inside higher than the threshold"""

    inside = 0
    # Calculate ending position of the proposal
    last = first + seq_len

    # For each permited genomic window within the mask
    for i, (start, end) in enumerate(mask[str(chrom)]):
        # If the start of proposal is inside this range, save the number of bp
        if start <= first < end:
            inside = end - first

            # Look for the genomic window where the proposal ends in the rest
            # of the mask, summing up the number of bp inside the mask meanwhile
            for start, end in mask[str(chrom)][i:]:

                # If found, break the loop and sum the last bps
                if start <= last < end:
                    inside += last - start
                    break
                else:
                    inside += end - start
            break

    # Return True if the fraction of bp inside the mask is higher than threshold
    return True if inside / seq_len > threshold else False


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
    """Plot a given genotype matrix"""

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
        "-z",
        "--zarr-path",
        help="Path pointing to the directory with the zarr objects containing genomic data",
        default="",
        type=str,
    )

    parser.add_argument(
        "-m",
        "--mask-file",
        help="Genomic mask to use for random sampling of VCF files from empirical data",
        default="",
        type=str,
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

    # Build the Genobuilder object
    genob = Genobuilder(
        source=args.source,
        num_samples=args.number_haplotypes,
        seq_len=args.sequence_length,
        maf_thresh=args.maf_threshold,
        fixed_dim=args.fixed_dimension,
        zarr_path=args.zarr_path,
        mask_file=args.mask_file,
        seed=args.seed,
        parallelism=args.parallelism,
    )

    if len(params_dict.keys()) >= 1:
        genob.set_parameters(sim_source="msprime", params=params_dict)
    else:
        print("No parameters detected in the parameter dictionary")
        genob = None

    # When the user only wants the Genobuilder object as a pickled file
    if genob and args.function == "init":

        output = str(args.output) + ".pkl"
        with open(output, "wb") as obj:
            pickle.dump(genob, obj, protocol=pickle.HIGHEST_PROTOCOL)

    # When the user wants the pickled Genobuilder and also genotype matrices
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
# python genobuilder.py download_genmats -n 1000 -s empirical -z data/zarr -m data/20140520.pilot_mask.autosomes.bed -se 2020 -o test -p 64
