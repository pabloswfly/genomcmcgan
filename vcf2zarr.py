import pysam
import allel
import argparse
import numpy as np


def samples_from_population(pop_file):
    """Return a list of sample IDs given a population file from the
    online repository of 1000 genomes project"""

    return np.loadtxt(pop_file, dtype=str, delimiter="\t", skiprows=1)[:, 0]


def vcf2zarr(vcf_files, pop_file, zarr_path):
    # Two veery good tutorials:
    # http://alimanfoo.github.io/2018/04/09/selecting-variants.html
    # http://alimanfoo.github.io/2017/06/14/read-vcf.html
    # TODO: Refactor to work without pysam and allel

    # Get a list of the wanted samples from one population which are found in
    # the VCF files. The files must be numbered, and that number must be
    # substituted in the input path string with {n}.
    first_vcf = pysam.VariantFile(vcf_files.replace("{n}", "1"))
    wanted_samples = samples_from_population(pop_file)
    found_samples = list(
        set(wanted_samples).intersection(list(first_vcf.header.samples))
    )

    # Create one zarr folder for each chromosome
    for chrom in range(1, 23):
        vcf = vcf_files.replace("{n}", str(chrom))
        print(f"Creating zarr object for chromosome {chrom}")
        allel.vcf_to_zarr(
            vcf,
            zarr_path,
            group=str(chrom),
            region=str(chrom),
            fields=["POS", "ALT", "samples", "GT"],
            samples=found_samples,
            overwrite=True,
        )

    print("VCF data transformed into Zarr objects")


if __name__ == "__main__":

    # Parser object to collect user input from terminal
    parser = argparse.ArgumentParser(
        description="Parse the variant data from VCF files from the "
        "desired population into a Zarr object"
    )

    parser.add_argument(
        "-f",
        "--vcf-files",
        help="Path pointing to the first of the VCF files to be parsed. "
        "The files can be zipped as .gz. Data transfer is quicker when "
        "the files are indexed with Tabix, ending in .tbi.",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--population-file",
        help="File in .tsv format containing the samples ID for the desired population.",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path where the Zarr folders and objects will be stored.",
        type=str,
    )

    # Get argument values from parser
    args = parser.parse_args()
    vcf2zarr(args.vcf_files, args.population_file, args.output)

# Example command:
# python vcf2zarr.py -f /willerslev/users-shared/science-snm-willerslev-dsw670/projects/sandbox/20180601-archaic.1000g/vcf/{n}.1000g.archaic.vcf.gz -p data/igsr-ceu.tsv.tsv -o data/zarr
