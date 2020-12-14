import msprime
import math


def onepop_constant(args):
    """Single population model with pop size Ne and constant growth"""

    genob, params, randomize, i, proposals, seed = args
    necessary_params = ["mu", "r", "Ne"]
    assert sorted(necessary_params) == sorted(
        list(params.keys())
    ), "Invalid combination of parameters. Needed: mu | r | Ne"

    if proposals:
        mu, r, Ne = [
            params[p].prop(i) if params[p].inferable else params[p].val
            for p in necessary_params
        ]
    else:
        mu, r, Ne = [
            params[p].rand() if randomize else params[p].val for p in necessary_params
        ]

    ts = msprime.simulate(
        sample_size=genob.num_samples,
        Ne=Ne,
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        random_seed=seed,
    )

    return ts


def onepop_exp(args):
    """Single population model with sudden population size increase from N1 to N2
    at time T1 and exponential growth at time T2"""

    genob, params, randomize, i, proposals, seed = args
    necessary_params = ["mu", "r", "T1", "N1", "T2", "N2", "growth"]
    assert sorted(necessary_params) == sorted(
        list(params.keys())
    ), "Invalid combination of parameters. Needed: mu | r | T1 | N1 | T2 | N2 | growth"

    if proposals:
        mu, r, T1, N1, T2, N2, growth = [
            params[p].prop(i) if params[p].inferable else params[p].val
            for p in necessary_params
        ]
    else:
        mu, r, T1, N1, T2, N2, growth = [
            params[p].rand() if randomize else params[p].val for p in necessary_params
        ]

    N0 = N2 / math.exp(growth * T2)

    # Time is given in generations unit (t/25)
    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0, growth_rate=growth),
        msprime.PopulationParametersChange(time=T2, initial_size=N2, growth_rate=0),
        msprime.PopulationParametersChange(time=T1, initial_size=N1),
    ]

    ts = msprime.simulate(
        sample_size=genob.num_samples,
        demographic_events=demographic_events,
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        random_seed=seed,
    )

    return ts


def onepop_migration(args):
    """Mass migration at time T1 from population 1 with pop size N2 to population
    0 with pop size N1. Samples are collected only from population 0."""

    genob, params, randomize, i, proposals, seed = args
    necessary_params = ["mu", "r", "T1", "N1", "N2", "mig"]
    assert sorted(necessary_params) == sorted(list(params.keys())), (
        "Invalid combination of parameters. Needed: mu | r | T1 | N1 | N2 | mig \n"
        f"Obtained: {list(params.keys())}"
    )

    if proposals:
        mu, r, T1, N1, N2, mig = [
            params[p].prop(i) if params[p].inferable else params[p].val
            for p in necessary_params
        ]
    else:
        mu, r, T1, N1, N2, mig = [
            params[p].rand() if randomize else params[p].val for p in necessary_params
        ]

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=genob.num_samples, initial_size=N1),
        msprime.PopulationConfiguration(sample_size=0, initial_size=N2),
    ]

    # migration from pop 1 into pop 0 (back in time)
    mig_event = msprime.MassMigration(time=T1, source=1, destination=0, proportion=mig)

    ts = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=[mig_event],
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        random_seed=seed,
    )

    return ts
