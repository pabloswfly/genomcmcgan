import msprime
import math


def constant(args):
    """Single population model with pop size Ne and constant growth"""

    genob, params, randomize, i, proposals = args
    necessary_params = ["mu", "r", "Ne"]
    for p in necessary_params:
        if p not in list(params.keys()):
            print("Invalid combination of parameters. Needed: mu | r | Ne")

    if proposals:
        mu, r, Ne = [
            params[p].prop(i) if params[p].inferable else params[p].val
            for p in necessary_params
        ]
    else:
        mu, r, Ne = [
            params[p].rand() if randomize else params[p].val for p in necessary_params
        ]

    return msprime.simulate(
        sample_size=genob.num_samples,
        Ne=Ne,
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        # random_seed=genob.seed,
    )


def exponential(args):
    """Single population model with sudden population size increase from N1 to N2
    at time T1 and exponential growth at time T2"""

    genob, params, randomize, i, proposals = args
    necessary_params = ["mu", "r", "T1", "N1", "T2", "N2", "growth"]
    for p in necessary_params:
        if p not in list(params.keys()):
            print(
                "Invalid combination of parameters. "
                "Needed: mu | r | T1 | N1 | T2 | N2 | growth"
            )

    if proposals:
        mu, r, T1, N1, T2, N2, growth = [
            params[p].prop(i) if params[p].inferable else params[p].val
            for p in necessary_params
        ]
    else:
        mu, r, T1, N1, T2, N2, growth = [
            params[p].rand() if randomize else params[p].val for p in necessary_params
        ]

    N0 = N1 / math.exp(-growth * T1)

    # Time is given in generations unit (t/25)
    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0, growth_rate=growth),
        msprime.PopulationParametersChange(time=T1, initial_size=N1),
        msprime.PopulationParametersChange(time=T2, initial_size=N2),
    ]

    return msprime.simulate(
        sample_size=genob.num_samples,
        demographic_events=demographic_events,
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        # random_seed=genob.seed,
    )


def zigzag(args):
    """Derived model used by Schiffels and Durbin (2014) and Terhorst and
    Terhorst, Kamm, and Song (2017) with periods of exponential growth and
    decline in a single population. Here, growth rates are changed to pop sizes.
    Schiffels and Durbin, 2014. https://doi.org/10.1038/ng.3015"""

    genob, params, randomize, i, proposals = args
    necessary_params = [
        "mu",
        "r",
        "T1",
        "N1",
        "T2",
        "N2",
        "T3",
        "N3",
        "T4",
        "N4",
        "T5",
        "N5",
    ]
    for p in necessary_params:
        if p not in list(params.keys()):
            print(
                "Invalid combination of parameters. Needed: "
                "mu | r | T1 | N1 | T2 | N2 | T3 | N3 | T4 | N4 | T5 | N5"
            )

    if proposals:
        mu, r, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5 = [
            params[p].prop(i) if params[p].inferable else params[p].val
            for p in necessary_params
        ]
    else:
        mu, r, T1, N1, T2, N2, T3, N3, T4, N4, T5, N5 = [
            params[p].rand() if randomize else params[p].val for p in necessary_params
        ]

    generation_time = 30
    N0 = 71560
    n_ancient = N0 / 10
    t_ancient = 34133.318528

    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0, population_id=0),
        msprime.PopulationParametersChange(time=T1, initial_size=N1, population_id=0),
        msprime.PopulationParametersChange(time=T2, initial_size=N2, population_id=0),
        msprime.PopulationParametersChange(time=T3, initial_size=N3, population_id=0),
        msprime.PopulationParametersChange(time=T4, initial_size=N4, population_id=0),
        msprime.PopulationParametersChange(time=T5, initial_size=N5, population_id=0),
        msprime.PopulationParametersChange(
            time=t_ancient, initial_size=n_ancient, population_id=0
        ),
    ]

    return msprime.simulate(
        sample_size=genob.num_samples,
        demographic_events=demographic_events,
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        # random_seed=genob.seed,
    )


def bottleneck(args):

    genob, params, randomize, i, proposals = args
    necessary_params = ["mu", "r", "N0", "T1", "N1", "T2", "N2"]

    for p in necessary_params:
        if p not in list(params.keys()):
            print(
                "Invalid combination of parameters. Needed: "
                "mu | r | N0 | T1 | N1 | T2 | N2"
            )

    if proposals:
        mu, r, N0, T1, N1, T2, N2 = [
            params[p].prop(i) if params[p].inferable else params[p].val
            for p in necessary_params
        ]
    else:
        mu, r, N0, T1, N1, T2, N2 = [
            params[p].rand() if randomize else params[p].val for p in necessary_params
        ]

    # Infer the 3 pop sizes, where N0 = N2
    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0),
        msprime.PopulationParametersChange(time=T1, initial_size=N1),
        msprime.PopulationParametersChange(time=T2, initial_size=N2),
    ]

    return msprime.simulate(
        sample_size=genob.num_samples,
        demographic_events=demographic_events,
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        # random_seed=genob.seed,
    )



def ghost_migration(args):
    """Mass migration at time T1 from population 1 with pop size N2 to population
    0 with pop size N1. Samples are collected only from population 0."""

    genob, params, randomize, i, proposals = args
    necessary_params = ["mu", "r", "T1", "N1", "N2", "mig"]
    for p in necessary_params:
        if p not in list(params.keys()):
            print(
                "Invalid combination of parameters. "
                "Needed: mu | r | T1 | N1 | N2 | mig"
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

    return msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=[mig_event],
        length=genob.seq_len,
        mutation_rate=mu,
        recombination_rate=r,
        # random_seed=genob.seed,
    )
