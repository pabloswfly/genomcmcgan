import numpy as np


class Parameter:
    def __init__(
        self,
        name,
        val,
        initial_guess,
        bounds,
        inferable,
        log=False,
        plotlog=False,
        **kwargs
    ):

        self.name = name
        self._val = val
        self._proposals = []
        self.bounds = bounds
        self.log = log
        self.initial_guess = initial_guess
        self.inferable = inferable
        self.plotlog = plotlog
        super(Parameter, self).__init__(**kwargs)

    @property
    def val(self):
        return self._val

    @property
    def proposals(self):
        return self._proposals

    @val.setter
    def val(self, x):
        self._val = x

    @proposals.setter
    def proposals(self, prop):
        self._proposals = prop

    def prop(self, i):
        return self.proposals[i]

    def rand(self):

        if not self.inferable:
            return self._val

        min, max = self.bounds
        if self.log:
            x = np.random.uniform(np.log10(min), np.log10(max))
            return np.float_power(10, x)
        else:
            return np.random.uniform(min, max)
