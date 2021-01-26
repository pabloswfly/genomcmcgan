import numpy as np


class Parameter:
    def __init__(
        self, name, val, bounds, inferable, log=False, plotlog=False, **kwargs
    ):

        self.name = name
        self._val = val
        self._proposals = []
        self.bounds = bounds
        self.log = log
        self.init = np.random.uniform(bounds[0], bounds[1])
        self.step_size = 0.1
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
            # RandomState() avoids getting the same random number from different
            # CPUs when running the code in several processes in parallel
            x = np.random.RandomState().uniform(np.log10(min), np.log10(max))
            return np.float_power(10, x)
        else:
            return np.random.RandomState().uniform(min, max)
