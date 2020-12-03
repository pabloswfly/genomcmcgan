import numpy as np


class Parameter:
    def __init__(
        self, name, val, initial_guess, bounds, inferable, log=False, **kwargs
    ):

        self.name = name
        if log:
            self._val = np.float_power(10, val)
        else:
            self._val = val
        self.bounds = bounds
        self.log = log
        self.initial_guess = initial_guess
        self.inferable = inferable
        super(Parameter, self).__init__(**kwargs)

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, x):
        if self.log:
            self._val = np.float_power(10, x)
        else:
            self._val = x

    def set_gauss(self, mean, std):

        self.gauss_mean = mean
        self.gauss_std = std

    def rand(self, gauss=False):

        if not self.inferable:
            return self._val

        if gauss:
            if self.log:
                return np.float_power(
                    10, np.random.normal(self.gauss_mean, self.gauss_std)
                )
            else:
                return np.random.normal(self.gauss_mean, self.gauss_std)
        else:
            min, max = self.bounds
            x = np.random.uniform(min, max)

            if self.log:
                return np.float_power(10, x)
            else:
                return x
