from tensorflow import keras


class Symmetric(keras.layers.Layer):
    """Class of keras layer from permutation invariant cnn. This layer collapses
    the dimension specified in the given axis using a summary statistic"""

    def __init__(self, function, axis, **kwargs):
        self.function = function
        self.axis = axis
        super(Symmetric, self).__init__(**kwargs)

    def call(self, x):
        if self.function == "sum":
            out = keras.backend.sum(x, axis=self.axis, keepdims=True)
        if self.function == "mean":
            out = keras.backend.mean(x, axis=self.axis, keepdims=True)
        if self.function == "min":
            out = keras.backend.min(x, axis=self.axis, keepdims=True)
        if self.function == "max":
            out = keras.backend.max(x, axis=self.axis, keepdims=True)
        return out

    # Without this, Its not possible to load and save the model
    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "function": self.function,
                "axis": self.axis,
            }
        )
        return config
