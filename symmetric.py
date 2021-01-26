import torch
import torch.nn as nn


class Symmetric(nn.Module):
    """Class for a symmetric layer for permutation invariant CNNs. This layer
    collapses the dimension specified in the given axis using a summary statistic
    """

    def __init__(self, function, axis, **kwargs):
        self.function = function
        self.axis = axis
        super(Symmetric, self).__init__(**kwargs)

    def forward(self, x):
        if self.function == "sum":
            out = torch.sum(x, dim=self.axis, keepdim=True)
        elif self.function == "mean":
            out = torch.mean(x, dim=self.axis, keepdim=True)
        elif self.function == "min":
            # torch.min and torch.max returns: (values, indices)
            out = torch.min(x, dim=self.axis, keepdim=True)[0]
        elif self.function == "max":
            out = torch.max(x, dim=self.axis, keepdim=True)[0]
        return out
