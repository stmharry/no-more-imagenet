from torch import nn


class Model(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        nn.Module.__init__(instance)
        return instance
