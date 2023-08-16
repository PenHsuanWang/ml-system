from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Base class for torch neural network models
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        forward pass
        :param x: input
        :return: output
        """
        raise NotImplementedError

