from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Base class for torch neural network models
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model
        :param x: input torch tensor
        :return: output torch tensor
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_hyper_parameters(self) -> dict:
        """
        get the model hyper-parameters
        :return: model hyper-parameters
        """
        raise NotImplementedError

