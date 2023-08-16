from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    A trainer is an object that handles the training process of a model
    """

    def __init__(self, model):
        self._model = model

    @abstractmethod
    def run_training_loop(self, epochs: int) -> None:
        raise NotImplementedError



