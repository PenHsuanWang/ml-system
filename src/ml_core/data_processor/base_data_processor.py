from abc import ABC, abstractmethod
import pandas as pd

# An abstraction class for data processor
# Which designed fulfill different type of pytorch model
# Converting raw pandas dataframe to pytorch input tensor


class BaseDataProcessor(ABC):
    """
    Abstract base class for all data processor.
    """

    def __init__(self, input_data):
        """
        :type input_data: pd.DataFrame
        :param input_data:
        """
        # if input data is not provided, set it to None in initialization stage is okey, but must be provided later.
        self._input_df = input_data
        self._training_tensor = None
        self._training_target_tensor = None
        self._testing_tensor = None
        self._testing_target_tensor = None

    def set_input_df(self, input_data: pd.DataFrame):
        self._input_df = input_data

    def preprocess_data(self):
        """A high-level method that preprocesses the data."""

        # check if input data is provided, or raise an process error
        if self._input_df is None:
            raise RuntimeError("Input data is not provided.")

        self._scaling_array()
        self._splitting()
        self._preprocessing()

    @abstractmethod
    def _scaling_array(self):
        raise NotImplementedError

    @abstractmethod
    def _splitting(self):
        raise NotImplementedError

    @abstractmethod
    def _preprocessing(self):
        raise NotImplementedError

    def get_training_tensor(self):
        return self._training_tensor

    def get_training_target_tensor(self):
        return self._training_target_tensor

    def get_testing_tensor(self):
        return self._testing_tensor

    def get_testing_target_tensor(self):
        return self._testing_target_tensor



