from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# An abstraction class for data processor
# Which designed fulfill different type of pytorch model
# Converting raw pandas dataframe to pytorch input tensor


class BaseDataProcessor(ABC):
    """
    Abstract base class for all data processor.
    """

    def __init__(self, input_data):
        """
        The data processor is to component designed as a operator to provide data transformation.
        The input data is a pandas dataframe. and the output data after desired transformation is a numpy array.
        :param input_data: the input data is a pandas dataframe.
        """
        # if input data is not provided, set it to None in initialization stage is okey, but must be provided later.
        self._input_df = input_data
        self._training_data_x = None
        self._training_target_y = None
        self._testing_data_x = None
        self._testing_target_y = None
        self._is_preprocessed = False

    def set_input_df(self, input_data: pd.DataFrame):
        self._input_df = input_data
        self._is_preprocessed = False

    def preprocess_data(self):
        """A high-level method that preprocesses the data."""

        if self._is_preprocessed:
            print("Data has already been preprocessed. Skipping preprocessing.")
            return

        # check if input data is provided, or raise an process error
        if self._input_df is None:
            raise RuntimeError("Input data is not provided.")

        self._scaling_array()
        self._splitting()
        self._preprocessing()
        self._is_preprocessed = True

    @abstractmethod
    def _scaling_array(self):
        raise NotImplementedError

    @abstractmethod
    def _splitting(self):
        raise NotImplementedError

    @abstractmethod
    def _preprocessing(self):
        raise NotImplementedError

    def get_training_data_x(self) -> np.ndarray:
        """
        get the training data x in numpy array format.
        :return:
        """
        return self._training_data_x

    def get_training_target_y(self) -> np.ndarray:
        """
        get the training target y in numpy array format.
        :return:
        """
        return self._training_target_y

    def get_testing_data_x(self) -> np.ndarray:
        """
        get the testing data x in numpy array format.
        the testing data size could be zero.
        :return:
        """
        return self._testing_data_x

    def get_testing_target_y(self) -> np.ndarray:
        """
        get the testing target y in numpy array format.
        the testing data size could be zero.
        :return:
        """
        return self._testing_target_y

