from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseDataProcessor(ABC):
    """
    Abstract base class for all data processors.
    """

    def __init__(self, input_data):
        """
        The data processor is designed as an operator to provide data transformation.
        The input data is a pandas dataframe, and the output data after desired transformation is a numpy array.
        :param input_data: the input data is a pandas dataframe.
        """
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
        Get the training data x in numpy array format.
        :return:
        """
        return self._training_data_x

    def get_training_target_y(self) -> np.ndarray:
        """
        Get the training target y in numpy array format.
        :return:
        """
        return self._training_target_y

    def get_testing_data_x(self) -> np.ndarray:
        """
        Get the testing data x in numpy array format.
        The testing data size could be zero.
        :return:
        """
        return self._testing_data_x

    def get_testing_target_y(self) -> np.ndarray:
        """
        Get the testing target y in numpy array format.
        The testing data size could be zero.
        :return:
        """
        return self._testing_target_y

    def to_dict(self):
        """
        Serialize the BaseDataProcessor object to a dictionary.
        """
        return {
            '_input_df': self._input_df.to_dict(orient='split') if self._input_df is not None else None,
            '_training_data_x': self._training_data_x.tolist() if self._training_data_x is not None else None,
            '_training_target_y': self._training_target_y.tolist() if self._training_target_y is not None else None,
            '_testing_data_x': self._testing_data_x.tolist() if self._testing_data_x is not None else None,
            '_testing_target_y': self._testing_target_y.tolist() if self._testing_target_y is not None else None,
            '_is_preprocessed': self._is_preprocessed
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Deserialize a dictionary to a BaseDataProcessor object.
        """
        obj = cls(input_data=pd.DataFrame(data['_input_df']) if data['_input_df'] is not None else None)
        obj._training_data_x = np.array(data['_training_data_x']) if data['_training_data_x'] is not None else None
        obj._training_target_y = np.array(data['_training_target_y']) if data['_training_target_y'] is not None else None
        obj._testing_data_x = np.array(data['_testing_data_x']) if data['_testing_data_x'] is not None else None
        obj._testing_target_y = np.array(data['_testing_target_y']) if data['_testing_target_y'] is not None else None
        obj._is_preprocessed = data['_is_preprocessed']
        return obj
