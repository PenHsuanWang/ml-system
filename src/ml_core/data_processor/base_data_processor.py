import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

# Configure logging
logger = logging.getLogger(__name__)


class BaseDataProcessor(ABC):
    """
    Base class for data processor
    """
    def __init__(self, input_data=None):
        """
        :param input_data: pandas dataframe
        """
        self._input_df = input_data
        self._training_data_x = None
        self._training_target_y = None
        self._testing_data_x = None
        self._testing_target_y = None
        self._is_preprocessed = False

    def set_input_data(self, input_data: pd.DataFrame):
        """
        Set the input data
        :param input_data: pandas dataframe
        :return:
        """
        self._input_df = input_data
        self._is_preprocessed = False
        logger.debug("Input data set. Marked as not preprocessed.")

    def preprocess_data(self, force=False):
        """
        A high-level method that preprocesses the data.
        """
        if self._is_preprocessed and not force:
            logger.info("Data has already been preprocessed. Skipping preprocessing.")
            return

        if self._input_df is None:
            raise RuntimeError("Input data is not provided.")

        logger.info("Starting data preprocessing.")
        self._scaling_array()
        self._splitting()
        self._preprocessing()
        self._is_preprocessed = True
        logger.info("Data preprocessing completed.")

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
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='split')
            return obj

        return {
            '_input_df': convert_to_native(self._input_df),
            '_training_data_x': convert_to_native(self._training_data_x),
            '_training_target_y': convert_to_native(self._training_target_y),
            '_testing_data_x': convert_to_native(self._testing_data_x),
            '_testing_target_y': convert_to_native(self._testing_target_y),
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
