import pandas as pd
import numpy as np
import torch

from src.ml_core.data_processor.base_data_processor import BaseDataProcessor
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataProcessor(BaseDataProcessor):
    """
    Concrete class for process time series data.
    """

    def __init__(
            self,
            extract_column: list[str],
            training_data_ratio: float = 0.8,
            training_window_size: int = 60,
            target_window_size: int = 1,
            input_data=None
    ):
        """
        Provide the raw dataframe as input parameters. Together with the desired column name.
        Extract the target column from the raw dataframe and convert to pytorch tensor.
        Splitting into training and testing set.
        provide the time window size and padding step to build training and testing set.
        :param input_data:
        :param extract_column:
        :param training_data_ratio:
        :param training_window_size:
        :param target_window_size:
        """
        super().__init__(input_data)

        # These object fields are used to extract desired data from raw dataframe by column name.
        # And scale the extracted data using MinMaxScaler.
        self._extract_column = extract_column
        self._extract_data_as_numpy_array = None
        self._scaler_by_column = {}

        # These object fields are used to split the extracted data into training and testing set.
        self._training_data_ratio = training_data_ratio
        self._training_array = None
        self._testing_array = None

        self._training_window_size = training_window_size
        self._target_window_size = target_window_size

    def _scaling_array(self):
        """
        Implement the scaling in the internal function. _extract_training_data_and_scale()
        due to multiple columns to extract is possible. Have to do the scaling after all the columns are extracted.
        and store all the scaler object in self._scaler_by_column for future use.
        :return:
        """
        self._extract_training_data_and_scale()

    def _splitting(self):
        """
        Using the transform
        :return:
        """
        self._training_array = self._extract_data_as_numpy_array[
                               :int(self._training_data_ratio * len(self._extract_data_as_numpy_array))
                               ]
        self._testing_array = self._extract_data_as_numpy_array[
                                int(self._training_data_ratio * len(self._extract_data_as_numpy_array)) - self._training_window_size:
                              ]

    def _preprocessing(self):
        """
        Support the extraction of column data as numpy array has been implemented.
        splitting training and test data has been implemented as well.
        Now implement the padding and sliding window to build the training and testing set.
        to make all model training material ready.
        :return:
        """

        # return object from sliding window mask is a list of numpy array.
        x_train, y_train = self._sliding_window_mask(self._training_array)
        x_test, y_test = self._sliding_window_mask(self._testing_array)

        # convert the list of numpy array to multidimensional numpy array.
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

        # convert the numpy array to pytorch tensor.
        # assign the tensor to object field.
        self._training_data_x = x_train
        self._training_target_y = y_train

        self._testing_data_x = x_test
        self._testing_target_y = y_test

    def _extract_training_data_and_scale(self) -> None:
        """
        Internal function to extract data from raw dataframe based on the provided column names list.
        And scale the extracted data using MinMaxScaler. store the scaler object in self._scaler_by_column
        :return:
        """
        if len(self._extract_column) == 0:
            raise ValueError("Extract column is not provided.")

        for i_column in self._extract_column:
            extract_column_series = self._input_df[i_column].values
            if extract_column_series is None:
                raise ValueError(f"Please check the column {i_column} exist in the input data!")
            extract_column_series = extract_column_series.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(extract_column_series)
            self._scaler_by_column[i_column] = scaler
            extract_column_series = scaler.transform(extract_column_series)
            if self._extract_data_as_numpy_array is None:
                self._extract_data_as_numpy_array = extract_column_series
            else:
                self._extract_data_as_numpy_array = np.concatenate((self._extract_data_as_numpy_array, extract_column_series), axis=1)

    def _sliding_window_mask(self, input_array: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Sliding window mask on provided data, return training data and target data,
        """
        x = []
        y = []

        for i in range(self._training_window_size, len(input_array) - self._target_window_size + 1):
            x.append(input_array[i - self._training_window_size:i])
            y.append(input_array[i:i + self._target_window_size, 0])

        return x, y

    def inverse_testing_scaler(self, data: np.ndarray, scaler_by_column_name: str) -> np.ndarray:
        """
        Inverse the scaling of the testing data.
        :param data:
        :param scaler_by_column_name:
        :return:
        """
        scaler = self._scaler_by_column[scaler_by_column_name]
        return scaler.inverse_transform(data)
