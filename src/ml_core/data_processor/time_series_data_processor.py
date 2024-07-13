import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.ml_core.data_processor.base_data_processor import BaseDataProcessor

class TimeSeriesDataProcessor(BaseDataProcessor):
    """
    Inherit from BaseDataProcessor, the data transformation from pandas dataframe to numpy array.
    The data transformation based on time series data. The data is extracted from pandas dataframe by column name.
    The core part of the data transformation is sliding window mask. The sliding window mask is used to build
    Get the output data with desired folded time window size.
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

        # Preprocess data if input_data is provided
        if input_data is not None:
            self.preprocess_data()

    def to_dict(self):
        """
        Serialize the TimeSeriesDataProcessor object to a dictionary.
        """
        data_dict = super().to_dict()
        data_dict.update({
            'type': 'TimeSeriesDataProcessor',
            'extract_column': self._extract_column,
            'training_data_ratio': self._training_data_ratio,
            'training_window_size': self._training_window_size,
            'target_window_size': self._target_window_size,
        })
        return data_dict

    @classmethod
    def from_dict(cls, data: dict):
        """
        Deserialize a dictionary to a TimeSeriesDataProcessor object.
        """
        instance = cls(
            extract_column=data.get('extract_column', []),
            training_data_ratio=data.get('training_data_ratio', 0.8),
            training_window_size=data.get('training_window_size', 60),
            target_window_size=data.get('target_window_size', 1),
            input_data=pd.DataFrame(data['_input_df']['data'], columns=data['_input_df']['columns']) if data['_input_df'] is not None else None
        )
        instance._training_data_x = np.array(data['_training_data_x']) if data['_training_data_x'] is not None else None
        instance._training_target_y = np.array(data['_training_target_y']) if data['_training_target_y'] is not None else None
        instance._testing_data_x = np.array(data['_testing_data_x']) if data['_testing_data_x'] is not None else None
        instance._testing_target_y = np.array(data['_testing_target_y']) if data['_testing_target_y'] is not None else None
        instance._is_preprocessed = data['_is_preprocessed']
        if instance._input_df is not None:
            instance._scaler_by_column = {
                col: MinMaxScaler().fit(np.array(instance._input_df[col]).reshape(-1, 1)) for col in instance._extract_column
            }
        return instance

    def __repr__(self):
        return (f"TimeSeriesDataProcessor("
                f"extract_column={self._extract_column}, "
                f"training_data_ratio={self._training_data_ratio}, "
                f"training_window_size={self._training_window_size}, "
                f"target_window_size={self._target_window_size})")

    def __str__(self):
        return self.__repr__()

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
        splitting the extracted data into training and testing set by training data ratio.
        the training data ratio to be 1 is acceptable. return empty testing set.
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

        # check the training window size and target window size is valid.
        if len(self._training_array) - self._target_window_size < self._training_window_size:
            raise ValueError("Training window size is too large.")

        # return object from sliding window mask is a list of numpy array.
        x_train, y_train = self._sliding_window_mask(self._training_array)
        x_test, y_test = self._sliding_window_mask(self._testing_array)

        # convert the list of numpy array to multidimensional numpy array.
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

        if y_test.shape[0] != 0:
            y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
        else:
            # if the testing set is empty, set the testing set to None. by pass to reshape the testing set.
            pass

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
        Sliding window mask on provided data, return training data and target data
        if provided input_array is empty, return list with zero size.
        """
        x = []
        y = []

        for i in range(self._training_window_size, len(input_array) - self._target_window_size + 1):
            x.append(input_array[i - self._training_window_size:i])
            y.append(input_array[i:i + self._target_window_size, 0])

        if len(x) != len(y):
            raise ValueError("The length of training data and target data is not the same.")

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
