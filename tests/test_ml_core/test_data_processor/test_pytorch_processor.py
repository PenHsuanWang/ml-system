import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.ml_core.data_processor.data_processor import DataProcessorFactory


def test_create_time_series_data_processor():
    """
    test create data processor
    :return:
    """

    # create a simple pandas dataframe with three columns and 100 rows
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

    data_processor = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df,
        extract_column=['A', 'B'],
        training_data_ratio=0.9,
        training_window_size=60,
        target_window_size=1
    )
    assert data_processor is not None

    data_processor.preprocess_data()

    training_dataset = data_processor.get_training_data_x()
    assert training_dataset is not None
    assert type(training_dataset) == np.ndarray

    testing_dataset = data_processor.get_training_target_y()
    assert testing_dataset is not None
    assert type(testing_dataset) == np.ndarray


def test_time_series_data_processor_with_training_data_ratio_be_one():
    """
    test create data processor
    :return:
    """

    # create a simple pandas dataframe with three columns and 100 rows
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

    data_processor = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df,
        extract_column=['A', 'B'],
        training_data_ratio=1.0,
        training_window_size=60,
        target_window_size=1
    )
    assert data_processor is not None

    data_processor.preprocess_data()

    training_dataset = data_processor.get_training_data_x()
    assert training_dataset is not None
    assert type(training_dataset) == np.ndarray

    testing_dataset = data_processor.get_training_target_y()
    assert testing_dataset is not None
    assert type(testing_dataset) == np.ndarray


def test_time_series_data_window_size_large_then_data_length():
    """
    test create data processor
    :return:
    """

    # create a simple pandas dataframe with three columns and 100 rows
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

    data_processor = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df,
        extract_column=['A', 'B'],
        training_data_ratio=1.0,
        training_window_size=101,
        target_window_size=1
    )
    assert data_processor is not None

    try:
        data_processor.preprocess_data()
    except ValueError as ve:
        print(ve)
        assert True

def test_time_series_data_processor_scaling():
    """
    Test the TimeSeriesDataProcessor internal scaling function.
    Provided a time series data with 100 rows with value between 0 and 100.
    :return:
    """

    # create a simple pandas dataframe with three columns and 100 rows
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

    # prepare original A, B ,and C column data
    original_a_column_data = df['A'].values
    original_b_column_data = df['B'].values
    original_c_column_data = df['C'].values

    # Scaling this data using MinMaxScaler
    scaler_a = MinMaxScaler()
    scaler_b = MinMaxScaler()
    scaler_c = MinMaxScaler()
    scaler_a.fit(original_a_column_data.reshape(-1, 1))
    scaler_b.fit(original_b_column_data.reshape(-1, 1))
    scaler_c.fit(original_c_column_data.reshape(-1, 1))
    scaled_a_column_data = scaler_a.transform(original_a_column_data.reshape(-1, 1))
    scaled_b_column_data = scaler_b.transform(original_b_column_data.reshape(-1, 1))
    scaled_c_column_data = scaler_c.transform(original_c_column_data.reshape(-1, 1))

    data_processor = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df,
        extract_column=['A', 'B', 'C'],
        training_data_ratio=0.9,
        training_window_size=60,
        target_window_size=1
    )
    assert data_processor is not None

    data_processor.preprocess_data()

    # get the scaler object
    data_x = data_processor.get_training_data_x()
    data_y = data_processor.get_training_target_y()

    # check the first batch of 60 rows data for column A
    assert data_x[0, :, 0] == pytest.approx(scaled_a_column_data[0:60, 0], 0.0001)
    assert data_x[0, :, 1] == pytest.approx(scaled_b_column_data[0:60, 0], 0.0001)
    assert data_x[0, :, 2] == pytest.approx(scaled_c_column_data[0:60, 0], 0.0001)

    assert data_y[0] == pytest.approx(scaled_a_column_data[60:61, 0], 0.0001)

def test_time_series_data_processor_scaler_inverse():
    """
    Test the TimeSeriesDataProcessor internal scaling function.
    Provided a time series data with 100 rows with value between 0 and 100.
    :return:
    """

    # create a simple pandas dataframe with three columns and 100 rows
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

    # prepare original A, B ,and C column data
    original_a_column_data = df['A'].values
    original_b_column_data = df['B'].values
    original_c_column_data = df['C'].values

    # Scaling this data using MinMaxScaler
    scaler_a = MinMaxScaler()
    scaler_b = MinMaxScaler()
    scaler_c = MinMaxScaler()
    scaler_a.fit(original_a_column_data.reshape(-1, 1))
    scaler_b.fit(original_b_column_data.reshape(-1, 1))
    scaler_c.fit(original_c_column_data.reshape(-1, 1))
    scaled_a_column_data = scaler_a.transform(original_a_column_data.reshape(-1, 1))
    scaled_b_column_data = scaler_b.transform(original_b_column_data.reshape(-1, 1))
    scaled_c_column_data = scaler_c.transform(original_c_column_data.reshape(-1, 1))

    data_processor = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df,
        extract_column=['A', 'B', 'C'],
        training_data_ratio=1,
        training_window_size=99,
        target_window_size=1
    )
    data_processor.preprocess_data()
    # data_x = data_processor.get_training_data_x()
    # target_y = data_processor.get_training_target_y()

    scaled_a_column_data_inverse = data_processor.inverse_testing_scaler(scaled_a_column_data, 'A')
    scaled_b_column_data_inverse = data_processor.inverse_testing_scaler(scaled_b_column_data, 'B')
    scaled_c_column_data_inverse = data_processor.inverse_testing_scaler(scaled_c_column_data, 'C')

    # inverse scaled data should be the same as original data
    assert original_a_column_data.reshape(-1, 1) == pytest.approx(scaled_a_column_data_inverse, 0.0001)
    assert original_b_column_data.reshape(-1, 1) == pytest.approx(scaled_b_column_data_inverse, 0.0001)
    assert original_c_column_data.reshape(-1, 1) == pytest.approx(scaled_c_column_data_inverse, 0.0001)
