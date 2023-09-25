import pytest
import pandas as pd
import numpy as np

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

