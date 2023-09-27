import pandas as pd
import numpy as np

from src.ml_core.data_processor.data_processor import DataProcessorFactory
import src.store.data_processor_store


def test_data_processor_store_add():

    # create a dataframe for testing with three columns and 100 rows
    # the columns are "A, B, C"
    # the value's range from 1 to 100
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

    # create a data processor
    data_processor = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df,
        extract_column=['A', 'B'],
        training_data_ratio=0.9
    )

    # add the data processor to the data processor store
    data_processor_store = src.store.data_processor_store.get_store()
    data_processor_store.add_data_processor(
        data_processor_id="test_data_processor_a",
        data_processor=data_processor
    )

    # get the data processor from the data processor store
    data_processor_from_store = data_processor_store.get_data_processor(
        data_processor_id="test_data_processor_a"
    )

    # check if the data processor is the same as the one we created
    assert data_processor_from_store == data_processor
    data_processor_from_store.preprocess_data()

    processed_data_x = data_processor_from_store.get_training_data_x()
    processed_data_y = data_processor_from_store.get_training_target_y()

    assert processed_data_x.shape == (30, 60, 2)
    assert processed_data_y.shape == (30, 1)


def test_data_processor_store_remove():

    # create a dataframe for testing with three columns and 100 rows
    # the columns are "A, B, C"
    # the value's range from 1 to 100
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

    # create a data processor
    data_processor = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df,
        extract_column=['A', 'B'],
        training_data_ratio=0.9
    )

    # add the data processor to the data processor store
    data_processor_store = src.store.data_processor_store.get_store()
    data_processor_store.add_data_processor(
        data_processor_id="test_data_processor_b",
        data_processor=data_processor
    )

    # get the data processor from the data processor store
    data_processor_from_store = data_processor_store.get_data_processor(
        data_processor_id="test_data_processor_b"
    )

    # check if the data processor is the same as the one we created
    assert data_processor_from_store == data_processor
    data_processor_from_store.preprocess_data()

    processed_data_x = data_processor_from_store.get_training_data_x()
    processed_data_y = data_processor_from_store.get_training_target_y()

    assert processed_data_x.shape == (30, 60, 2)
    assert processed_data_y.shape == (30, 1)

    # remove the data processor from the data processor store
    data_processor_store.remove_data_processor(
        data_processor_id="test_data_processor_b"
    )

    # get the data processor from the data processor store
    data_processor_from_store = data_processor_store.get_data_processor(
        data_processor_id="test_data_processor_b"
    )

    # check if the data processor is the same as the one we created
    assert data_processor_from_store is None


def test_data_processor_store_add_same_id_twice():

        # create a dataframe for testing with three columns and 100 rows
        # the columns are "A, B, C"
        # the value's range from 1 to 100
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list('ABC'))

        # create a data processor
        data_processor_1 = DataProcessorFactory.create_data_processor(
            data_processor_type="time_series",
            input_data=df,
            extract_column=['A', 'B'],
            training_data_ratio=0.9
        )

        data_processor_2 = DataProcessorFactory.create_data_processor(
            data_processor_type="time_series",
            input_data=df,
            extract_column=['A', 'B'],
            training_data_ratio=0.9
        )

        # add the data processor to the data processor store
        data_processor_store = src.store.data_processor_store.get_store()
        assert data_processor_store.add_data_processor(
            data_processor_id="test_data_processor_c",
            data_processor=data_processor_1
        ) == True

        assert data_processor_store.add_data_processor(
            data_processor_id="test_data_processor_c",
            data_processor=data_processor_2
        ) == False


def test_data_processor_store_remove_not_exist_id():

    data_processor_store = src.store.data_processor_store.get_store()
    assert data_processor_store.remove_data_processor(
        data_processor_id="test_data_processor_d"
    ) == False
