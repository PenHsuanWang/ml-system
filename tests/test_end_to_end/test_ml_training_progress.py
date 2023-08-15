from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.ml_core.data_processor.time_series_data_processor import TimeSeriesDataProcessor


def test_complete_ml_training_process():
    TRAINING_WINDOW_SIZE = 60
    TARGET_WINDOW_SIZE = 1

    # Load data from yahoo finance using data fetcher
    fetcher = DataFetcherFactory.create_data_fetcher("yfinance")
    fetcher.fetch_from_source(stock_id="AAPL", start_date="2022-01-01", end_date="2023-01-01")
    apple_raw_df = fetcher.get_as_dataframe()

    # Preprocess data
    data_processor = TimeSeriesDataProcessor(
        input_data=apple_raw_df,
        extract_column=['Close', 'Volume'],
        training_data_ratio=0.6,
        training_window_size=TRAINING_WINDOW_SIZE,
        target_window_size=TARGET_WINDOW_SIZE
    )

    data_processor.preprocess_data()
    train_tensor = data_processor.get_training_tensor()
    test_tensor = data_processor.get_testing_tensor()

    print(train_tensor.shape)
    print(test_tensor.shape)

