import pytest
import pandas as pd
from src.data_io.data_fetcher.yfinance_data_adaptor import YahooFinanceAdaptor

def test__extract_fetch_stock_and_time_range_params():
    adaptor = YahooFinanceAdaptor()

    # Test missing parameters
    with pytest.raises(ValueError, match="Missing input parameters"):
        adaptor._extract_fetch_stock_and_time_range_params()

    # Test invalid stock id
    with pytest.raises(ValueError, match="Invalid stock id: INVALID_ID"):
        adaptor._extract_fetch_stock_and_time_range_params(stock_id="INVALID_ID", start_date="2022-01-01", end_date="2023-01-01")

    # Test invalid date format
    with pytest.raises(ValueError, match="Invalid date format"):
        adaptor._extract_fetch_stock_and_time_range_params(stock_id="AAPL", start_date="01-01-2022", end_date="2023-01-01")

    # Test invalid date range
    with pytest.raises(ValueError, match="Invalid date range"):
        adaptor._extract_fetch_stock_and_time_range_params(stock_id="AAPL", start_date="2023-01-01", end_date="2022-01-01")

def test_fetch_from_source():
    adaptor = YahooFinanceAdaptor()

    # Test valid stock id and date range
    adaptor.fetch_from_source(stock_id="AAPL", start_date="2022-01-01", end_date="2023-01-01")
    assert isinstance(adaptor._fetched_data, pd.DataFrame)

    # # Test fetching again without calling get_as_dataframe
    # TODO: Add this test after warning logging implemented
    # with pytest.warns(UserWarning, match="Warning: self._fetched_data is not empty"):
    #     adaptor.fetch_from_source(stock_id="AAPL", start_date="2022-01-01", end_date="2023-01-01")

def test_get_as_dataframe():
    adaptor = YahooFinanceAdaptor()

    # Test get data before fetch
    with pytest.raises(ValueError, match="Invalid fetched data format"):
        adaptor.get_as_dataframe()

    # Test get data after fetch
    adaptor.fetch_from_source(stock_id="AAPL", start_date="2022-01-01", end_date="2023-01-01")
    assert isinstance(adaptor.get_as_dataframe(), pd.DataFrame)
