import pandas as pd
import pytest
from unittest.mock import patch
from src.data_io.data_fetcher.local_file_data_adaptor import LocalFileAdaptor


@patch("src.data_io.data_fetcher.local_file_data_adaptor.LocalFileAdaptor._check_fetched_data_is_valid")
@patch("pandas.read_csv")
@patch("pandas.read_excel")
@patch("pandas.read_json")
@patch("os.path.exists")
def test_fetch_from_source(mock_exists, mock_read_json, mock_read_excel, mock_read_csv, mock_check_data):
    mock_exists.return_value = True
    mock_df = pd.DataFrame({"A": [1, 2, 3]})
    mock_read_csv.return_value = mock_df
    mock_read_json.return_value = mock_df
    mock_read_excel.return_value = mock_df
    mock_check_data.return_value = True

    adaptor = LocalFileAdaptor()

    # Test .csv file
    adaptor.fetch_from_source(file_path="mock_path.csv")
    assert adaptor._fetched_data.equals(mock_df)
    mock_read_csv.assert_called_once_with("mock_path.csv")
    fetched_df = adaptor.get_as_dataframe()  # Store the result
    assert isinstance(fetched_df, pd.DataFrame)
    assert fetched_df.equals(mock_df)  # Use the stored result
    assert adaptor._fetched_data.empty

    mock_read_csv.reset_mock()

    # Test .json file
    adaptor.fetch_from_source(file_path="mock_path.json")
    assert adaptor._fetched_data.equals(mock_df)
    mock_read_json.assert_called_once_with("mock_path.json")
    fetched_df = adaptor.get_as_dataframe()  # Store the result
    assert isinstance(fetched_df, pd.DataFrame)
    assert fetched_df.equals(mock_df)  # Use the stored result
    assert adaptor._fetched_data.empty

    mock_read_json.reset_mock()

    # Test .xlsx file
    adaptor.fetch_from_source(file_path="mock_path.xlsx")
    assert adaptor._fetched_data.equals(mock_df)
    mock_read_excel.assert_called_once_with("mock_path.xlsx")
    fetched_df = adaptor.get_as_dataframe()  # Store the result
    assert isinstance(fetched_df, pd.DataFrame)
    assert fetched_df.equals(mock_df)  # Use the stored result
    assert adaptor._fetched_data.empty

    mock_read_excel.reset_mock()

    # # Test file not exist
    # mock_exists.return_value = False
    # with pytest.raises(FileNotFoundError, match="File not found"):
    #     adaptor.fetch_from_source(file_path="not_exist_path.csv")
    #     assert adaptor._fetched_data.equals(mock_df)
    #
    # mock_exists.reset_mock()
    #
    # # Test unsupported file format
    # with pytest.raises(ValueError, match="Unsupported file format"):
    #     adaptor.fetch_from_source(file_path="unsupported_format.unknown")
    #     assert adaptor._fetched_data.equals(mock_df)

@patch("src.data_io.data_fetcher.local_file_data_adaptor.LocalFileAdaptor._check_fetched_data_is_valid")
@patch("pandas.read_csv")
@patch("pandas.read_excel")
@patch("pandas.read_json")
@patch("os.path.exists")
def test_file_not_exist(mock_exists, mock_read_json, mock_read_excel, mock_read_csv, mock_check_data):
    mock_exists.return_value = False # To simulate file not exist

    adaptor = LocalFileAdaptor()

    # The FileNotFoundError should be raised
    with pytest.raises(FileNotFoundError, match="File not found"):
        adaptor.fetch_from_source(file_path="not_exist_path.csv")


@patch("src.data_io.data_fetcher.local_file_data_adaptor.LocalFileAdaptor._check_fetched_data_is_valid")
@patch("pandas.read_csv")
@patch("pandas.read_excel")
@patch("pandas.read_json")
@patch("os.path.exists")
def test_unsupported_file_format(mock_exists, mock_read_json, mock_read_excel, mock_read_csv, mock_check_data):
    mock_exists.return_value = True # To simulate file exist

    adaptor = LocalFileAdaptor()

    # The ValueError should be raised
    with pytest.raises(ValueError, match="Unsupported file format"):
        adaptor.fetch_from_source(file_path="unsupported_format.unknown")


@pytest.mark.parametrize("file_path", [
    "valid_path.csv",
    "valid_path.json",
    "valid_path.xlsx"
])
def test_get_as_dataframe(file_path):
    mock_df = pd.DataFrame({"A": [1, 2, 3]})

    adaptor = LocalFileAdaptor()
    adaptor._fetched_data = mock_df
    result = adaptor.get_as_dataframe()

    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_df)
    assert adaptor._fetched_data.equals(pd.DataFrame())
