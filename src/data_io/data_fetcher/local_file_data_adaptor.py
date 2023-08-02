import pandas as pd
from os import path

from src.data_io.data_fetcher.base_data_fetcher import BaseDataFetcher


class LocalFileAdaptor(BaseDataFetcher):
    """
    Adaptor class for Reading data from local file.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _extract_fetch_file_path_params(*args, **kwargs) -> str:
        """
        Check the input parameters are valid.
        check the necessary input parameters are provided.
        :param args:
        :param kwargs:
        :return:
        """
        # check the input parameters are valid or raise error
        if not all(key in kwargs for key in ("file_path",)):
            raise ValueError("Missing input parameters")

        file_path = kwargs["file_path"]

        # check the file_path is valid and exist or not
        if not path.exists(file_path):
            raise FileNotFoundError("File not found")

        return file_path

    def fetch_from_source(self, *args, **kwargs) -> None:
        """
        Fetch data from local file.
        Store the fetched data in self._fetched_data temperately.
        The fetched data is in pandas dataframe format.
        :return:
        """

        file_path = self._extract_fetch_file_path_params(*args, **kwargs)

        # define the file reader
        _file_readers = {
            ".csv": self._read_csv,
            ".txt": self._read_txt,
            ".json": self._read_json,
            ".xlsx": self._read_excel,
        }

        file_extension = path.splitext(file_path)[1]
        if file_extension in _file_readers:

            if self._fetched_data.shape != (0, 0):
                print("Warning: self._fetched_data is not empty, "
                      "please `get_as_dataframe` to get the temporary data first, ")
            else:
                self._fetched_data = _file_readers[file_extension](file_path)
        else:
            raise ValueError("Unsupported file format")

    def get_as_dataframe(self, *args, **kwargs) -> pd.DataFrame:
        """
        return the fetched data in self._fetched_data
        provided desired data format.
        :param args:
        :param kwargs:
        :return:
        """
        # check the type of self._fetched_data is pandas dataframe or not
        if not self._check_fetched_data_is_valid():
            # raise error if fetched data is not valid
            # let user to handling the error from outer fetcher
            raise ValueError("Invalid fetched data format")

        # tuple unpacking to flush the data
        self._fetched_data, flush_data = pd.DataFrame(), self._fetched_data

        return flush_data

    @staticmethod
    def _read_csv(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def _read_txt(file_path):
        return pd.read_csv(file_path, sep="\t")

    @staticmethod
    def _read_json(file_path):
        return pd.read_json(file_path)

    @staticmethod
    def _read_excel(file_path):
        return pd.read_excel(file_path)