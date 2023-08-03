import os
from typing import Callable

import pandas as pd

from src.data_io.data_sinker.base_data_sinker import BaseDataSinker
from src.data_io.data_sinker.sink_pandas_dataframe import SinkPandasDataFrameFactory


class LocalFileDataSinker(BaseDataSinker):
    """
    LocalFileDataSinker class for sinking data to local file system.
    The support file format is csv, txt, json, xlsx
    """
    def __init__(self):
        super().__init__()

    def sink_pandas_dataframe(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Using internal function `_create_sink_file_adaptor` to parse the destination file path to get the adaptor.
        check the data type is valid pandas dataframe and not empty.
        :param data: pandas dataframe to sink
        :param destination: the location to store the data
        :return: True if the data sunk successfully, otherwise False
        """
        # check the data type is valid pandas dataframe and not empty
        if not isinstance(data, pd.DataFrame) or data.shape == (0, 0):
            raise ValueError("Invalid data type")

        # create the file sinker
        sink_adaptor = self._create_sink_adaptor(destination)

        # sink the data
        sink_adaptor.sink(data, destination)

        return True

    @staticmethod
    def _create_sink_adaptor(destination: str):
        """
        Parsing the destination file path to get the file extension.
        check the path format is valid and exist.
        check the file extension is supported or not.
        check the file exist or create the file.
        based on the extension to create the file sinker.

        :param destination:
        :return:
        """
        # check the destination is existed or not
        if not os.path.exists(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))

        # based on file extension to create the file sinker
        file_extension = os.path.splitext(destination)[1]

        try:
            sink_adaptor = SinkPandasDataFrameFactory.create_sink_adaptor(file_extension)
            return sink_adaptor
        except KeyError:
            raise KeyError("Unsupported file extension")

