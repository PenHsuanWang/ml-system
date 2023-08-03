from abc import ABC, abstractmethod

import pandas as pd


class BaseDataSinker:
    """
    Abstract base class for all sinkers.
    To sink data to the designated location.
    """

    def __init__(self):
        pass

    @abstractmethod
    def sink_pandas_dataframe(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Sink pandas dataframe to the designated location.
        The location to store the data can be various, the specific destination store api will
        be implemented in the concrete class. The method will return True if the data sunk
        :param data: pandas dataframe
        :param destination: the location to store the data
        :return: True if the data sunk successfully, otherwise False
        """
        raise NotImplementedError


