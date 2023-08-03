from abc import ABC, abstractmethod

import pandas as pd


class SinkPandasDataFrame(ABC):
    """
    The Abstraction class for sinking pandas dataframe to the designated location.
    The support file format is csv, txt, json, xlsx
    """

    def __init__(self):
        pass

    @staticmethod
    def _check_data_is_valid_or_raise_error(data: pd.DataFrame) -> None:
        """
        Check the data is valid.
        :param data: pandas dataframe
        :return:
        """
        # check the data type is valid pandas dataframe and not empty
        if isinstance(data, pd.DataFrame) and data.shape != (0, 0):
            pass
        else:
            raise ValueError("Invalid data type")

    @abstractmethod
    def sink(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Sink pandas dataframe to the designated location.
        The location to store the data can be various, the specific destination store api will
        be implemented in the concrete class. The method will return True if the data sunk
        :param data: pandas dataframe
        :param destination: the location to store the data
        :return: True if the data sunk successfully, otherwise False
        """
        raise NotImplementedError


class SinkPandasDataFrameToCSV(SinkPandasDataFrame):
    """ The Concrete class for sinking pandas dataframe to the designated csv file."""

    def __init__(self):
        super().__init__()

    def sink(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Sink pandas dataframe to the designated location.
        The location to store the data can be various, the specific destination store api will
        be implemented in the concrete class. The method will return True if the data sunk
        :param data: pandas dataframe
        :param destination: the location to store the data
        :return: True if the data sunk successfully, otherwise False
        """
        # check the data type is valid pandas dataframe and not empty
        self._check_data_is_valid_or_raise_error(data)

        try:
            data.to_csv(destination, index=False)
            return True
        except FileNotFoundError:
            print("File path not found.")
        except PermissionError:
            print("Permission denied.")
        except IOError:
            print("An I/O error occurred.")
        except ValueError:
            print("A value error occurred, possibly due to an invalid parameter.")
        except UnicodeEncodeError:
            print("A Unicode encoding error occurred.")
        except MemoryError:
            print("A memory error occurred, possibly due to insufficient system memory.")
        except AttributeError:
            print("An attribute error occurred, possibly due to calling 'to_csv' on a non-DataFrame object.")
        except Exception as e:
            print("An unknown error occurred.")
            print(e)

        return False


class SinkPandasDataFrameToTxt(SinkPandasDataFrame):
    """The Concrete class for sinking pandas dataframe to the designated txt file."""

    def __init__(self):
        super().__init__()

    def sink(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Sink pandas dataframe to the designated location.
        The location to store the data can be various, the specific destination store api will
        be implemented in the concrete class. The method will return True if the data sunk
        :param data: pandas dataframe
        :param destination: the location to store the data
        :return: True if the data sunk successfully, otherwise False
        """
        # check the data type is valid pandas dataframe and not empty
        self._check_data_is_valid_or_raise_error(data)

        try:
            data.to_csv(destination, index=False)
            return True
        except FileNotFoundError:
            print("File path not found.")
        except PermissionError:
            print("Permission denied.")
        except IOError:
            print("An I/O error occurred.")
        except ValueError:
            print("A value error occurred, possibly due to an invalid parameter.")
        except UnicodeEncodeError:
            print("A Unicode encoding error occurred.")
        except MemoryError:
            print("A memory error occurred, possibly due to insufficient system memory.")
        except AttributeError:
            print("An attribute error occurred, possibly due to calling 'to_csv' on a non-DataFrame object.")
        except Exception as e:
            print("An unknown error occurred.")
            print(e)

        return False


class SinkPandasDataFrameToJSON(SinkPandasDataFrame):
    """ The Concrete class for sinking pandas dataframe to the designated json file."""

    def __init__(self):
        super().__init__()

    def sink(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Sink pandas dataframe to the designated location.
        The location to store the data can be various, the specific destination store api will
        be implemented in the concrete class. The method will return True if the data sunk
        :param data: pandas dataframe
        :param destination: the location to store the data
        :return: True if the data sunk successfully, otherwise False
        """
        # check the data type is valid pandas dataframe and not empty
        self._check_data_is_valid_or_raise_error(data)

        try:
            data.to_json(destination, index=False)
            return True
        except FileNotFoundError:
            print("File path not found.")
        except PermissionError:
            print("Permission denied.")
        except IOError:
            print("An I/O error occurred.")
        except ValueError:
            print("A value error occurred, possibly due to an invalid parameter.")
        except UnicodeEncodeError:
            print("A Unicode encoding error occurred.")
        except MemoryError:
            print("A memory error occurred, possibly due to insufficient system memory.")
        except AttributeError:
            print("An attribute error occurred, possibly due to calling 'to_csv' on a non-DataFrame object.")
        except Exception as e:
            print("An unknown error occurred.")
            print(e)

        return False


class SinkPandasDataFrameToExcel(SinkPandasDataFrame):
    """ The Concrete class for sinking pandas dataframe to the designated excel file."""

    def __init__(self):
        super().__init__()

    def sink(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Sink pandas dataframe to the designated location.
        The location to store the data can be various, the specific destination store api will
        be implemented in the concrete class. The method will return True if the data sunk
        :param data: pandas dataframe
        :param destination: the location to store the data
        :return: True if the data sunk successfully, otherwise False
        """
        # check the data type is valid pandas dataframe and not empty
        self._check_data_is_valid_or_raise_error(data)

        try:
            data.to_excel(destination, index=False)
            return True
        except FileNotFoundError:
            print("File path not found.")
        except PermissionError:
            print("Permission denied.")
        except IOError:
            print("An I/O error occurred.")
        except ValueError:
            print("A value error occurred, possibly due to an invalid parameter.")
        except UnicodeEncodeError:
            print("A Unicode encoding error occurred.")
        except MemoryError:
            print("A memory error occurred, possibly due to insufficient system memory.")
        except AttributeError:
            print("An attribute error occurred, possibly due to calling 'to_csv' on a non-DataFrame object.")
        except Exception as e:
            print("An unknown error occurred.")
            print(e)

        return False


class SinkPandasDataFrameFactory:
    @staticmethod
    def create_sink_adaptor(sink_type: str) -> SinkPandasDataFrame:
        """
        Factory method to create the sink adaptor based on the sink type
        :param sink_type:
        :return:
        """
        if sink_type == "csv":
            return SinkPandasDataFrameToCSV()
        elif sink_type == "txt":
            return SinkPandasDataFrameToTxt()
        elif sink_type == "json":
            return SinkPandasDataFrameToJSON()
        elif sink_type == "excel":
            return SinkPandasDataFrameToExcel()
        else:
            raise KeyError(f"Invalid sink type: {sink_type}")
