
from src.data_io.data_sinker.local_file_data_sinker import LocalFileDataSinker

class DataSinkerFactory:
    """
    Implement a factory to create data sinker instance
    """

    @staticmethod
    def create_data_sinker(data_sink_type: str):
        """
        create a data sinker instance
        provide the data sink type and data sink dir
        :param data_sink_type:
        :param data_sink_dir:
        :return:
        """
        if data_sink_type == "local_file":
            return LocalFileDataSinker()
        else:
            raise Exception("Data sink type not supported")



