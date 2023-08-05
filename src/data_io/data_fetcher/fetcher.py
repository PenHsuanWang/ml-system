
from src.data_io.data_fetcher.local_file_fetch_adaptor import LocalFileAdaptor
from src.data_io.data_fetcher.yfinance_fetch_adaptor import YahooFinanceAdaptor


class DataFetcherFactory:
    """
    Implement a factory to create data fetcher instance
    """

    @staticmethod
    def create_data_fetcher(data_source_type: str):
        """
        create a data fetcher instance
        provide the data source type and data source dir
        :param data_source_type:
        :param data_source_dir:
        :return:
        """
        if data_source_type == "local_file":
            return LocalFileAdaptor()
        elif data_source_type == "yfinance":
            return YahooFinanceAdaptor()
        else:
            raise Exception("Data source type not supported")