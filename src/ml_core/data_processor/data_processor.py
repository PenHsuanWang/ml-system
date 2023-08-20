from src.ml_core.data_processor.time_series_data_processor import TimeSeriesDataProcessor


class DataProcessorFactory:

    @staticmethod
    def create_data_processor(data_processor_type: str, **kwargs):
        """
        create a data processor instance
        :param data_processor_type: data processor type
        :return: data processor instance
        """
        if data_processor_type == "time_series":
            return TimeSeriesDataProcessor(**kwargs)
        else:
            raise Exception("Data processor type not supported")

