import threading
from typing import Optional
from src.ml_core.data_processor.base_data_processor import BaseDataProcessor
from src.ml_core.data_processor.time_series_data_processor import TimeSeriesDataProcessor

class DataProcessStore:
    """
    The data process store is a central store across the ML system, it is designed in singleton pattern
    to store the data processor object which created, and operated during the ML system process
    """
    _app = None
    _app_lock = threading.Lock()
    _data_processor_store = {}

    def __new__(cls, *args, **kwargs):
        """
        Define singleton pattern to get the app instance, provide thread safety
        :param args:
        :param kwargs:
        :return:
        """
        with cls._app_lock:
            if cls._app is None:
                cls._app = super(DataProcessStore, cls).__new__(cls)
            return cls._app

    @classmethod
    def _is_data_process_exist(cls, data_processor_id: str) -> bool:
        """
        Check if the data processor object exists in the data processor store
        :param data_processor_id: provided by register
        :return: True if exists, False if not exist
        """
        return data_processor_id in cls._data_processor_store.keys()

    @classmethod
    def add_data_processor(cls, data_processor_id: str, data_processor: BaseDataProcessor) -> bool:
        """
        Add the data processor object to the data processor store
        :param data_processor_id: provided by register
        :param data_processor: the data processor object
        :return:
        """
        if cls._is_data_process_exist(data_processor_id):
            return False
        cls._data_processor_store[data_processor_id] = data_processor.to_dict()
        return True

    @classmethod
    def get_data_processor(cls, data_processor_id: str) -> Optional[BaseDataProcessor]:
        """
        Get the data processor object from the data processor store
        :param data_processor_id: provided by register
        :return: the data processor object or None if not found
        """
        data_processor_dict = cls._data_processor_store.get(data_processor_id, None)
        if data_processor_dict:
            processor_type = data_processor_dict.get('type')
            if processor_type == 'TimeSeriesDataProcessor':
                return TimeSeriesDataProcessor.from_dict(data_processor_dict)
        return None

    @classmethod
    def remove_data_processor(cls, data_processor_id: str) -> bool:
        """
        Remove the data processor object from the data processor store
        :param data_processor_id: provided by register
        :return: True if remove success, False if not
        """
        if cls._is_data_process_exist(data_processor_id):
            del cls._data_processor_store[data_processor_id]
            return True
        return False

    @classmethod
    def update_data_processor(cls, data_processor_id: str, new_data_processor: BaseDataProcessor) -> bool:
        if cls._is_data_process_exist(data_processor_id):
            cls._data_processor_store[data_processor_id] = new_data_processor.to_dict()
            return True
        return False

    @classmethod
    def list_data_processors(cls) -> list:
        """
        List all data processor IDs in the data processor store
        :return: A list of data processor IDs
        """
        return list(cls._data_processor_store.keys())

def get_store():
    """
    Get the data process manager object in singleton pattern
    :return: the data process manager object
    """
    return DataProcessStore()
