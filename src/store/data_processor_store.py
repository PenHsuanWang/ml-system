
import threading


class DataProcessStore:
    """
    the data process store is a central store across the ml system, it is designed in singleton pattern
    to store the data processor object which created, and operated during the ml system process
    """
    _app = None
    _app_lock = threading.Lock()

    _data_processor_store = {}

    def __new__(cls, *args, **kwargs):
        """
        define singleton pattern to get the app instance, provide thread safe
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
        check if the data processor object is exist in the data processor store
        :param data_processor_id: provided by register
        :return: True if exist, False if not exist
        """
        return data_processor_id in cls._data_processor_store.keys()

    @classmethod
    def add_data_processor(cls, data_processor_id: str, data_processor: object) -> bool:
        """
        add the data processor object to the data processor store
        :param data_processor_id: provided by register
        :param data_processor: the data processor object
        :return:
        """

        if cls._is_data_process_exist(data_processor_id):
            return False

        cls._data_processor_store[data_processor_id] = data_processor
        return True

    @classmethod
    def get_data_processor(cls, data_processor_id: str) -> object:
        """
        get the data processor object from the data processor store
        :param data_processor_id: provided by register
        :return: the data processor object
        """

        try:
            return cls._data_processor_store[data_processor_id]
        except KeyError:
            # Handle the case when the ID is not found
            return None

    @classmethod
    def remove_data_processor(cls, data_processor_id: str) -> bool:
        """
        remove the data processor object from the data processor store
        :param data_processor_id: provided by register
        :return: True if remove success, False if not
        """

        if cls._is_data_process_exist(data_processor_id):
            del cls._data_processor_store[data_processor_id]
            return True
        else:
            return False


def get_app():
    """
    get the data process manager object in singleton pattern
    :return: the data process manager object
    """
    return DataProcessStore()

