# singleton instance
import threading


class DataIOServingApp:

    # define singleton pattern to get the app instance, provide thread safe
    _app = None
    _app_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._app_lock:
            if cls._app is None:
                cls._app = super(DataIOServingApp, cls).__new__(cls)
                cls._app._initialized = False  # Set initialized to False
            return cls._app

    def __init__(self):
        if not self._initialized:
            self._data_fetcher = {}
            self._data_sinker = {}
            self._initialized = True

    @property
    def data_fetcher(self):
        return self._data_fetcher

    @property
    def data_sinker(self):
        return self._data_sinker


# Define get_app dependency
def get_app():
    app = DataIOServingApp()
    print(f"Initializing the DataIOServingApp from singleton {app}")
    return app



