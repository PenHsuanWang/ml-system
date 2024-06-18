import threading

class ModelStore:
    """
    The model store is a central store across the ML system,
    designed in a singleton pattern to store model objects.
    """
    _app = None
    _app_lock = threading.Lock()
    _model_store = {}

    def __new__(cls, *args, **kwargs):
        with cls._app_lock:
            if cls._app is None:
                cls._app = super(ModelStore, cls).__new__(cls)
            return cls._app

    @classmethod
    def _is_model_exist(cls, model_id: str) -> bool:
        return model_id in cls._model_store.keys()

    @classmethod
    def add_model(cls, model_id: str, model: object) -> bool:
        if cls._is_model_exist(model_id):
            return False
        cls._model_store[model_id] = model
        return True

    @classmethod
    def get_model(cls, model_id: str) -> object:
        return cls._model_store.get(model_id, None)

    @classmethod
    def remove_model(cls, model_id: str) -> bool:
        if cls._is_model_exist(model_id):
            del cls._model_store[model_id]
            return True
        return False

    @classmethod
    def update_model(cls, model_id: str, new_model: object) -> bool:
        if cls._is_model_exist(model_id):
            cls._model_store[model_id] = new_model
            return True
        return False

def get_model_store():
    return ModelStore()
