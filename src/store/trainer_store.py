import threading

class TrainerStore:
    """
    The trainer store is a central store across the ML system,
    designed in a singleton pattern to store trainer objects.
    """
    _app = None
    _app_lock = threading.Lock()
    _trainer_store = {}

    def __new__(cls, *args, **kwargs):
        with cls._app_lock:
            if cls._app is None:
                cls._app = super(TrainerStore, cls).__new__(cls)
            return cls._app

    @classmethod
    def _is_trainer_exist(cls, trainer_id: str) -> bool:
        return trainer_id in cls._trainer_store.keys()

    @classmethod
    def add_trainer(cls, trainer_id: str, trainer: object) -> bool:
        if cls._is_trainer_exist(trainer_id):
            return False
        cls._trainer_store[trainer_id] = trainer
        return True

    @classmethod
    def get_trainer(cls, trainer_id: str) -> object:
        return cls._trainer_store.get(trainer_id, None)

    @classmethod
    def remove_trainer(cls, trainer_id: str) -> bool:
        if cls._is_trainer_exist(trainer_id):
            del cls._trainer_store[trainer_id]
            return True
        return False

    @classmethod
    def update_trainer(cls, trainer_id: str, new_trainer: object) -> bool:
        if cls._is_trainer_exist(trainer_id):
            cls._trainer_store[trainer_id] = new_trainer
            return True
        return False

def get_trainer_store():
    return TrainerStore()
