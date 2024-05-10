import threading
from fastapi import HTTPException
from src.model_ops_manager.mlflow_agent.client import MLFlowClientModelAgent


class MLFlowModelsService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MLFlowModelsService, cls).__new__(cls)
                cls._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.client = MLFlowClientModelAgent
            self.client.init_mlflow_client()
            self._initialized = True

    def list_models(self):
        try:
            return self.client.list_all_registered_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

def get_mlflow_models_service():
    return MLFlowModelsService()
