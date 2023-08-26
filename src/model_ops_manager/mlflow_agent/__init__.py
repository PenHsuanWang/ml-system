from src.model_ops_manager.mlflow_agent.tracking import MLFlowTracking
from src.model_ops_manager.mlflow_agent.configuration import MLFlowConfiguration
from src.model_ops_manager.mlflow_agent.registration import MLFlowModelRegistry


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MLFlowAgent(MLFlowTracking, MLFlowConfiguration, MLFlowModelRegistry, metaclass=SingletonMeta):
    pass
