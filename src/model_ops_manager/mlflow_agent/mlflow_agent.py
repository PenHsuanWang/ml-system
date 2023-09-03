from src.model_ops_manager.mlflow_agent.singleton_meta import SingletonMeta
from src.model_ops_manager.mlflow_agent.tracking import MLFlowTracking
from src.model_ops_manager.mlflow_agent.configuration import MLFlowConfiguration
from src.model_ops_manager.mlflow_agent.registration import MLFlowModelRegistry


class NullMLFlowAgent(MLFlowTracking, MLFlowConfiguration, MLFlowModelRegistry, metaclass=SingletonMeta):
    """
    NullMLFlowAgent class here is a null object class for MLFlowAgent, Used by Trainer if the MLFlow is not enabled
    Implement the same methods as MLFlowAgent, but do nothing.
    The method should be synchronized with MLFlowAgent if any new added or deleted
    """
    @staticmethod
    def start_run(experiment_name, run_name, *args, **kwargs):
        print("NullMLFlowAgent: start_run")
        pass

    @staticmethod
    def log_params_many(*args, **kwargs):
        print("NullMLFlowAgent: log_params_many")
        pass

    @staticmethod
    def log_param(*args, **kwargs):
        print("NullMLFlowAgent: log_param")
        pass

    @staticmethod
    def log_metrics_many(*args, **kwargs):
        print("NullMLFlowAgent: log_metrics_many")
        pass

    @staticmethod
    def log_metric(*args, **kwargs):
        print("NullMLFlowAgent: log_metric")
        pass

    @staticmethod
    def end_run():
        print("NullMLFlowAgent: end_run")
        pass

    @staticmethod
    def register_model(*args, **kwargs):
        print("NullMLFlowAgent: register_model")
        pass


class MLFlowAgent(MLFlowTracking, MLFlowConfiguration, MLFlowModelRegistry, metaclass=SingletonMeta):
    """
    MLFlowAgent class here integrates all the MLFlow functionalities into a single class
    To check the exact sub-class methods, please refer to the corresponding sub-class
    MLFlowConfiguration: setting MLFlow tracking uri
    MLFlowTracking: logging model metrics and parameters
    MLFlowModelRegistry: registering the model
    """
    pass


