import os
import mlflow

from src.model_ops_manager.mlflow_agent.mlflow_agent import MLFlowAgent
# from src.model_ops_manager.mlflow_agent.model_downloader import MLFlowClientModelLoader


class MLFlowModelDownloadServingApp:
    """
    A singleton class to provide mlflow interface to download model
    provide a global mlflow client resource and share with ml-system components
    """

    # singleton instance of MLFlowClientModelLoader
    # _mlflow_model_downloader = MLFlowClientModelLoader

    _mlflow_agent = MLFlowAgent()

    def __init__(self):
        """
        All the field needed for ml process is put in class level
        """
        pass


    @classmethod
    def set_mlflow_tracking_uri(cls, tracking_uri: str) -> None:
        """
        Set the mlflow tracking uri
        :param tracking_uri: the mlflow tracking uri
        :return:
        """
        # mlflow.set_tracking_uri(tracking_uri)
        print("Setting mlflow tracking via mlflow_agent")
        cls._mlflow_agent.set_tracking_uri(tracking_uri)

    @classmethod
    def init_mlflow_downloader_client(cls) -> None:
        """
        Initialize the mlflow client
        :return:
        """
        try:
            cls._mlflow_agent.init_mlflow_client()
        except ValueError:
            print("mlflow tracking uri is not set, please set the tracking uri first by api ")

    @classmethod
    def download_mlflow_pyfunc_model(cls, model_name: str, model_version: int = None, model_stage: str = None):
        """
        Download the mlflow pyfunc model
        :param model_name: the name of the model
        :param model_version: the version of the model
        :param model_stage: the stage of the model
        :return:
        """
        model_uri = cls._mlflow_agent.get_model_download_source_uri(model_name, model_version, model_stage)
        model = cls._mlflow_agent.load_pyfunc_model(model_uri)
        return model

    @classmethod
    def download_mlflow_original_model(cls, model_name: str, model_version: int = None, model_stage: str = None):
        """
        Download the mlflow original model
        :param model_name: the name of the model
        :param model_version: the version of the model
        :param model_stage: the stage of the model
        :return:
        """
        model_uri = cls._mlflow_agent.get_model_download_source_uri(model_name, model_version, model_stage)
        model = cls._mlflow_agent.load_original_model(model_uri)
        return model

    @classmethod
    def check_mlflow_tracking_uri(cls) -> str:
        """
        Check if the mlflow tracking uri is set or not
        :return: True if mlflow tracking uri is set
        """
        mlflow_tracking_uri = mlflow.get_tracking_uri()
        if mlflow_tracking_uri is None:
            raise ValueError("mlflow tracking uri is not set, please set the tracking uri first by api ")
        return mlflow_tracking_uri


def get_app():
    """
    Get the singleton instance of MLFlowModelDownloadServingApp
    :return: the singleton instance of MLFlowModelDownloadServingApp
    """
    return MLFlowModelDownloadServingApp()
