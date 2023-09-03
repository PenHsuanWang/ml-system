"""
This module is defined for the client of mlflow_agent
"""
from abc import ABC, abstractmethod

import mlflow

# import src.model_ops_manager.mlflow_agent.utils as mlflow_utils

# here need to define a global mlflow_client to make this resource shared by all the methods
# before the mlflow client initialized have to make sure the tracking uri is set
# let mlflow_client to be a class level resource


class MLFlowClient(ABC):

    mlflow_client = None

    @classmethod
    def init_mlflow_client(cls):
        if not cls.mlflow_client:
            cls.mlflow_client = mlflow.tracking.MlflowClient()

    def __init__(self):
        pass


class MLFlowClientModelAgent(MLFlowClient):

    @classmethod
    def is_model_name_exist(cls, model_name: str) -> bool:
        """
        check the model name is valid
        :param model_name:
        :return:
        """
        if not isinstance(model_name, str):
            raise TypeError("The model name should be string")

        # model name exist in the registry
        model_registry = cls.mlflow_client.get_registered_model(
            name=model_name
        )
        if not model_registry:
            raise ValueError(f"The model name: {model_name} is not registered")

        # to implement more validation logic if needed

        return True

    @classmethod
    def get_target_model_version(cls, model_name: str, model_version=None, model_stage=None) -> int:
        """
        Provide the model name, model version or model stage to get the model version
        return the corresponding model version to access model artifact.
        The model_name is required, and provide model version or model stage to get the latest version with corresponding stage,
        if the model version is provided, the model_stage will be ignored.
        if the model version is not provided, the model stage is required.
        :param model_name:
        :param model_version:
        :param model_stage:
        :return:
        """

        if not cls.is_model_name_exist(model_name):
            raise ValueError(f"The model name: {model_name} is not registered")

        if model_version:
            return model_version

        if model_stage:
            # to check the model stage is in category
            category = ["None", "Staging", "Production", "Archived"]
            if model_stage in category:
                pass
            else:
                raise ValueError(
                    f"The model stage should be in {category}, but the provided model stage is {model_stage}")

            model_version_details = cls.mlflow_client.get_latest_versions(
                name=model_name,
                stages=[model_stage]
            )
            # get the model version from the model_version_details
            model_version = model_version_details[0].version

        return model_version


class MLFlowClientModelLoader(MLFlowClientModelAgent):

    @classmethod
    def get_download_model_uri(cls, model_name: str, model_version=None, model_stage=None):
        """
        to get the model uri for downloading the model with provided model name, model version and model stage
        provide model_name is required, and provide model_stage to get the latest version with corresponding stage,
        if the model_version is provided, the model_stage will be ignored.
        The model stage is only used when the model version is not provided. and the latest version with corresponding
        stage will be fetched
        :param model_name:
        :param model_version:
        :param model_stage:
        :return:
        """

        try:
            cls.is_model_name_exist(model_name)
        except Exception as e:
            print(f"provide model name is not valid, please check the model name: {model_name}")

        if not model_version and not model_stage:
            # warn the user of the usage of the method
            print("Please provide the model version or model stage")
            return None

        if model_version and model_stage:
            # warning the user that the model stage will be ignored
            print(f"The model stage will be ignored, since the model version: {model_version} is provided")

        # Make sure the needed parameters are provided

        if not model_version:
            # get the latest model version with corresponding stage
            model_version_details = cls.mlflow_client.get_latest_versions(
                name=model_name,
                stages=[model_stage]
            )
            # get the model version from the model_version_details
            model_version = model_version_details[0].version

        download_model_uri = cls.mlflow_client.get_model_version_download_uri(
            name=model_name,
            version=model_version
        )

        print(download_model_uri)

        return download_model_uri

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5001")

    # client = MLFlowClient()

    MLFlowClientModelLoader.init_mlflow_client()

    MLFlowClientModelLoader.get_download_model_uri(model_name="Pytorch_Model", model_stage="Production")

    # model_uri = "models:/Pytorch_Model/Production"
    # model = mlflow.pytorch.load_model(model_uri)
    # breakpoint()
