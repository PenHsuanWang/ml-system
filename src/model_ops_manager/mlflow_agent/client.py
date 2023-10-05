"""
This module is defined for the client of mlflow_agent
"""
from abc import ABC, abstractmethod

import mlflow
import mlflow.entities.model_registry as mlflow_model_registry
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.exceptions import MlflowException
from src.model_ops_manager.mlflow_agent.singleton_meta import SingletonMeta

# import src.model_ops_manager.mlflow_agent.utils as mlflow_utils

# here need to define a global mlflow_client to make this resource shared by all the methods
# before the mlflow client initialized have to make sure the tracking uri is set
# let mlflow_client to be a class level resource


class MLFlowClient:
    """
    The base class of mlflow client to interact with mlflow server
    Singleton design of client to make all the concrete class share the same mlflow client
    for specific application or use case need particular implement of mlflow client can inherit this class
    """

    mlflow_client = None

    @classmethod
    def init_mlflow_client(cls):
        """
        initialize the mlflow client, make sure the tracking uri is set before initialize the client
        :return:
        """
        if not mlflow.tracking.get_tracking_uri():
            raise ValueError("The tracking uri is not set, please set the tracking uri first")
        if not cls.mlflow_client:
            cls.mlflow_client = mlflow.tracking.MlflowClient()
            print(f"mlflow client initialized, the tracking uri is {mlflow.tracking.get_tracking_uri()}")


class MLFlowClientModelAgent(MLFlowClient):
    """
    The class responsible for interact with mlflow server to get the model information
    Various Composition of Model feature is accessible through this class.
    implement several universal methods of model check logic and model uri composition logic
    """

    # ======================================================================
    # Simple unit function to check the specific model is registered or not
    # ======================================================================

    @classmethod
    def is_model_version_registered(cls, model_name: str, model_version: int) -> bool:
        """
        A simple method to check the model version is registered or not
        provide model_name and model_version to check the model version is registered
        :param model_name:
        :param model_version:
        :return:
        """
        try:
            desired_model_version = cls.mlflow_client.get_model_version(model_name, model_version)
            if isinstance(desired_model_version, mlflow_model_registry.ModelVersion):
                return True
            else:
                return False
        except MlflowException as mle:
            print(
                f"fetching the model {model_name} with version: {model_version} from mlflow server failed, the error is {mle}")
            print(f"current client tracking uri is {mlflow.tracking.get_tracking_uri()}, please check the tracking uri")
            return False
        except Exception as e:
            print("Unexpected error:", e)
            return False

    # ======================================================================
    # Simple unit function to fetch model information
    # ======================================================================

    @classmethod
    def get_mlflow_registered_model(cls, model_name: str) -> RegisteredModel:
        """
        check the model name is valid
        :param model_name:
        :return:
        """
        if not isinstance(model_name, str):
            raise TypeError("The model name should be string")

        try:
            model_registry = cls.mlflow_client.get_registered_model(
                name=model_name
            )
            return model_registry
        except MlflowException as mle:
            print(f"fetching the model {model_name} from mlflow server failed, the error is {mle}")
            print(f"current client tracking uri is {mlflow.tracking.get_tracking_uri()}, please check the tracking uri")
            raise ValueError(f"The model name: {model_name} is not registered")

    @classmethod
    def get_model_latest_version(cls, model_name: str, model_stage=None) -> int:
        """
        Provide the model name, and optional desired stage(s) to get the model latest version for provided stage(s)
        return the corresponding model version to access model artifact.
        The model_name is required, and provide model version or model stage to get the latest version with corresponding stage,
        if the model version is provided, the model_stage will be ignored.
        if the model version is not provided, the model stage is required.
        :param model_name: the model name should be a string
        :param model_stage: desired stage(s) to get the model latest version for provided stage(s), should be in category of ["None", "Staging", "Production", "Archived"]
        :return:
        """

        if model_stage:
            # to check the model stage is in category
            category = ["None", "Staging", "Production", "Archived"]
            if model_stage in category:
                pass
            else:
                raise ValueError(
                    f"The model stage should be in {category}, but the provided model stage is {model_stage}")
            try:
                model_version = cls.mlflow_client.get_latest_versions(
                    name=model_name,
                    stages=[model_stage]
                )[0].version
                return model_version
            except MlflowException as mle:
                print(
                    f"fetching the model {model_name} with stage: {model_stage} from mlflow server failed, the error is {mle}")
                print(
                    f"current client tracking uri is {mlflow.tracking.get_tracking_uri()}, please check the tracking uri")
                raise ValueError(f"The model name: {model_name} with stage: {model_stage} is not registered")

        else:
            try:
                model_version = cls.mlflow_client.get_latest_versions(
                    name=model_name,
                )[0].version
                return model_version
            except MlflowException as mle:
                print(f"fetching the model {model_name} from mlflow server failed, the error is {mle}")
                print(
                    f"current client tracking uri is {mlflow.tracking.get_tracking_uri()}, please check the tracking uri")
                raise ValueError(f"The model name: {model_name} is not registered")

    # ===============================================================================
    # function to composite the referenced resource to fetch model from mlflow server
    # ===============================================================================

    @classmethod
    def compose_model_uri(cls, model_name: str, model_version=None, model_stage=None) -> str:
        """
        providing the arbitrary model name, model version and model stage composition.
        To get the specific model uri to access the model artifact by model_name and exact version number
        model_name is necessary, and provide optionally model_version or model_stage to compose model uri,
        if only provide model_name, the latest model version will be fetched
        if provide model_version together, will check model with provided version is registered
        if provide model_stage together, will get the latest version with corresponding stage.
        if provide both model_version and model_stage, model_version will be used. Cross-check the desired model stage match with the model version or provide warning message
        :param model_name: the model name should be a string
        :param model_version: the model version should be an integer
        :param model_stage: the model stage should be in category of ["None", "Staging", "Production", "Archived"]
        :return: models:/{model_name}/{model_version}
        """

        if isinstance(model_name, str):
            pass
        else:
            raise TypeError("The model name should be string")

        if isinstance(model_version, int):
            if cls.is_model_version_registered(model_name, model_version):
                pass
            else:
                print(f"The model name: {model_name} with version: {model_version} is not registered")
                raise ValueError(f"The model name: {model_name} with version: {model_version} is not registered")

        elif model_version is None and isinstance(model_stage, str):
            model_version = cls.get_model_latest_version(model_name, model_stage)

        else:
            model_version = cls.get_model_latest_version(model_name)

        return f"models:/{model_name}/{model_version}"

    @classmethod
    def get_model_download_source_uri(cls, model_name: str, model_version=None, model_stage=None) -> str:
        """
        to get the model uri for downloading the model with provided model name, model version and model stage
        provide model_name is required, and provide model_stage to get the latest version with corresponding stage,
        if the model_version is provided, the model_stage will be ignored.
        The model stage is only used when the model version is not provided. and the latest version with corresponding
        stage will be fetched
        :param model_name: the model name should be a string
        :param model_version: the model version should be an integer
        :param model_stage: the model stage should be in category of ["None", "Staging", "Production", "Archived"]
        :return: the artifact server uri for downloading the model artifact
        """

        if model_stage:
            stage_latest_version = cls.get_model_latest_version(model_name, model_stage)
            if model_version:
                if int(model_version) == int(stage_latest_version):
                    pass
                else:
                    print(
                        f"The model version: {model_version} is not match with the latest version with corresponding stage: {stage_latest_version}")
                    raise ValueError(
                        f"The model version: {model_version} is not match with the latest version with corresponding stage: {stage_latest_version}")
            else:
                print(f"The model version is not provided, the latest version with corresponding stage: {model_stage} is {stage_latest_version}, will be fetched")
                model_version = stage_latest_version

        if model_version is None:
            model_version = cls.get_model_latest_version(model_name)

        download_model_uri = cls.mlflow_client.get_model_version_download_uri(
            name=model_name,
            version=model_version
        )
        return download_model_uri



if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5011")

    mlflow_client = MLFlowClientModelAgent()
    mlflow_client.init_mlflow_client()

    model_uri = mlflow_client.compose_model_uri("Pytorch_Model", 1)
    print(model_uri)

    print(mlflow_client.get_model_latest_version("Pytorch_Model", "Production"))
