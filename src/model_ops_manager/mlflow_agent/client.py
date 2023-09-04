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
    """
    The class responsible for interact with mlflow server to get the model information
    Various Composition of Model feature is accessible through this class.
    implement several universal methods of model check logic and model uri composition logic
    """

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

    @classmethod
    def get_model_latest_version(cls, model_name: str) -> int:
        """
        Simply provide the model name to get the latest version of the model
        :param model_name:
        :return:
        """
        if not cls.is_model_name_exist(model_name):
            raise ValueError(f"The model name: {model_name} is not registered")

        model_version_details = cls.mlflow_client.get_latest_versions(
            name=model_name,
        )
        # get the model version from the model_version_details
        model_version = model_version_details[0].version

        return model_version

    @classmethod
    def compose_model_uri(cls, model_name: str, model_version=None, model_stage=None) -> str:
        """
        providing the arbitrary model name, model version and model stage composition.
        To get the specific model uri to access the model artifact by model_name and exact version number
        :param model_name: the model name should be a string
        :param model_version: the model version should be an integer
        :param model_stage: the model stage should be in category of ["None", "Staging", "Production", "Archived"]
        :return: models:/{model_name}/{model_version}
        """
        model_version = cls.get_target_model_version(model_name, model_version, model_stage)
        model_uri = f"models:/{model_name}/{model_version}"
        return model_uri


class MLFlowClientModelLoader(MLFlowClientModelAgent):

    @classmethod
    def get_download_model_uri(cls, model_name: str, model_version=None, model_stage=None) -> str:
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

        model_version = cls.get_target_model_version(model_name, model_version, model_stage)

        download_model_uri = cls.mlflow_client.get_model_version_download_uri(
            name=model_name,
            version=model_version
        )
        return download_model_uri

    @classmethod
    def load_model_as_pyfunc(cls, *args, **kwargs) -> mlflow.pyfunc:
        """
        load the model with provided address to reach the model artifact.
        here can provide the model name together with model version or model stage to get the model,
        or provide the model artifact server uri directly to load the model
        usage 1: load_model(model_name: str, model_version: int, model_stage: str)
        usage 2: load_model(model_artifact_uri: str)
        hint, the model_stage should in the category of ["None", "Staging", "Production", "Archived"]
        the model artifact server uri should be in the format of
        "http://<artifact_server_ip>:<port>/api/2.0/mlflow-artifacts/artifacts/experiments/.../artifacts/<...>-model"
        :param args:
        :param kwargs:
        :return:
        """

        # only two case, args or kwargs provided
        # if args provided, several criteria should be checked
        # args should be in length of 1 to 3
        # args[0] must be string, it can be model_name or model_artifact_uri
        # if model_artifact_uri provided, the model_name, model_version and model_stage should be ignored
        # if model_name provided, the model_version or model_stage should be provided
        # once the args[0] is model_name, the args[1] should be model_version or model_stage
        # distinguish the model_name first
        # then distinguish the model_version or model_stage by following rules
        # model_version is integer and model_stage is string in category of ["None", "Staging", "Production", "Archived"]
        # Once all args is distinguished, the model uri can be get by calling the method of compose_model_uri
        if args:
            if len(args) > 3:
                raise ValueError("The args should be in length of 1 to 3")
            if not isinstance(args[0], str):
                raise TypeError("The first arg should be model_name or model_artifact_uri")
            if len(args) == 1:
                # only model_artifact_uri provided
                model_artifact_uri = args[0]
                model = mlflow.pyfunc.load_model(model_artifact_uri)
                return model
            if len(args) == 2:
                # model_name and model_version or model_stage provided
                model_name = args[0]
                if isinstance(args[1], int):
                    model_version = args[1]
                    model_uri = cls.compose_model_uri(model_name, model_version)
                    model = mlflow.pyfunc.load_model(model_uri)
                    return model
                if isinstance(args[1], str):
                    model_stage = args[1]
                    model_uri = cls.compose_model_uri(model_name, model_stage=model_stage)
                    model = mlflow.pyfunc.load_model(model_uri)
                    return model
            if len(args) == 3:
                # model_name, model_version and model_stage provided
                model_name, arg2, arg3 = args
                # Determine which argument is the model_version and which is the model_stage
                if isinstance(arg2, int):
                    model_version, model_stage = arg2, arg3
                else:
                    model_stage, model_version = arg2, arg3
                model_uri = cls.compose_model_uri(model_name, model_version, model_stage)
                model = mlflow.pyfunc.load_model(model_uri)
                return model



    @classmethod
    def get_all_version_registered_model(cls, model_name: str):
        """
        provide the model name only and get all the version of the registered model
        :param model_name:
        :return:
        """
        if not cls.is_model_name_exist(model_name):
            raise ValueError(f"The model name: {model_name} is not registered")

        cls.mlflow_client.get_registered_model()




if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5011")

    # client = MLFlowClient()

    # MLFlowClientModelLoader.init_mlflow_client()
    # MLFlowClientModelLoader.get_download_model_uri(model_name="Pytorch_Model", model_stage="Production")

    # model_uri = "models:/Pytorch_Model/Production"
    # model = mlflow.pytorch.load_model(model_uri)
    # breakpoint()
