
import mlflow

from src.model_ops_manager.mlflow_agent.client import MLFlowClientModelAgent


class MLFlowClientModelLoader(MLFlowClientModelAgent):


    @classmethod
    def _parsing_adhoc_input_to_model_uri(cls, *args, **kwargs) -> str:
        """
        this method is used to parse the input to model uri
        the adhoc input of model information includes.
        Provided "model_name" together optional "model_version" and/or "model_stage"
        or Provided "model_artifact_uri"
        that can be provided by args or kwargs
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
                if "http" in args[0] or "s3" in args[0]:
                    model_uri = args[0]
                    return model_uri
                else:
                    model_uri = cls.get_download_model_uri(args[0])
                    return model_uri
            if len(args) == 2:
                # model_name and model_version or model_stage provided
                model_name = args[0]
                if isinstance(args[1], int):
                    model_version = args[1]
                    model_uri = cls.compose_model_uri(model_name, model_version)
                    return model_uri
                if isinstance(args[1], str):
                    model_stage = args[1]
                    model_uri = cls.compose_model_uri(model_name, model_stage=model_stage)
                    return model_uri
            if len(args) == 3:
                # model_name, model_version and model_stage provided
                model_name, arg2, arg3 = args
                # Determine which argument is the model_version and which is the model_stage
                if isinstance(arg2, int):
                    model_version, model_stage = arg2, arg3
                else:
                    model_stage, model_version = arg2, arg3
                model_uri = cls.compose_model_uri(model_name, model_version, model_stage)
                return model_uri

    @classmethod
    def load_pyfunc_model(cls, *args, **kwargs) -> mlflow.pyfunc:
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

        model_uri = cls._parsing_adhoc_input_to_model_uri(*args, **kwargs)
        model = mlflow.pyfunc.load_model(model_uri)
        return model


    @classmethod
    def load_original_model(cls, *args, **kwargs):
        """
        load the model with provided address to reach the model artifact.
        get the mlflow pyfunc model at first and extract the original model flavor from it.
        Using mlflow provided method to load the original model.
        :return:
        """

        loaded_pyfunc_model = cls.load_pyfunc_model(*args, **kwargs)
        original_flavor_loader_module = (loaded_pyfunc_model.load_model(args[0])._model_meta.flavors["python_function"]["loader_module"])
        print(type(original_flavor_loader_module))



if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5011")
    model_downloader = MLFlowClientModelLoader
    model_downloader.init_mlflow_client()
    model_uri = model_downloader.get_download_model_uri("Pytorch_Model", model_stage="Staging")
    print(model_uri)

    model = model_downloader.load_pyfunc_model("Pytorch_Model")
    print(model)

    # model = model_downloader.load_original_model(model_uri)


