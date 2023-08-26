import mlflow.pytorch


class MLFlowModelRegistry:

    @staticmethod
    def register_model(pytorch_model, model_name):
        """
        The method to register the model with MLFlow
        :param pytorch_model:
        :param model_name:
        :return:
        """
        # TODO: make MLFlow model folder configurable
        print(f"start Registering the model {model_name} to MLFlow server")

        artifact_path = "pytorch-model"
        run_id = mlflow.active_run().info.run_id
        model_uri = "{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

        mlflow.pytorch.log_model(
            pytorch_model=pytorch_model,
            artifact_path=model_uri,
            registered_model_name=model_name
        )
        # mlflow.pytorch.autolog()

        #
        # model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"Finish Registering the model {model_name} to MLFlow server")
        return

