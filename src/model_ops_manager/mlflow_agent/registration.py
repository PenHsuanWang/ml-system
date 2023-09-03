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
        run = mlflow.active_run()

        if run is not None:
            run_id = run.info.run_id

            mlflow.pytorch.log_model(
                pytorch_model=pytorch_model,
                artifact_path=artifact_path,
                registered_model_name=model_name  # this should register the model
            )


            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"Finish Registering the model {model_name} to MLFlow server")
        return

