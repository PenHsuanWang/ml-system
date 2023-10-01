
import threading

from mlflow.exceptions import MlflowException

from src.model_ops_manager.mlflow_agent.mlflow_agent import MLFlowAgent
from src.ml_core.inferencer.inferencer import InferencerFactory


class MlInferenceServingApp:

    """
    the ML inference serving app is a singleton class to serve the model inference process
    storing the model in serving list, for expose model inference api to client

    provide the method to client to
    1. add model from mlflow model artifact
    2. remove model from mlflow model artifact
    3. check current model in serving
    4. get model inference result
    """

    _app = None
    _app_lock = threading.Lock()

    _mlflow_agent = MLFlowAgent()

    _model_in_serving = {}

    def __new__(cls, *args, **kwargs):
        with cls._app_lock:
            if cls._app is None:
                cls._app = super(MlInferenceServingApp, cls).__new__(cls)
                cls._app._initialized = False
            return cls._app

    def __init__(self):
        pass

    @classmethod
    def _is_model_exist(cls, model_name: str):
        """
        check if model is already in serving
        :param model_name: model name
        :return: True if model is already in serving
        """
        if model_name in cls._model_in_serving.keys():
            return True
        else:
            return False

    @classmethod
    def setup_model_inference_serving_app(cls, *args, **kwargs):
        """
        set up the model inference serving app
        :return:
        """

        mlflow_tracking_server = kwargs.get("mlflow_tracking_server", None)

        if mlflow_tracking_server is None:
            raise ValueError("mlflow tracking server is not provided")

        cls._mlflow_agent.set_tracking_uri(mlflow_tracking_server)
        cls._mlflow_agent.init_mlflow_client()

    @classmethod
    def list_all_model_in_serving(cls):
        return cls._model_in_serving.keys()

    @classmethod
    def load_original_model_from_mlflow_artifact(cls, model_name: str, model_version=None, model_stage=None):
        """
        load original model from mlflow artifact
        implementation in mlflow_agent, distinguish the origin model flavor by mlflow model registry
        :param model_name: model name
        :param model_version: model version
        :return: True if model is successfully loaded
        """

        # check mlflow agent setup tracking server uri ready or not
        if cls._mlflow_agent.mlflow_connect_tracking_server_uri is None:
            raise ValueError("mlflow tracking server is not provided")

        load_model_args = [model_name]
        if model_version is not None:
            load_model_args.append(model_version)
        if model_stage is not None:
            load_model_args.append(model_stage)

        if not cls._is_model_exist(model_name):
            try:
                fetched_model = cls._mlflow_agent.load_original_model(*load_model_args)
                cls._model_in_serving[model_name] = fetched_model
                print(f"model {model_name} is successfully saved in serving list")
                print(f"model serving list{cls._model_in_serving.keys()}")
            except MlflowException as me:
                print(f"load model {model_name} failed, error message: {me}")
                return False

            except RuntimeError as re:
                print(f"Unexpected run time error happen, exception stack: {re}")
                return False

            return True
        else:
            print(f"model {model_name} is already in serving")
            return False

    @classmethod
    def load_pyfunc_model_from_mlflow_artifact(cls, model_name: str, model_version: str, model_stage: str = "Production"):
        """
        load pyfunc model from mlflow artifact
        implementation in mlflow_agent, distinguish the origin model flavor by mlflow model registry
        :param model_name:
        :param model_version:
        :param model_stage:
        :return:
        """

        # check mlflow agent setup tracking server uri ready or not
        if cls._mlflow_agent.mlflow_connect_tracking_server_uri is None:
            raise ValueError("mlflow tracking server is not provided")

        if not cls._is_model_exist(model_name):
            try:
                fetched_model = cls._mlflow_agent.load_pyfunc_model(model_name, model_version, model_stage)
                cls._model_in_serving[model_name] = fetched_model
            except MlflowException as me:
                print(f"load model {model_name} failed, error message: {me}")
                return False

            except RuntimeError as re:
                print(f"Unexpected run time error happen, exception stack: {re}")
                return False

            return True
        else:
            print(f"model {model_name} is already in serving")
            return False

    @classmethod
    def set_mode(cls, model_name: str, model: object):
        """
        To set the in-memory model from current python process to model serving
        add model into serving list with model name
        :param model_name:
        :param model:
        :return:
        """
        if not cls._is_model_exist(model_name):
            cls._model_in_serving[model_name] = model
            return True
        else:
            print(f"model {model_name} is not in serving")
            return False

    @classmethod
    def remove_model_from_serving(cls, model_name: str):
        """
        remove model from serving
        :param model_name: model name
        :return: True if model is successfully removed
        """
        if cls._is_model_exist:
            cls._model_in_serving.pop(model_name)
            return True
        else:
            print(f"model {model_name} is not in serving")
            return False

    @classmethod
    def inference(cls, model_name, data_input, device="cpu"):
        """
        get the model inference result
        :param model_name: model name
        :param data_input: data input
        :return: model inference result
        """
        if cls._is_model_exist(model_name):

            # create inferencer
            model_inferencer = InferencerFactory.create_inferencer(model_flavor="pytorch", model=cls._model_in_serving[model_name])
            return model_inferencer.predict(data_input, device=device)
        else:
            print(cls._model_in_serving.keys())
            print(f"model {model_name} is not in serving")
            return None


def get_app():
    app = MlInferenceServingApp()
    print(f"Initializing the MlInferenceServingApp from singleton {app}")
    return app



