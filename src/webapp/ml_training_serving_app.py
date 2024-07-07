import os
import mlflow
import torch
from torch.utils.data.dataloader import DataLoader
import pandas as pd

import src.webapp.data_io_serving_app
import src.store.data_processor_store
import src.store.trainer_store
import src.store.model_store
from src.ml_core.data_processor.data_processor import DataProcessorFactory
from src.ml_core.data_loader.base_dataset import TimeSeriesDataset
from src.ml_core.models.torch_nn_models.model import TorchNeuralNetworkModelFactory
from src.ml_core.trainer.trainer import TrainerFactory
from src.model_ops_manager.mlflow_agent.mlflow_agent import MLFlowAgent, NullMLFlowAgent


class MLTrainingServingApp:
    """
    A singleton class to serve the model training process
    provided the operation interface to client via REST api
    1. operating the data fetcher by data_io_serving_app, another exported REST api to user to init the data fetcher.
       provide the request body with information about fetch_data parameter
    """

    _data_io_serving_app = src.webapp.data_io_serving_app.get_app()
    _data_processor_store = src.store.data_processor_store.get_store()
    _trainer_store = src.store.trainer_store.get_trainer_store()
    _model_store = src.store.model_store.get_model_store()

    _data_fetcher = None
    _data_processor = None
    _trainer = None

    _raw_pandas_dataframe = None
    _training_tensor = None
    _training_target_tensor = None
    _testing_tensor = None
    _testing_target_tensor = None
    _model = None

    def __init__(self):
        """
        All the field needed for ml process is put in class level
        """
        pass

    @classmethod
    def set_data_fetcher(cls, data_fetcher_name: str) -> bool:
        """
        Design for an exposed REST api to let client set the data fetcher
        The data fetcher is initialized by data_io_serving_app, it should be ready in the process.
        set the existing data fetcher to self._data_fetcher
        :param data_fetcher_name: data fetcher name
        :return: True if data fetcher is successfully initialized
        """
        cls._data_io_serving_app = src.webapp.data_io_serving_app.get_app()
        try:
            print(cls._data_io_serving_app.data_fetcher)
            cls._data_fetcher = cls._data_io_serving_app.data_fetcher[data_fetcher_name]
        except KeyError:
            print(f"Data fetcher: {data_fetcher_name} not found")
            return False
        return True

    @classmethod
    def fetcher_data(cls, args: list, kwargs: dict) -> bool:
        """
        Design for an exposed REST api to let client fetch data from source
        The data fetcher is initialized by data_io_serving_app, it should be ready in the process.
        :param data_fetcher_name: data fetcher name
        :param args: args for data fetcher
        :param kwargs: kwargs for data fetcher
        :return: True if data fetcher is successfully initialized
        """
        try:
            cls._data_fetcher.fetch_from_source(
                stock_id=kwargs["stock_id"],
                start_date=kwargs["start_date"],
                end_date=kwargs["end_date"],
            )
        except RuntimeError:
            print("Failed to fetch data from source")
            return False
        return True

    @classmethod
    def init_data_processor_from_df(cls, data_processor_id: str, data_processor_type: str, dataframe_json: dict,
                                    **kwargs) -> bool:
        """
        Initialize the data processor from a JSON-encoded DataFrame.
        :param data_processor_id: The ID of the data processor to initialize
        :param data_processor_type: The type of data processor to initialize
        :param dataframe_json: JSON-encoded DataFrame
        :param kwargs: Additional parameters for the data processor
        :return: True if data processor is successfully initialized
        """
        try:
            dataframe = pd.DataFrame.from_records(dataframe_json["data"], columns=dataframe_json["columns"])
            cls._raw_pandas_dataframe = dataframe
            print(f"DataFrame initialized: {dataframe.head()}")
            print(f"Received kwargs: {kwargs}")
            cls._data_processor = DataProcessorFactory.create_data_processor(
                data_processor_type,
                input_data=cls._raw_pandas_dataframe,
                **kwargs
            )
            print(f"Created data processor: {cls._data_processor}")
            cls._data_processor_store.add_data_processor(
                data_processor_id=data_processor_id,
                data_processor=cls._data_processor
            )
            print(f"Data processor stored with ID: {data_processor_id} and details: {cls._data_processor}")
        except Exception as e:
            print("Failed to init data processor from DataFrame")
            print(e)
            return False

        return True

    @classmethod
    def init_data_processor(cls, data_processor_id: str, data_processor_type: str, dataframe: pd.DataFrame = None,
                            **kwargs) -> bool:
        """
        Design for an exposed REST api to let client init the data preprocessor
        To initialize the data preprocessor
        Initialize parameters from kwargs provided by client via REST api request body.
        If a DataFrame is provided, it will be used instead of fetching data from the data fetcher.
        :param data_processor_id: The ID of the data processor to initialize
        :param data_processor_type: The type of data processor to initialize
        :param dataframe: Optional pandas DataFrame containing the data to process
        :param kwargs: Additional parameters for the data processor
        :return: True if data processor is successfully initialized
        """
        if dataframe is not None:
            # Use the provided DataFrame
            cls._raw_pandas_dataframe = dataframe
        else:
            # Check if the data fetcher is initialized
            if cls._data_fetcher is None:
                print("Data fetcher is not initialized")
                return False
            if not cls._raw_pandas_dataframe:
                try:
                    cls._raw_pandas_dataframe = cls._data_fetcher.get_as_dataframe()
                except ValueError as ve:
                    print("Failed to get data from data fetcher, try fetcher again")
                    cls._raw_pandas_dataframe = cls._data_fetcher.fetch_from_source()
                    cls._raw_pandas_dataframe = cls._data_fetcher.get_as_dataframe()

        try:
            cls._data_processor = DataProcessorFactory.create_data_processor(
                data_processor_type,
                input_data=cls._raw_pandas_dataframe,
                **kwargs
            )
            # Register the data processor to data processor manager
            cls._data_processor_store.add_data_processor(
                data_processor_id=data_processor_id,
                data_processor=cls._data_processor
            )
        except Exception as e:
            print("Failed to init data processor")
            print(e)
            return False

        return True

    @classmethod
    def init_model(cls, model_type: str, model_id: str, **kwargs) -> bool:
        """
        Design for an exposed REST api to let client init the model
        To initialize the model
        Initialize parameters from kwargs provided by client via REST api request body.
        :param model_type: The type of model to initialize
        :param model_id: The ID to associate with the model in the store
        :param kwargs: Additional parameters for the model
        :return: True if model is successfully initialized
        """
        try:
            cls._model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
                model_type,
                **kwargs
            )
            cls._model_store.add_model(model_id, cls._model)
        except Exception as e:
            print("Failed to init model")
            return False
        return True

    @classmethod
    def init_trainer(cls, trainer_type: str, trainer_id: str, **kwargs) -> bool:
        """
        Design for an exposed REST api to let client init the trainer
        To initialize the trainer
        Initialize parameters from kwargs provided by client via REST api request body.
        The trainer provides the function to add mlflow agent to log the training process and register the model
        create mlflow agent and setting the tracking uri here.
        :param trainer_type: The type of trainer to initialize
        :param trainer_id: The ID to associate with the trainer in the store
        :param kwargs: Additional parameters for the trainer
        :return: True if trainer is successfully initialized
        """
        if cls._model is None:
            print("Model is not initialized")
            return False

        criterion = None
        if kwargs["loss_function"] == "mse":
            criterion = torch.nn.MSELoss()

        optimizer = None
        if kwargs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(cls._model.parameters(), lr=float(kwargs["learning_rate"]))

        # Extract the mlflow environment variables from kwargs
        mlflow_agent = NullMLFlowAgent()
        try:
            mlflow_tracking_username = kwargs["mlflow_tracking_username"]
            mlflow_tracking_password = kwargs["mlflow_tracking_password"]
            mlflow_tracking_uri = kwargs["mlflow_tracking_uri"]

            if mlflow_tracking_uri:
                mlflow_agent = MLFlowAgent()
                os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
                os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password
                mlflow_agent.set_tracking_uri(mlflow_tracking_uri)
                # TODO: test mlflow_agent is connected to mlflow server
        except KeyError:
            pass

        try:
            cls._trainer = TrainerFactory.create_trainer(
                trainer_type,
                criterion=criterion,
                optimizer=optimizer,
                device=torch.device(kwargs["device"]),
                mlflow_agent=mlflow_agent
            )
            cls._trainer_store.add_trainer(trainer_id, cls._trainer)
        except Exception as e:
            print("Failed to init trainer")
            return False
        return True

    @classmethod
    def set_mlflow_model_name(cls, model_name: str) -> bool:
        """
        Design for an exposed REST api to let client set the mlflow model name
        :param model_name: The model name to be used in MLflow tracking
        :return: True if model name is successfully set
        """
        if cls._trainer is None:
            print("Trainer is not initialized")
            return False

        cls._trainer.set_mlflow_model_name(model_name)
        return True

    @classmethod
    def set_mlflow_experiment_name(cls, experiment_name: str) -> bool:
        """
        Design for an exposed REST api to let client set the mlflow experiment name
        :param experiment_name: The experiment name to be used in MLflow tracking
        :return: True if experiment name is successfully set
        """
        if cls._trainer is None:
            print("Trainer is not initialized")
            return False

        cls._trainer.set_mlflow_experiment_name(experiment_name)
        return True

    @classmethod
    def set_mlflow_run_name(cls, run_name: str) -> bool:
        """
        Design for an exposed REST api to let client set the mlflow run name
        :param run_name: The run name to be used in MLflow tracking
        :return: True if run name is successfully set
        """
        if cls._trainer is None:
            print("Trainer is not initialized")
            return False

        cls._trainer.set_mlflow_run_name(run_name)
        return True

    @classmethod
    def run_ml_training(cls, epochs: int, progress_callback=None) -> bool:
        """
        Once the data fetcher prepared and trainer is initialized
        Run the ml training process
        :param epochs: training epochs
        :param progress_callback: callback function for progress updates
        :return: True if training is successful
        """
        # Check the data processor is ready
        if cls._data_processor is None:
            print("Data processor is not initialized")
            return False

        cls._data_processor.preprocess_data()

        training_data = cls._data_processor.get_training_data_x()
        training_target = cls._data_processor.get_training_target_y()

        print(f"Training data shape: {training_data.shape}")
        print(f"Training target shape: {training_target.shape}")

        time_series_dataset = TimeSeriesDataset(training_data, training_target)
        torch_dataloader = DataLoader(
            time_series_dataset,
            batch_size=len(time_series_dataset),
            shuffle=False
        )

        # Check the model is initialized
        if cls._model is None:
            print("Model is not initialized")
            return False

        # Check the trainer is ready, model and training data is set
        if cls._trainer is None:
            print("Trainer is not initialized")
            return False
        else:
            cls._trainer.set_model(cls._model)
            cls._trainer.set_training_data_loader(torch_dataloader)

        try:
            print(f"Training the model for {epochs} epochs")
            cls._trainer.run_training_loop(epochs, progress_callback=progress_callback)
            print("Training finished")
        except RuntimeError as re:
            print(f"RuntimeError during training: {re}")
            return False

        cls._model.eval()
        return True

    @classmethod
    def get_model(cls, model_id: str):
        """
        Get the model by model_id
        :param model_id: The ID of the model to fetch
        :return: The model object if found, else None
        """
        return cls._model_store.get_model(model_id)

    @classmethod
    def get_trainer(cls, trainer_id: str):
        """
        Get the trainer by trainer_id
        :param trainer_id: The ID of the trainer to fetch
        :return: The trainer object if found, else None
        """
        return cls._trainer_store.get_trainer(trainer_id)

    @classmethod
    def get_data_processor(cls, data_processor_id: str):
        """
        Get the data processor by data_processor_id
        :param data_processor_id: The ID of the data processor to fetch
        :return: The data processor object if found, else None
        """
        return cls._data_processor_store.get_data_processor(data_processor_id)

    @classmethod
    def list_models(cls) -> list:
        return list(cls._model_store._model_store.keys())

    @classmethod
    def list_trainers(cls) -> list:
        return list(cls._trainer_store._trainer_store.keys())

    @classmethod
    def list_data_processors(cls) -> list:
        return list(cls._data_processor_store._data_processor_store.keys())

    @classmethod
    def update_model(cls, model_id: str, model_params: dict) -> bool:
        model = cls._model_store.get_model(model_id)
        if model:
            for param, value in model_params.items():
                setattr(model, param, value)
            return cls._model_store.update_model(model_id, model)
        return False

    @classmethod
    def update_trainer(cls, trainer_id: str, trainer_params: dict) -> bool:
        trainer = cls._trainer_store.get_trainer(trainer_id)
        if trainer:
            for param, value in trainer_params.items():
                setattr(trainer, param, value)
            return cls._trainer_store.update_trainer(trainer_id, trainer)
        return False

    @classmethod
    def update_data_processor(cls, data_processor_id: str, data_processor_params: dict) -> bool:
        data_processor = cls._data_processor_store.get_data_processor(data_processor_id)
        if data_processor:
            for param, value in data_processor_params.items():
                setattr(data_processor, param, value)

            # Ensure data shape is consistent with model input
            original_shape = data_processor.get_training_data_x().shape
            if cls._data_processor_store.update_data_processor(data_processor_id, data_processor):
                updated_shape = data_processor.get_training_data_x().shape
                print(f"Updated data processor {data_processor_id} with new params: {data_processor_params}")
                print(f"Original training data shape: {original_shape}")
                print(f"Updated training data shape: {updated_shape}")
                if updated_shape != original_shape:
                    print(
                        f"Warning: Data shape changed from {original_shape} to {updated_shape} which may not be compatible with the model")
                return True
        return False


def get_app():
    app = MLTrainingServingApp()
    return app
