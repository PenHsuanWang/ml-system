import os

import threading

import mlflow
import torch
from torch.utils.data.dataloader import DataLoader

import src.webapp.data_io_serving_app
import src.store.data_processor_store
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

    # internal module tools for ml training job
    _data_fetcher = None
    _data_processor = None
    _trainer = None

    # the port of object join the ml training job
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
        # if the data fetcher is already initialized, overwrite it.
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
    def init_data_processor(cls, data_processor_type: str, **kwargs) -> bool:
        """
        Design for an exposed REST api to let client init the data preprocessor
        To initialize the data preprocessor
        Initialize parameters from kwargs provided by client via REST api request body.
        if data fetcher is ready:
         check the raw dataframe is ready or not. trigger data fetcher to fetch data and get the raw dataframe.
        or raise exception to teach user to init data fetcher first.
        :param data_processor_type:
        :param kwargs:
        :return:
        """

        # check the self._raw_pandas_dataframe is ready or not, else get the raw dataframe from data fetcher
        # if the fetcher is not initialized, raise exception to teach user to init data fetcher first.
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
            # register the data processor to data processor manager
            cls._data_processor_store.add_data_processor(
                data_processor_id="pytorch_lstm_aapl",
                data_processor=cls._data_processor
            )
        except Exception as e:
            print("Failed to init data processor")
            return False

        return True

    @classmethod
    def init_model(cls, model_type: str, **kwargs) -> bool:
        """
        Design for an exposed REST api to let client init the model
        To initialize the model
        Initialize parameters from kwargs provided by client via REST api request body.
        :param model_type:
        :param kwargs:
        :return:
        """
        try:
            cls._model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
                model_type,
                **kwargs
            )
        except Exception as e:
            print("Failed to init model")
            return False
        return True

    @classmethod
    def init_trainer(cls, trainer_type: str, **kwargs) -> bool:
        """
        Design for an exposed REST api to let client init the trainer
        To initialize the trainer
        Initialize parameters from kwargs provided by client via REST api request body.
        The trainer provide the function to add mlflow agent to log the training process and register the model
        create mlflow agent and setting the tracking uri here.
        :param trainer_type:
        :param kwargs:
        :return:
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

        # extract the mlflow environment variables from kwargs

        mlflow_agent = NullMLFlowAgent()
        try:
            mlflow_tracking_username = kwargs["mlflow_tracking_username"]
            mlflow_tracking_password = kwargs["mlflow_tracking_password"]
            mlflow_tracking_uri = kwargs["mlflow_tracking_uri"]

            # check mlflow_tracking_uri is provided and valid, else skip mlflow agent initialization
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
        except Exception as e:
            print("Failed to init trainer")
            return False
        return True

    @classmethod
    def run_ml_training(cls, epochs: int) -> bool:
        """
        Once the data fetcher prepared and trainer is initialized
        Run the ml training process
        :param epochs: training epochs
        :return: True if training is successful
        """

        # check the data_preprocessor is ready
        if cls._data_processor is None:
            print("Data processor is not initialized")
            return False

        cls._data_processor.preprocess_data()

        time_series_dataset = TimeSeriesDataset(
            cls._data_processor.get_training_data_x(),
            cls._data_processor.get_training_target_y()
        )
        torch_dataloader = DataLoader(
            time_series_dataset,
            batch_size=len(time_series_dataset),
            shuffle=False
        )


        # check the model is initialized
        if cls._model is None:
            print("Model is not initialized")
            return False

        # check the trainer is ready, model and training data is set
        if cls._trainer is None:
            print("Trainer is not initialized")
            return False
        else:
            cls._trainer.set_model(cls._model)
            cls._trainer.set_training_data_loader(torch_dataloader)

        try:
            print(f"Training the model for {epochs} epochs")
            cls._trainer.run_training_loop(epochs)
            print("Training finished")
        except RuntimeError as re:
            print(re)
            return False

        cls._model.eval()

    @classmethod
    def get_model(cls):
        return cls._model


def get_app():
    app = MLTrainingServingApp()
    return app

