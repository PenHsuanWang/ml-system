# src/webapp/ml_training_serving_app.py

import os
import mlflow
import torch
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import json
import threading
import logging

import src.webapp.data_io_serving_app
import src.store.data_processor_store
import src.store.trainer_store
import src.store.model_store
from src.ml_core.data_processor.data_processor import DataProcessorFactory
from src.ml_core.data_loader.base_dataset import TimeSeriesDataset
from src.ml_core.models.torch_nn_models.model import TorchNeuralNetworkModelFactory
from src.ml_core.trainer.trainer import TrainerFactory
from src.model_ops_manager.mlflow_agent.mlflow_agent import MLFlowAgent, NullMLFlowAgent
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLTrainingServingApp:
    """
    A singleton class to serve the model training process
    provided the operation interface to client via REST api
    """

    _data_io_serving_app = src.webapp.data_io_serving_app.get_app()
    _data_processor_store = src.store.data_processor_store.get_store()
    _trainer_store = src.store.trainer_store.get_trainer_store()
    _model_store = src.store.model_store.get_model_store()

    _data_fetcher = None
    _raw_pandas_dataframe = None
    _training_progress = {}
    _lock = threading.Lock()
    _model = None  # Added to keep track of the current model

    def __init__(self):
        """
        All the field needed for ml process is put in class level
        """
        pass

    @classmethod
    def set_data_fetcher(cls, data_fetcher_name: str) -> bool:
        """
        Set the data fetcher using the provided name.
        :param data_fetcher_name: Name of the data fetcher.
        :return: True if successful, False otherwise.
        """
        cls._data_io_serving_app = src.webapp.data_io_serving_app.get_app()
        try:
            cls._data_fetcher = cls._data_io_serving_app.data_fetcher[data_fetcher_name]
            logger.info(f"Data fetcher '{data_fetcher_name}' set successfully.")
        except KeyError:
            logger.error(f"Data fetcher '{data_fetcher_name}' not found.")
            return False
        return True

    @classmethod
    def fetcher_data(cls, args: list, kwargs: dict) -> bool:
        """
        Fetch data from source using the data fetcher.
        :param args: Arguments for data fetcher.
        :param kwargs: Keyword arguments for data fetcher.
        :return: True if data fetched successfully, False otherwise.
        """
        if cls._data_fetcher is None:
            logger.error("Data fetcher is not initialized.")
            return False

        try:
            cls._data_fetcher.fetch_from_source(
                stock_id=kwargs.get("stock_id"),
                start_date=kwargs.get("start_date"),
                end_date=kwargs.get("end_date"),
            )
            logger.info("Data fetched from source successfully.")
        except Exception as e:
            logger.error(f"Failed to fetch data from source: {e}")
            return False
        return True

    @classmethod
    def init_data_processor_from_df(cls, data_processor_id: str, data_processor_type: str, dataframe_json: dict,
                                    **kwargs) -> bool:
        """
        Initialize the data processor from a JSON-encoded DataFrame.
        :param data_processor_id: The ID of the data processor to initialize.
        :param data_processor_type: The type of data processor to initialize.
        :param dataframe_json: JSON-encoded DataFrame.
        :param kwargs: Additional parameters for the data processor.
        :return: True if data processor is successfully initialized.
        """
        try:
            dataframe = pd.DataFrame.from_records(dataframe_json["data"], columns=dataframe_json["columns"])
            cls._raw_pandas_dataframe = dataframe
            logger.info(f"DataFrame initialized with columns: {dataframe.columns.tolist()}")

            data_processor = DataProcessorFactory.create_data_processor(
                data_processor_type,
                input_data=cls._raw_pandas_dataframe,
                **kwargs
            )
            logger.info(f"Created data processor: {data_processor}")

            cls._data_processor_store.add_data_processor(
                data_processor_id=data_processor_id,
                data_processor=data_processor
            )
            logger.info(f"Data processor stored with ID: {data_processor_id}")
        except Exception as e:
            logger.error(f"Failed to init data processor from DataFrame: {e}")
            return False
        return True

    @classmethod
    def init_data_processor(cls, data_processor_id: str, data_processor_type: str, dataframe: pd.DataFrame = None,
                            **kwargs) -> bool:
        """
        Initialize the data processor.
        :param data_processor_id: The ID of the data processor to initialize.
        :param data_processor_type: The type of data processor to initialize.
        :param dataframe: Optional pandas DataFrame containing the data to process.
        :param kwargs: Additional parameters for the data processor.
        :return: True if data processor is successfully initialized.
        """
        try:
            if dataframe is not None:
                cls._raw_pandas_dataframe = dataframe
            else:
                if cls._data_fetcher is None:
                    logger.error("Data fetcher is not initialized.")
                    return False
                if cls._raw_pandas_dataframe is None:
                    cls._raw_pandas_dataframe = cls._data_fetcher.get_as_dataframe()

            data_processor = DataProcessorFactory.create_data_processor(
                data_processor_type,
                input_data=cls._raw_pandas_dataframe,
                **kwargs
            )
            cls._data_processor_store.add_data_processor(
                data_processor_id=data_processor_id,
                data_processor=data_processor
            )
            logger.info(f"Data processor '{data_processor_id}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to init data processor: {e}")
            return False
        return True

    @classmethod
    def init_model(cls, model_type: str, model_id: str, **kwargs) -> bool:
        """
        Initialize the model.
        :param model_type: The type of model to initialize.
        :param model_id: The ID to associate with the model in the store.
        :param kwargs: Additional parameters for the model.
        :return: True if model is successfully initialized.
        """
        try:
            cls._model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
                model_type,
                **kwargs
            )
            cls._model_store.add_model(model_id, cls._model)
            logger.info(f"Model '{model_id}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to init model: {e}")
            return False
        return True

    @classmethod
    def init_trainer(cls, trainer_type: str, trainer_id: str, **kwargs) -> bool:
        """
        Initialize the trainer.
        :param trainer_type: The type of trainer to initialize.
        :param trainer_id: The ID to associate with the trainer in the store.
        :param kwargs: Additional parameters for the trainer.
        :return: True if trainer is successfully initialized.
        """
        if cls._model is None:
            logger.error("Model is not initialized.")
            return False

        try:
            # Initialize criterion
            loss_function = kwargs.get("loss_function", "mse").lower()
            if loss_function == "mse":
                criterion = torch.nn.MSELoss()
            elif loss_function == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss()
            else:
                logger.error(f"Unsupported loss function: {loss_function}")
                return False

            # Initialize optimizer
            optimizer_type = kwargs.get("optimizer", "adam").lower()
            learning_rate = float(kwargs.get("learning_rate", 0.001))
            if optimizer_type == "adam":
                optimizer = torch.optim.Adam(cls._model.parameters(), lr=learning_rate)
            elif optimizer_type == "sgd":
                optimizer = torch.optim.SGD(cls._model.parameters(), lr=learning_rate)
            else:
                logger.error(f"Unsupported optimizer: {optimizer_type}")
                return False

            # Initialize MLFlow agent if tracking URI is provided
            mlflow_agent = NullMLFlowAgent()
            mlflow_tracking_uri = kwargs.get("mlflow_tracking_uri")
            if mlflow_tracking_uri:
                mlflow_agent = MLFlowAgent()
                mlflow_tracking_username = kwargs.get("mlflow_tracking_username")
                mlflow_tracking_password = kwargs.get("mlflow_tracking_password")
                if mlflow_tracking_username and mlflow_tracking_password:
                    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
                    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password
                mlflow_agent.set_tracking_uri(mlflow_tracking_uri)

            # Create the trainer
            trainer = TrainerFactory.create_trainer(
                trainer_type,
                trainer_id=trainer_id,
                criterion=criterion,
                optimizer=optimizer,
                device=torch.device(kwargs.get("device", "cpu")),
                mlflow_agent=mlflow_agent
            )
            cls._trainer_store.add_trainer(trainer_id, trainer)
            logger.info(f"Trainer '{trainer_id}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to init trainer: {e}")
            return False
        return True

    @classmethod
    def set_mlflow_model_name(cls, model_name: str) -> bool:
        """
        Set the MLflow model name for tracking.
        :param model_name: The model name to be used in MLflow tracking.
        :return: True if successful, False otherwise.
        """
        try:
            trainer_id = cls._trainer_store.list_trainers()[-1]
            trainer = cls._trainer_store.get_trainer(trainer_id)
            if trainer is None:
                logger.error("Trainer is not initialized.")
                return False
            trainer.set_mlflow_model_name(model_name)
            logger.info(f"MLflow model name set to '{model_name}'.")
            return True
        except IndexError:
            logger.error("No trainers available to set MLflow model name.")
            return False
        except Exception as e:
            logger.error(f"Error setting MLflow model name: {e}")
            return False

    @classmethod
    def set_mlflow_experiment_name(cls, experiment_name: str) -> bool:
        """
        Set the MLflow experiment name for tracking.
        :param experiment_name: The experiment name to be used in MLflow tracking.
        :return: True if successful, False otherwise.
        """
        try:
            trainer_id = cls._trainer_store.list_trainers()[-1]
            trainer = cls._trainer_store.get_trainer(trainer_id)
            if trainer is None:
                logger.error("Trainer is not initialized.")
                return False
            trainer.set_mlflow_experiment_name(experiment_name)
            logger.info(f"MLflow experiment name set to '{experiment_name}'.")
            return True
        except IndexError:
            logger.error("No trainers available to set MLflow experiment name.")
            return False
        except Exception as e:
            logger.error(f"Error setting MLflow experiment name: {e}")
            return False

    @classmethod
    def set_mlflow_run_name(cls, run_name: str) -> bool:
        """
        Set the MLflow run name for tracking.
        :param run_name: The run name to be used in MLflow tracking.
        :return: True if successful, False otherwise.
        """
        try:
            trainer_id = cls._trainer_store.list_trainers()[-1]
            trainer = cls._trainer_store.get_trainer(trainer_id)
            if trainer is None:
                logger.error("Trainer is not initialized.")
                return False
            trainer.set_mlflow_run_name(run_name)
            logger.info(f"MLflow run name set to '{run_name}'.")
            return True
        except IndexError:
            logger.error("No trainers available to set MLflow run name.")
            return False
        except Exception as e:
            logger.error(f"Error setting MLflow run name: {e}")
            return False

    @classmethod
    def run_ml_training(cls, trainer_id: str, epochs: int, progress_callback=None) -> bool:
        """
        Run the machine learning training process.
        :param trainer_id: ID of the trainer to use for training.
        :param epochs: Number of epochs to train the model.
        :param progress_callback: Optional; callback for real-time updates.
        :return: True if training is successful, False otherwise.
        """
        try:
            trainer = cls._trainer_store.get_trainer(trainer_id)
            if not trainer:
                logger.error(f"Trainer '{trainer_id}' not found.")
                return False

            if cls._model is None:
                logger.error("Model is not initialized.")
                return False
            trainer.set_model(cls._model)

            data_processor_id = cls._data_processor_store.list_data_processors()[-1]
            data_processor = cls._data_processor_store.get_data_processor(data_processor_id)
            if data_processor is None:
                logger.error("Data processor is not initialized.")
                return False

            data_processor.preprocess_data(force=True)  # Force reprocessing to ensure latest data
            training_data = data_processor.get_training_data_x()
            training_target = data_processor.get_training_target_y()

            logger.info(f"Training data shape: {training_data.shape}")
            logger.info(f"Training target shape: {training_target.shape}")

            input_size = cls._model.input_size
            if training_data.shape[-1] != input_size:
                logger.error(
                    f"Input size mismatch: Model expects input_size={input_size}, "
                    f"but got data with shape {training_data.shape}"
                )
                return False

            if training_data.shape[0] != training_target.shape[0]:
                logger.error(
                    f"Mismatch in training data and target sizes: {training_data.shape[0]} vs {training_target.shape[0]}"
                )
                return False

            time_series_dataset = TimeSeriesDataset(training_data, training_target)
            torch_dataloader = DataLoader(
                time_series_dataset,
                batch_size=len(time_series_dataset),
                shuffle=False
            )
            trainer.set_training_data_loader(torch_dataloader)

            logger.info(f"Starting training for {epochs} epochs.")

            # Define internal progress callback
            def internal_progress_callback(epoch, total_epochs, loss):
                cls.update_progress(trainer_id, epoch, loss)
                if progress_callback:
                    progress_callback(epoch, total_epochs, loss)

            trainer.run_training_loop(epochs, progress_callback=internal_progress_callback)
            logger.info("Training finished successfully.")
        except Exception as e:
            logger.error(f"Exception during training: {e}", exc_info=True)
            cls.update_progress(trainer_id, 'error', 0)
            return False

        cls._model.eval()
        cls.update_progress(trainer_id, 'finished', 0)
        return True

    @classmethod
    def get_model(cls, model_id: str):
        """
        Get the model by model_id.
        :param model_id: The ID of the model to fetch.
        :return: The model object if found, else None.
        """
        return cls._model_store.get_model(model_id)

    @classmethod
    def get_trainer(cls, trainer_id: str):
        """
        Get the trainer by trainer_id.
        :param trainer_id: The ID of the trainer to fetch.
        :return: The trainer object if found, else None.
        """
        return cls._trainer_store.get_trainer(trainer_id)

    @classmethod
    def get_trainer_details(cls, trainer_id: str) -> dict:
        trainer = cls._trainer_store.get_trainer(trainer_id)
        return trainer.to_dict() if trainer else {}

    @classmethod
    def get_data_processor(cls, data_processor_id: str):
        """
        Get the data processor by data_processor_id.
        :param data_processor_id: The ID of the data processor to fetch.
        :return: The data processor object if found, else None.
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
            try:
                for param, value in model_params.items():
                    if hasattr(model, param):
                        setattr(model, param, value)
                    else:
                        logger.warning(f"Model does not have attribute '{param}'.")
                return cls._model_store.update_model(model_id, model)
            except Exception as e:
                logger.error(f"Failed to update model '{model_id}': {e}")
                return False
        logger.error(f"Model '{model_id}' not found.")
        return False

    @classmethod
    def update_trainer(cls, trainer_id: str, trainer_params: dict) -> bool:
        trainer = cls._trainer_store.get_trainer(trainer_id)
        if trainer:
            try:
                for param, value in trainer_params.items():
                    if hasattr(trainer, param):
                        setattr(trainer, param, value)
                    else:
                        logger.warning(f"Trainer does not have attribute '{param}'.")
                return cls._trainer_store.update_trainer(trainer_id, trainer)
            except Exception as e:
                logger.error(f"Failed to update trainer '{trainer_id}': {e}")
                return False
        logger.error(f"Trainer '{trainer_id}' not found.")
        return False

    @classmethod
    def update_data_processor(cls, data_processor_id: str, data_processor_params: dict) -> bool:
        data_processor = cls._data_processor_store.get_data_processor(data_processor_id)
        if data_processor:
            try:
                for param, value in data_processor_params.items():
                    if hasattr(data_processor, param):
                        setattr(data_processor, param, value)
                    else:
                        logger.warning(f"Data processor does not have attribute '{param}'.")
                data_processor.preprocess_data()
                return cls._data_processor_store.update_data_processor(data_processor_id, data_processor)
            except Exception as e:
                logger.error(f"Failed to update data processor '{data_processor_id}': {e}")
                return False
        logger.error(f"Data processor '{data_processor_id}' not found.")
        return False

    @classmethod
    def update_progress(cls, trainer_id: str, epoch: int, loss: float):
        """
        Update the training progress for the given trainer.
        :param trainer_id: The ID of the trainer.
        :param epoch: The current epoch number.
        :param loss: The current loss value.
        """
        with cls._lock:
            if trainer_id not in cls._training_progress:
                cls._training_progress[trainer_id] = {}
            cls._training_progress[trainer_id][epoch] = loss
            if epoch == 'finished' or epoch == 'error':
                cls._training_progress[trainer_id]['finished'] = True

    @classmethod
    def remove_trainer(cls, trainer_id: str) -> bool:
        """
        Remove a trainer by ID.
        :param trainer_id: The ID of the trainer to remove.
        :return: True if trainer was successfully removed, False otherwise.
        """
        if cls._trainer_store.remove_trainer(trainer_id):
            logger.info(f"Trainer '{trainer_id}' removed successfully.")
            return True
        logger.error(f"Trainer '{trainer_id}' not found.")
        return False


def jsonable_encoder(obj):
    """
    Custom jsonable encoder to handle numpy objects and other non-serializable types.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='split')
    return str(obj)


def get_app():
    app = MLTrainingServingApp()
    return app

