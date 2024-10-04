import time
from torch.utils.data.dataloader import DataLoader
from src.ml_core.trainer.base_trainer import BaseTrainer
from src.model_ops_manager.mlflow_agent.mlflow_agent import NullMLFlowAgent
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TorchNeuralNetworkTrainer(BaseTrainer):
    """
    A concrete class for torch neural network models
    """

    def __init__(self, trainer_id, criterion, optimizer, device,
                 model=None,
                 training_data_loader=None,
                 mlflow_agent=NullMLFlowAgent(),
                 track_hyperparameters=True,
                 track_training_data_info=True,
                 track_metrics=True,
                 track_model_architecture=True):
        """
        Construction of a pytorch neural network trainer, provide basic information for training
        :param trainer_id: The unique identifier for the trainer
        :param criterion: The loss function used for training
        :param optimizer: The optimizer used for training
        :param device: The device on which the model will be trained (e.g., 'cpu' or 'cuda')
        :param model: The PyTorch model to be trained
        :param training_data_loader: The DataLoader for the training dataset
        :param mlflow_agent: Optional, the agent for tracking and registering the model to MLFlow server
        :param track_hyperparameters: Boolean flag to track hyperparameters
        :param track_training_data_info: Boolean flag to track training data information
        :param track_metrics: Boolean flag to track metrics
        :param track_model_architecture: Boolean flag to track model architecture
        """
        super(TorchNeuralNetworkTrainer, self).__init__(model)
        self._trainer_id = trainer_id
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._mlflow_agent = mlflow_agent
        self._training_data_loader = training_data_loader
        self._track_hyperparameters = track_hyperparameters
        self._track_training_data_info = track_training_data_info
        self._track_metrics = track_metrics
        self._track_model_architecture = track_model_architecture
        self._mlflow_model_name = "Pytorch_Model"  # Default model name
        self._mlflow_experiment_name = "ml-system-dev-test"  # Default experiment name
        self._mlflow_run_name = "Pytorch Run"  # Default run name

    def to_dict(self):
        """
        Serialize the trainer object to a dictionary.
        """
        return {
            'trainer_id': self._trainer_id,
            'trainer_type': self.__class__.__name__,
            'criterion': str(self._criterion),
            'optimizer': str(self._optimizer),
            'device': str(self._device),
            'model': self._model.to_dict() if self._model else None,
            'training_data_loader': None,  # DataLoader can't be easily serialized; consider handling this separately.
            'mlflow_agent': self._mlflow_agent.to_dict() if hasattr(self._mlflow_agent, 'to_dict') else None,
            'track_hyperparameters': self._track_hyperparameters,
            'track_training_data_info': self._track_training_data_info,
            'track_metrics': self._track_metrics,
            'track_model_architecture': self._track_model_architecture,
            'mlflow_model_name': self._mlflow_model_name,
            'mlflow_experiment_name': self._mlflow_experiment_name,
            'mlflow_run_name': self._mlflow_run_name,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Deserialize a dictionary to a TorchNeuralNetworkTrainer object.
        """
        instance = cls(
            trainer_id=data.get('trainer_id'),
            criterion=data.get('criterion'),
            optimizer=data.get('optimizer'),
            device=data.get('device'),
            model=None,  # Handle model separately if needed
            training_data_loader=None,  # Handle DataLoader separately if needed
            mlflow_agent=None,  # Handle MLFlow agent separately if needed
            track_hyperparameters=data.get('track_hyperparameters', True),
            track_training_data_info=data.get('track_training_data_info', True),
            track_metrics=data.get('track_metrics', True),
            track_model_architecture=data.get('track_model_architecture', True)
        )
        instance._mlflow_model_name = data.get('mlflow_model_name', "Pytorch_Model")
        instance._mlflow_experiment_name = data.get('mlflow_experiment_name', "ml-system-dev-test")
        instance._mlflow_run_name = data.get('mlflow_run_name', "Pytorch Run")
        return instance

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return self.__repr__()

    def set_model(self, model):
        """
        Provide the method to set the model
        If the model is not prepared before, this method can be used to set the model.
        The model is required to be a pytorch model, extract the model hyper-parameters from the model.
        :param model:
        :return:
        """
        self._model = model

    def set_training_data_loader(self, training_data_loader: DataLoader) -> None:
        """
        Provide the method to set the training data and label
        If the training tensor is not prepared before, this method can be used to set the training data and label
        :param training_data_loader: the torch DataLoader object
        :return:
        """
        self._training_data_loader = training_data_loader

    def set_mlflow_model_name(self, model_name: str) -> None:
        """
        Provide the method to set the MLflow model name for tracking
        :param model_name: The model name to be used in MLflow tracking
        :return: None
        """
        self._mlflow_model_name = model_name

    def set_mlflow_experiment_name(self, experiment_name: str) -> None:
        """
        Provide the method to set the MLflow experiment name for tracking
        :param experiment_name: The experiment name to be used in MLflow tracking
        :return: None
        """
        self._mlflow_experiment_name = experiment_name

    def set_mlflow_run_name(self, run_name: str) -> None:
        """
        Provide the method to set the MLflow run name for tracking
        :param run_name: The run name to be used in MLflow tracking
        :return: None
        """
        self._mlflow_run_name = run_name

    def log_training_data_info(self):
        """
        Log the information about the training data
        :return: None
        """
        training_data_info = {
            "num_samples": len(self._training_data_loader.dataset),
            "batch_size": self._training_data_loader.batch_size,
            # Add more data-related info if needed
        }
        self._mlflow_agent.log_params_many(training_data_info)
        logger.debug(f"Logged training data info: {training_data_info}")

    def run_training_loop(self, epochs: int, progress_callback=None) -> None:
        """
        Implement the training logic and the process pipeline
        :param epochs: The number of epochs to train the model
        :param progress_callback: Optional callback to provide real-time updates
        :return: None
        """
        # Check if the model and training data are ready
        if self._model is None:
            raise RuntimeError("Model is not provided.")

        if self._training_data_loader is None:
            raise RuntimeError("Training data loader is not provided.")

        self._mlflow_agent.start_run(
            experiment_name=self._mlflow_experiment_name,
            run_name=self._mlflow_run_name
        )

        if self._track_hyperparameters:
            # Log model hyperparameters
            model_hyper_parameters = self._model.get_model_hyper_parameters()
            self._mlflow_agent.log_params_many(model_hyper_parameters)
            logger.debug(f"Logged model hyperparameters: {model_hyper_parameters}")

        if self._track_training_data_info:
            # Log training data info
            self.log_training_data_info()

        logger.info(f"Training the model for {epochs} epochs")
        self._model.train()

        try:
            for epoch in range(epochs):
                epoch_loss = 0
                for i, data in enumerate(self._training_data_loader):
                    x, y = data
                    x = x.to(self._device)
                    y = y.to(self._device)

                    self._model.to(self._device)
                    self._model.train()
                    outputs = self._model(x)
                    loss = self._criterion(outputs, y)

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    epoch_loss += loss.item()

                    if self._track_metrics:
                        # Log loss metric
                        self._mlflow_agent.log_metric("loss", loss.item(), step=epoch)

                avg_epoch_loss = epoch_loss / len(self._training_data_loader)

                logger.info(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_epoch_loss}")

                # Callback for progress update
                if progress_callback:
                    progress_callback(epoch + 1, epochs, avg_epoch_loss)
        except Exception as e:
            logger.error(f"Error during training loop or model registration: {e}", exc_info=True)
            raise e
        finally:
            try:
                if self._track_model_architecture:
                    # Log model architecture
                    self._mlflow_agent.log_param("model_architecture", str(self._model))
                    logger.debug("Logged model architecture.")

                # Register the model
                logger.info(f"Registering the model {self._mlflow_model_name} to MLFlow server")
                self._mlflow_agent.register_model(self._model, self._mlflow_model_name)
                logger.info("Model registered with MLflow successfully.")
            except Exception as e:
                logger.error(f"Error during model registration: {e}", exc_info=True)
                raise e
            finally:
                self._mlflow_agent.end_run()
                logger.info("Training run ended.")