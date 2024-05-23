from torch.utils.data.dataloader import DataLoader

from src.ml_core.trainer.base_trainer import BaseTrainer
from src.model_ops_manager.mlflow_agent.mlflow_agent import NullMLFlowAgent


class TorchNeuralNetworkTrainer(BaseTrainer):

    """
    A concrete class for torch neural network models
    """

    def __init__(self, criterion, optimizer, device,
                 model=None,
                 training_data_loader=None,
                 mlflow_agent=NullMLFlowAgent(),
                 track_hyperparameters=True,
                 track_training_data_info=True,
                 track_metrics=True,
                 track_model_architecture=True
                 ):
        """
        Construction of a pytorch neural network trainer, provide basic information for training
        the parts of mlflow_agent is using for tracking and registering the model to MLFlow server.
        The mlflow_agent should be passed by the outer scope, if not, the NullMLFlowAgent will be used.
        The NullMLFlowAgent is a dummy agent, which will not do anything to let the following training process work
        without mlflow_agent.
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
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._mlflow_agent = mlflow_agent
        self._training_data_loader = training_data_loader
        self._track_hyperparameters = track_hyperparameters
        self._track_training_data_info = track_training_data_info
        self._track_metrics = track_metrics
        self._track_model_architecture = track_model_architecture

    def set_model(self, model):
        """
        Provide the method to set the model
        If the model is not prepare before, this method can be used to set the model.
        The model is required to be a pytorch model, extract the model hyper-parameters from the model.
        :param model:
        :return:
        """
        self._model = model

    def set_training_data_loader(self, training_data_loader: DataLoader) -> None:
        """
        Provide the method to set the training data and label
        If the training tensor id not prepare before, this method can be used to set the training data and label
        :param training_data_loader: the torch DataLoader object
        :return:
        """
        self._training_data_loader = training_data_loader

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

    def run_training_loop(self, epochs: int) -> None:
        """
        Implement the training logic and the process pipeline
        :param epochs: The number of epochs to train the model
        :return: None
        """
        # Check if the model and training data are ready
        if self._model is None:
            raise RuntimeError("Model is not provided.")

        self._mlflow_agent.start_run(
            experiment_name="ml-system-dev-test",
            run_name="Pytorch Run"
        )

        if self._track_hyperparameters:
            # Log model hyperparameters
            model_hyper_parameters = self._model.get_model_hyper_parameters()
            self._mlflow_agent.log_params_many(model_hyper_parameters)

        if self._track_training_data_info:
            # Log training data info
            self.log_training_data_info()

        print(f"Training the model for {epochs} epochs")
        self._model.train()

        for epoch in range(epochs):
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

                if self._track_metrics:
                    # Log loss metric
                    self._mlflow_agent.log_metric("loss", loss.item(), step=epoch)

                print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}")

        if self._track_model_architecture:
            # Log model architecture
            self._mlflow_agent.log_param("model_architecture", str(self._model))

        # Register the model
        self._mlflow_agent.register_model(self._model, "Pytorch_Model")
        self._mlflow_agent.end_run()
        print("Training done")