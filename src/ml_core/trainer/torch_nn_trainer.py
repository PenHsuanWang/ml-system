import torch

from src.ml_core.trainer.base_trainer import BaseTrainer
from src.model_ops_manager.mlflow_agent.mlflow_agent import NullMLFlowAgent


class TorchNeuralNetworkTrainer(BaseTrainer):

    """
    A concrete class for torch neural network models
    """

    def __init__(self, criterion, optimizer, device,
                 model=None,
                 training_data=None,
                 training_labels=None,
                 mlflow_agent=NullMLFlowAgent()
                 ):
        """
        Construction of a pytorch neural network trainer, provide basic information for training
        the parts of mlflow_agent is using for tracking and registering the model to MLFlow server.
        The mlflow_agent should be passed by the outer scpoe, if not, the NullMLFlowAgent will be used.
        The NullMLFlowAgent is a dummy agent, which will not do anything to let the following training process works
        without mlflow_agent.
        :param criterion:
        :param optimizer:
        :param device:
        :param model:
        :param training_data:
        :param training_labels:
        :param mlflow_agent: Optional, the agent for tracking and registering the model to MLFlow server.
        """
        super(TorchNeuralNetworkTrainer, self).__init__(model)
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._mlflow_agent = mlflow_agent

        # it is allowed to set the training data and label later
        if training_data is not None and training_labels is not None:
            self._training_data = training_data.to(self._device)
            self._training_labels = training_labels.to(self._device)
        else:
            self._training_data = None
            self._training_labels = None

    def set_model(self, model):
        """
        Provide the method to set the model
        If the model is not prepare before, this method can be used to set the model
        :param model:
        :return:
        """
        self._model = model

    def set_training_tensor(self, training_data: torch.Tensor, trainer_label: torch.Tensor) -> None:
        """
        Provide the method to set the training data and label
        If the training tensor id not prepare before, this method can be used to set the training data and label
        :param training_data:
        :param trainer_label:
        :return:
        """
        self._training_data = training_data.to(self._device)
        self._training_labels = trainer_label.to(self._device)

    def run_training_loop(self, epochs: int) -> None:
        """
        Implement the training logic and the process pipeline
        :param epochs:
        :return:
        """
        # to check the model and training data is ready
        if self._model is None:
            raise RuntimeError("Model is not provided.")
        if self._training_data is None or self._training_labels is None:
            raise RuntimeError("Training data or label is not provided.")

        print(f"Training the model for {epochs} epochs")
        self._model.train()

        self._mlflow_agent.start_run(
            experiment_name="Pytorch Experiment",
            run_name="Pytorch Run"
        )

        for epoch in range(epochs):
            self._model.to(self._device)
            self._model.train()
            outputs = self._model(self._training_data)
            loss = self._criterion(outputs, self._training_labels)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            """If the mlflow agent is provided, log the loss"""
            self._mlflow_agent.log_metric("loss", loss.item())

            print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}")

        """If the mlflow agent is provided, end the mlflow run"""
        self._mlflow_agent.end_run()

        print("Training done")



