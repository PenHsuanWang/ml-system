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

        self._training_data_loader = training_data_loader

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

    def run_training_loop(self, epochs: int) -> None:
        """
        Implement the training logic and the process pipeline
        :param epochs:
        :return:
        """
        # to check the model and training data is ready
        if self._model is None:
            raise RuntimeError("Model is not provided.")
        # if self._training_data is None or self._training_labels is None:
        #     raise RuntimeError("Training data or label is not provided.")

        self._mlflow_agent.start_run(
            experiment_name="ml-system-dev-test",
            run_name="Pytorch Run"
        )

        # extract the model hyperparameters from the model
        model_hyper_parameters = self._model.get_model_hyper_parameters()
        self._mlflow_agent.log_params_many(model_hyper_parameters)

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

                """If the mlflow agent is provided, log the loss"""
                self._mlflow_agent.log_metric("loss", loss.item())

                print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}")

        """If the mlflow agent is provided, end the mlflow run"""

        self._mlflow_agent.register_model(self._model, "Pytorch_Model")

        self._mlflow_agent.end_run()

        print("Training done")



