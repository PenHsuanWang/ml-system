import torch

from src.ml_core.trainer.base_trainer import BaseTrainer


class TorchNeuralNetworkTrainer(BaseTrainer):

    """
    A concrete class for torch neural network models
    """

    def __init__(self, criterion, optimizer, device,
                 model=None,
                 training_data=None,
                 training_labels=None
                 ):
        super(TorchNeuralNetworkTrainer, self).__init__(model)
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = 'mps'

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

        # to check the model and training data is ready
        if self._model is None:
            raise RuntimeError("Model is not provided.")
        if self._training_data is None or self._training_labels is None:
            raise RuntimeError("Training data or label is not provided.")

        print(f"Training the model for {epochs} epochs")
        self._model.train()
        for epoch in range(epochs):
            self._model.to(self._device)
            self._model.train()
            outputs = self._model(self._training_data)
            loss = self._criterion(outputs, self._training_labels)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}")
        print("Training done")



