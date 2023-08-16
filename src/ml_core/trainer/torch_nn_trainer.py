import torch

from src.ml_core.trainer.base_trainer import BaseTrainer


class TorchNeuralNetworkTrainer(BaseTrainer):

    """
    A concrete class for torch neural network models
    """

    def __init__(self, model, criterion, optimizer, device,
                 training_data: torch.Tensor,
                 training_labels: torch.Tensor
                 ):
        super(TorchNeuralNetworkTrainer, self).__init__(model)
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._training_data = training_data.to(self._device)
        self._training_labels = training_labels.to(self._device)

    def run_training_loop(self, epochs: int) -> None:
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



