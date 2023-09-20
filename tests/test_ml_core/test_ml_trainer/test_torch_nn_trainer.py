import pytest
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from src.ml_core.trainer.torch_nn_trainer import TorchNeuralNetworkTrainer
from src.ml_core.models.torch_nn_models.model import TorchNeuralNetworkModelFactory
from src.ml_core.data_loader.base_dataset import TimeSeriesDataset

testing_model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
    model_type="lstm",
    input_size=1,
    hidden_size=10,
    output_size=1
)

# a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


def test_constructor():
    model = testing_model
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    device = 'cpu'

    trainer = TorchNeuralNetworkTrainer(criterion, optimizer, device, model=model)

    assert trainer._model == model
    assert trainer._criterion == criterion
    assert trainer._optimizer == optimizer
    assert trainer._device == 'cpu'  # device should be a string


def test_set_model():
    trainer = TorchNeuralNetworkTrainer(None, None, None)
    model = testing_model
    trainer.set_model(model)

    assert trainer._model == model


def test_set_training_tensor():
    trainer = TorchNeuralNetworkTrainer(None, None, None)
    data = torch.tensor([[1.0], [2.0]])
    labels = torch.tensor([[1.0], [2.0]])

    # Convert PyTorch tensors to NumPy arrays
    data_np = data.numpy()
    labels_np = labels.numpy()

    dataset = TimeSeriesDataset(data_np, labels_np)  # Use NumPy arrays

    # Create a DataLoader for the data and labels
    training_data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    trainer.set_training_data_loader(training_data_loader)

    assert trainer._training_data_loader == training_data_loader


def test_run_training_loop():
    # Create LSTM model from the factory
    model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
        model_type="lstm",
        input_size=1,
        hidden_size=10,
        output_size=1,
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    device = 'cpu'

    # Dimensions: (batch_size, seq_len, input_size)
    data = torch.tensor([[[i * 1.0] for i in range(1, 6)],
                         [[i * 1.0] for i in range(6, 11)]])  # Extended the sequence

    # Dimensions: (batch_size, output_size)
    labels = torch.tensor([[5.0], [10.0]])  # Updated labels

    # Convert PyTorch tensors to NumPy arrays
    data_np = data.numpy()
    labels_np = labels.numpy()

    # Create a DataLoader for the data and labels
    dataset = TimeSeriesDataset(data_np, labels_np)  # Use NumPy arrays

    trainer = TorchNeuralNetworkTrainer(criterion, optimizer, device, model=model, training_data_loader=dataset)

    initial_params = [p.clone() for p in model.parameters()]

    trainer.run_training_loop(10)

    final_params = [p for p in model.parameters()]

    for p_initial, p_final in zip(initial_params, final_params):
        assert not torch.equal(p_initial, p_final), f"Initial params: {p_initial}, Final params: {p_final}"



def test_invalid_model_or_data():
    # Create a dummy DataLoader with sample data
    dummy_data = torch.tensor([[1.0], [2.0]])
    dummy_labels = torch.tensor([[1.0], [2.0]])
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dummy_data_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1, shuffle=False)

    # Create the trainer with the dummy DataLoader
    trainer = TorchNeuralNetworkTrainer(None, None, None, training_data_loader=dummy_data_loader)

    with pytest.raises(RuntimeError, match="Model is not provided."):
        trainer.run_training_loop(1)

    trainer.set_model(testing_model)

    with pytest.raises(RuntimeError, match="Training data or label is not provided."):
        trainer.run_training_loop(1)
