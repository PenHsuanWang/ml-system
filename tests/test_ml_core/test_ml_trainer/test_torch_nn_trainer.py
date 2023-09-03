import pytest
import torch
from torch import nn, optim

from src.ml_core.trainer.torch_nn_trainer import TorchNeuralNetworkTrainer
from src.ml_core.models.torch_nn_models.model import TorchNeuralNetworkModelFactory

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
    model = testing_model()
    trainer.set_model(model)

    assert trainer._model == model


def test_set_training_tensor():
    trainer = TorchNeuralNetworkTrainer(None, None, None)
    data = torch.tensor([[1.0], [2.0]])
    labels = torch.tensor([[1.0], [2.0]])

    trainer.set_training_tensor(data, labels)

    assert torch.equal(trainer._training_data, data)
    assert torch.equal(trainer._training_labels, labels)


def test_run_training_loop():
    # Create LSTM model from the factory
    model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
        model_type="lstm",
        input_size=1,
        hidden_size=10,
        output_size=1
    )
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    device = 'cpu'

    # Dimensions: (batch_size, seq_len, input_size)
    data = torch.tensor([[[i * 1.0] for i in range(1, 6)],
                         [[i * 1.0] for i in range(6, 11)]])  # Extended the sequence

    # Dimensions: (batch_size, output_size)
    labels = torch.tensor([[5.0], [10.0]])  # Updated labels

    trainer = TorchNeuralNetworkTrainer(criterion, optimizer, device, model=model, training_data=data,
                                        training_labels=labels)

    initial_params = [p.clone() for p in model.parameters()]

    trainer.run_training_loop(10)

    final_params = [p for p in model.parameters()]

    for p_initial, p_final in zip(initial_params, final_params):
        assert not torch.equal(p_initial, p_final), f"Initial params: {p_initial}, Final params: {p_final}"



def test_invalid_model_or_data():
    trainer = TorchNeuralNetworkTrainer(None, None, None)

    with pytest.raises(RuntimeError, match="Model is not provided."):
        trainer.run_training_loop(1)

    trainer.set_model(testing_model)

    with pytest.raises(RuntimeError, match="Training data or label is not provided."):
        trainer.run_training_loop(1)
