import torch
import pytest
from src.ml_core.models.torch_nn_models.lstm_model import LSTMModel
from src.ml_core.models.torch_nn_models.base_model import BaseModel
from src.ml_core.models.torch_nn_models.model import TorchNeuralNetworkModelFactory


def test_inheritance():
    lstm_model = LSTMModel(10, 20, 5)
    assert isinstance(lstm_model, BaseModel)


def test_forward_pass():
    # Create an LSTM model instance
    lstm_model = LSTMModel(input_size=10, hidden_size=20, output_size=5)

    # Create a mock input tensor of shape (batch_size, sequence_length, input_size)
    x = torch.randn(32, 10, 10)

    # Run forward pass
    output = lstm_model(x)

    # Check the output shape
    assert output.shape == (32, 5)


def test_hyper_parameters():
    lstm_model = LSTMModel(input_size=10, hidden_size=20, output_size=5)
    hyper_params = lstm_model.get_model_hyper_parameters()

    # Verifying the shapes of LSTM and FC layers
    assert hyper_params['lstm1.weight_ih_l0'] == (80, 10)
    assert hyper_params['lstm1.weight_hh_l0'] == (80, 20)
    assert hyper_params['fc.weight'] == (5, 20)


def test_model_factory():
    lstm_model = TorchNeuralNetworkModelFactory.create_torch_nn_model("lstm", input_size=10, hidden_size=20,
                                                                      output_size=5)

    # Check if it is an instance of LSTMModel
    assert isinstance(lstm_model, LSTMModel)

    # Create a mock input tensor of shape (batch_size, sequence_length, input_size)
    x = torch.randn(32, 10, 10)

    # Run forward pass
    output = lstm_model(x)

    # Check the output shape
    assert output.shape == (32, 5)


def test_model_factory_exception():
    with pytest.raises(Exception):
        _ = TorchNeuralNetworkModelFactory.create_torch_nn_model("unknown", input_size=10, hidden_size=20,
                                                                 output_size=5)
