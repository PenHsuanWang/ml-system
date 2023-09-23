import pytest
from src.webapp.ml_training_serving_app import MLTrainingServingApp
from unittest.mock import patch, Mock, MagicMock
from torch.nn import Module


@pytest.fixture
def mock_ml_training_app():
    return MLTrainingServingApp()

# Create a custom mock class for PyTorch models
class MockPyTorchModel(Module):
    def __init__(self):
        super(MockPyTorchModel, self).__init__()


def test_set_data_fetcher(mock_ml_training_app):
    with patch('src.webapp.data_io_serving_app.get_app') as mock_get_app:
        mock_get_app.return_value.data_fetcher = {"sample_fetcher": "data"}

        # Successful scenario
        assert mock_ml_training_app.set_data_fetcher("sample_fetcher") == True

        # Failure scenario
        assert mock_ml_training_app.set_data_fetcher("nonexistent_fetcher") == False


def test_fetcher_data(mock_ml_training_app):
    with patch('src.webapp.ml_training_serving_app.MLTrainingServingApp._data_fetcher',
               MagicMock()) as mock_data_fetcher:
        mock_data_fetcher.fetch_from_source.return_value = None

        assert mock_ml_training_app.fetcher_data([], {"stock_id": 1, "start_date": "2021-01-01",
                                                      "end_date": "2021-12-31"}) == True

        mock_data_fetcher.fetch_from_source.side_effect = RuntimeError
        assert mock_ml_training_app.fetcher_data([], {"stock_id": 1, "start_date": "2021-01-01",
                                                      "end_date": "2021-12-31"}) == False


def test_init_data_preprocessor(mock_ml_training_app):
    with patch.object(mock_ml_training_app, '_data_fetcher', MagicMock()) as mock_data_fetcher:
        mock_data_fetcher.get_as_dataframe.return_value = None

        assert mock_ml_training_app.init_data_preprocessor("some_type") == False

        mock_data_fetcher.get_as_dataframe.side_effect = ValueError("mocked error")
        assert mock_ml_training_app.init_data_preprocessor("some_type") == False


def test_init_model(mock_ml_training_app):
    assert mock_ml_training_app.init_model("mock_model") == False
    with patch('src.ml_core.models.torch_nn_models.model.TorchNeuralNetworkModelFactory.create_torch_nn_model') as mock_factory:
        mock_factory.return_value = None
        assert mock_ml_training_app.init_model("mock_model") == True


@patch.object(MLTrainingServingApp, '_model', MockPyTorchModel())  # Create an instance of the custom mock model
@patch('src.webapp.ml_training_serving_app.TorchNeuralNetworkModelFactory.create_torch_nn_model')
def test_init_trainer(mock_create_torch_nn_model, mock_ml_training_app):
    # Create a mock PyTorch model
    mock_pytorch_model = MockPyTorchModel()

    # Configure the mock to return an iterable of mock parameters when parameters is accessed
    mock_parameters = [Mock(), Mock()]  # Add more Mocks as needed
    mock_pytorch_model.parameters = Mock(return_value=list(mock_parameters))

    # Configure the mock to return the mock PyTorch model when create_torch_nn_model is called
    mock_create_torch_nn_model.return_value = mock_pytorch_model

    # Test init_trainer with successful initialization
    assert mock_ml_training_app.init_trainer("mock_trainer", loss_function="mse", optimizer="adam",
                                             learning_rate="0.01", device="cpu") is True

    # Test init_trainer with an exception raised
    mock_create_torch_nn_model.side_effect = Exception("mocked error")
    assert mock_ml_training_app.init_trainer("mock_trainer", loss_function="mse", optimizer="adam",
                                             learning_rate="0.01", device="cpu") is False


def test_run_ml_training(mock_ml_training_app):
    with patch.object(mock_ml_training_app, '_data_processor', MagicMock()), patch.object(mock_ml_training_app,
                                                                                          '_model',
                                                                                          MagicMock()), patch.object(
            mock_ml_training_app, '_trainer', MagicMock()):
        mock_ml_training_app._data_processor.get_training_data_x.return_value = None
        mock_ml_training_app._data_processor.get_training_target_y.return_value = None
        mock_ml_training_app._trainer.run_training_loop.return_value = None

        assert mock_ml_training_app.run_ml_training(10) == True

        mock_ml_training_app._trainer.run_training_loop.side_effect = RuntimeError("mocked error")
        assert mock_ml_training_app.run_ml_training(10) == False

        mock_ml_training_app._data_processor = None
        assert mock_ml_training_app.run_ml_training(10) == False

        mock_ml_training_app._data_processor = MagicMock()
        mock_ml_training_app._model = None
        assert mock_ml_training_app.run_ml_training(10) == False

        mock_ml_training_app._model = MagicMock()
        mock_ml_training_app._trainer = None
        assert mock_ml_training_app.run_ml_training(10) == False

