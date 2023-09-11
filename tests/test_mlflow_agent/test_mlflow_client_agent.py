import mlflow
import pytest
from unittest.mock import Mock
from src.model_ops_manager.mlflow_agent.client import MLFlowClientModelAgent
from src.model_ops_manager.mlflow_agent.model_downloader import MLFlowClientModelLoader

# Create a mock MlflowClient
mlflow_client_mock = Mock()

# Patch the MLFlowClient.mlflow_client attribute
@pytest.fixture(autouse=True)
def setup_mlflow_client():
    MLFlowClientModelAgent.mlflow_client = mlflow_client_mock
    MLFlowClientModelLoader.mlflow_client = mlflow_client_mock

# Define test cases
class TestMLFlowClientModelAgent:

    def test_is_model_name_exist(self):
        mlflow_client_mock.get_registered_model.return_value = {"name": "test_model"}
        assert MLFlowClientModelAgent.is_model_name_exist("test_model") is True

    def test_get_target_model_version_with_version(self):
        assert MLFlowClientModelAgent.get_latest_model_version("test_model", model_version=2) == 2

    def test_get_target_model_version_with_stage(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]
        assert MLFlowClientModelAgent.get_latest_model_version("test_model", model_stage="Staging") == 1

    def test_get_target_model_version_invalid_stage(self):
        with pytest.raises(ValueError):
            MLFlowClientModelAgent.get_latest_model_version("test_model", model_stage="Invalid")

    def test_get_model_latest_version(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]
        assert MLFlowClientModelAgent.get_model_latest_version("test_model") == 1

    def test_compose_model_uri_with_version(self):
        assert MLFlowClientModelAgent.compose_model_uri("test_model", model_version=1) == "models:/test_model/1"

    def test_compose_model_uri_with_stage(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 2}]
        assert MLFlowClientModelAgent.compose_model_uri("test_model", model_stage="Staging") == "models:/test_model/2"


@pytest.fixture
def mlflow_model_loader():
    mlflow.set_tracking_uri("http://localhost:5011")
    mlflow_model_loader = MLFlowClientModelLoader()
    mlflow_model_loader.init_mlflow_client()
    return mlflow_model_loader

def test_parse_model_name(mlflow_model_loader):
    # Test when only the model name is provided
    model_uri = mlflow_model_loader._parsing_adhoc_input_to_model_uri("Pytorch_Model")
    assert model_uri == "models:/Pytorch_Model"

def test_parse_model_name_and_version(mlflow_model_loader):
    # Test when model name and version are provided
    model_uri = mlflow_model_loader._parsing_adhoc_input_to_model_uri("Pytorch_Model", 1)
    assert model_uri == "models:/Pytorch_Model/1"

def test_parse_model_name_and_stage(mlflow_model_loader):
    # Test when model name and stage are provided
    model_uri = mlflow_model_loader._parsing_adhoc_input_to_model_uri("Pytorch_Model", "Staging")
    assert model_uri == "models:/Pytorch_Model/Staging"

def test_parse_model_name_version_and_stage(mlflow_model_loader):
    # Test when model name, version, and stage are provided
    model_uri = mlflow_model_loader._parsing_adhoc_input_to_model_uri("Pytorch_Model", 1, "Production")
    assert model_uri == "models:/Pytorch_Model/1/Production"

def test_parse_model_artifact_uri(mlflow_model_loader):
    # Test when a model artifact URI is provided
    model_uri = mlflow_model_loader._parsing_adhoc_input_to_model_uri("http://example.com/models/Pytorch_Model/1")
    assert model_uri == "http://example.com/models/Pytorch_Model/1"

def test_invalid_args_length(mlflow_model_loader):
    # Test when an invalid number of args is provided
    with pytest.raises(ValueError):
        mlflow_model_loader._parsing_adhoc_input_to_model_uri("Pytorch_Model", 1, "Staging", "ExtraArg")

def test_invalid_first_arg_type(mlflow_model_loader):
    # Test when the first arg is not a string
    with pytest.raises(TypeError):
        mlflow_model_loader._parsing_adhoc_input_to_model_uri(123)

def test_invalid_stage_value(mlflow_model_loader):
    # Test when an invalid stage value is provided
    with pytest.raises(ValueError):
        mlflow_model_loader._parsing_adhoc_input_to_model_uri("Pytorch_Model", "InvalidStage")

