import pytest
from unittest.mock import Mock
from src.model_ops_manager.mlflow_agent.client import MLFlowClientModelAgent, MLFlowClientModelLoader

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
        assert MLFlowClientModelAgent.get_target_model_version("test_model", model_version=2) == 2

    def test_get_target_model_version_with_stage(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]
        assert MLFlowClientModelAgent.get_target_model_version("test_model", model_stage="Staging") == 1

    def test_get_target_model_version_invalid_stage(self):
        with pytest.raises(ValueError):
            MLFlowClientModelAgent.get_target_model_version("test_model", model_stage="Invalid")

    def test_get_model_latest_version(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]
        assert MLFlowClientModelAgent.get_model_latest_version("test_model") == 1

    def test_compose_model_uri_with_version(self):
        assert MLFlowClientModelAgent.compose_model_uri("test_model", model_version=1) == "models:/test_model/1"

    def test_compose_model_uri_with_stage(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 2}]
        assert MLFlowClientModelAgent.compose_model_uri("test_model", model_stage="Staging") == "models:/test_model/2"

class TestMLFlowClientModelLoader:

    def test_get_download_model_uri_with_version(self):
        mlflow_client_mock.get_model_version_download_uri.return_value = "http://example.com/model/1"
        assert MLFlowClientModelLoader.get_download_model_uri("test_model", model_version=1) == "http://example.com/model/1"

    def test_get_download_model_uri_with_stage(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 2}]
        mlflow_client_mock.get_model_version_download_uri.return_value = "http://example.com/model/2"
        assert MLFlowClientModelLoader.get_download_model_uri("test_model", model_stage="Staging") == "http://example.com/model/2"

    def test_load_model_with_model_artifact_uri(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]
        mlflow_client_mock.get_model_version_download_uri.return_value = "http://example.com/model/1"
        model = MLFlowClientModelLoader.load_model_as_pyfunc("http://example.com/model/1")
        assert model is not None

    def test_load_model_with_model_name_and_version(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]
        mlflow_client_mock.get_model_version_download_uri.return_value = "http://example.com/model/1"
        model = MLFlowClientModelLoader.load_model_as_pyfunc("test_model", model_version=1)
        assert model is not None

    def test_load_model_with_model_name_and_stage(self):
        mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]
        mlflow_client_mock.get_model_version_download_uri.return_value = "http://example.com/model/1"
        model = MLFlowClientModelLoader.load_model_as_pyfunc("test_model", model_stage="Staging")
        assert model is not None

    def test_load_model_invalid_args(self):
        with pytest.raises(ValueError):
            MLFlowClientModelLoader.load_model_as_pyfunc("test_model", 1, "Staging", "Invalid")

    def test_get_all_version_registered_model(self):
        mlflow_client_mock.get_registered_model.return_value = {"name": "test_model"}
        versions = MLFlowClientModelLoader.get_all_version_registered_model("test_model")
        assert isinstance(versions, list)