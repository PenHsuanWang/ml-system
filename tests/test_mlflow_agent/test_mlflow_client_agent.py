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


class TestMLFlowClientModelLoader:

    @classmethod
    def setup_class(cls):
        # Setup any necessary test fixtures or configurations
        cls.mlflow_client_mock = Mock()
        cls.mlflow_client_mock.get_latest_versions.return_value = [{"version": 1}]  # Mock the return value

    @classmethod
    def teardown_class(cls):
        # Teardown any resources after testing
        pass

    def test_valid_model_name(self):
        # Test when only model_name is provided
        model_name = "MyModel"
        model_uri = MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri(model_name)
        assert model_uri == "expected_model_uri"

    def test_valid_model_name_and_version(self):
        # Test when model_name and model_version are provided
        model_name = "MyModel"
        model_version = 2
        model_uri = MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri(model_name, model_version)
        assert model_uri == "expected_model_uri"

    def test_valid_model_name_and_stage(self):
        # Test when model_name and model_stage are provided
        model_name = "MyModel"
        model_stage = "Staging"
        model_uri = MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri(model_name, model_stage)
        assert model_uri == "expected_model_uri"

    def test_valid_model_name_version_and_stage(self):
        # Test when model_name, model_version, and model_stage are provided
        model_name = "MyModel"
        model_version = 2
        model_stage = "Production"
        model_uri = MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri(model_name, model_version, model_stage)
        assert model_uri == "expected_model_uri"

    def test_invalid_args_length(self):
        # Test when more than 3 arguments are provided
        with pytest.raises(ValueError):
            MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri("arg1", "arg2", "arg3", "arg4")

    def test_invalid_first_arg_type(self):
        # Test when the first argument is not a string
        with pytest.raises(TypeError):
            MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri(123)

    def test_invalid_args_combination(self):
        # Test when an invalid combination of arguments is provided
        with pytest.raises(ValueError):
            MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri("model_name", 2, "Staging")

    def test_invalid_model_stage(self):
        # Test when an invalid model_stage is provided
        with pytest.raises(ValueError):
            MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri("model_name", "Invalid_Stage")

    def test_valid_model_artifact_uri(self):
        # Test when a valid model artifact URI is provided
        model_artifact_uri = "http://example.com/model/artifact"
        model_uri = MLFlowClientModelLoader._adhoc_input_to_model_download_source_uri(model_artifact_uri)
        assert model_uri == model_artifact_uri

