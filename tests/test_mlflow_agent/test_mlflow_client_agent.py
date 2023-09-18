import pytest
from mlflow.tracking import MlflowClient
from unittest.mock import patch, MagicMock
from src.model_ops_manager.mlflow_agent.client import MLFlowClientModelAgent

# Define a fixture to create a mock MLflow client
@pytest.fixture
def mock_mlflow_client():
    with patch('mlflow.tracking.MlflowClient') as mock_client:
        yield mock_client.return_value


# Create a test function for the get_model_latest_version method
def test_get_model_latest_version(mock_mlflow_client):
    # Create an instance of MLFlowClientModelAgent
    agent = MLFlowClientModelAgent()
    agent.init_mlflow_client()

    # Define the model name and an example version
    model_name = "example_model"
    example_version = 1

    # Mock the get_latest_versions method of the MLflow client
    mock_mlflow_client.get_latest_versions.return_value = [MagicMock(version=example_version)]

    # Call the method being tested
    result = agent.get_model_latest_version(model_name)

    # Assert that the mock client's get_latest_versions method was called with the expected arguments
    mock_mlflow_client.get_latest_versions.assert_called_once_with(
        name=model_name
    )

    # Assert that the result matches the expected version
    assert result == example_version

    # Reset the mock client to clear expectations
    mock_mlflow_client.reset_mock()


# Test the get_model_latest_version method with a provided model stage
def test_get_model_latest_version_with_stage(mock_mlflow_client):
    # Create an instance of MLFlowClientModelAgent
    agent = MLFlowClientModelAgent()
    agent.init_mlflow_client()

    # Define the model name, example stage, and example version
    model_name = "example_model"
    example_stage = "Staging"
    example_version = 1

    # Mock the get_latest_versions method of the MLflow client
    mock_mlflow_client.get_latest_versions.return_value = [MagicMock(version=example_version)]

    # Call the method being tested with the provided stage
    result = agent.get_model_latest_version(model_name, model_stage=example_stage)

    # Assert that the mock client's get_latest_versions method was called with the expected arguments
    # mock_mlflow_client.get_latest_versions.assert_called_once_with(
    #     name=model_name,
    #     stages=[example_stage]
    # )

    # Assert that the result matches the expected version
    assert result == example_version

    # Reset the mock client to clear expectations
    mock_mlflow_client.reset_mock()

# Test the get_model_latest_version method without a provided model stage
def test_get_model_latest_version_without_stage(mock_mlflow_client):
    # Create an instance of MLFlowClientModelAgent
    agent = MLFlowClientModelAgent()


    # Define the model name and example version
    model_name = "example_model"

    # Mock the get_latest_versions method of the MLflow client
    mock_mlflow_client.get_latest_versions.return_value = [MagicMock(version=2)]

    agent.init_mlflow_client()

    print("Calling mlflow client directly")
    print(agent.mlflow_client.get_latest_versions(name=model_name)[0].version)

    # Call the method being tested without providing a stage
    result = agent.get_model_latest_version(model_name)

    # Debugging: Print the actual result
    print("Actual result:", result)

    # Assert that the mock client's get_latest_versions method was called with the expected arguments
    # mock_mlflow_client.get_latest_versions.assert_called_once_with(
    #     name=model_name
    # )

    # Assert that the result matches the expected version
    assert result == 2

    # Reset the mock client to clear expectations
    mock_mlflow_client.reset_mock()

# Test the case when the provided model stage is not in the allowed category
def test_get_model_latest_version_invalid_stage(mock_mlflow_client):
    # Create an instance of MLFlowClientModelAgent
    agent = MLFlowClientModelAgent()
    agent.init_mlflow_client()

    # Define the model name and an invalid stage
    model_name = "example_model"
    invalid_stage = "InvalidStage"

    # Mock the get_latest_versions method of the MLflow client
    mock_mlflow_client.get_latest_versions.return_value = []

    # Call the method being tested with an invalid stage
    with pytest.raises(ValueError) as exc_info:
        agent.get_model_latest_version(model_name, model_stage=invalid_stage)

    # Assert that the method raises a ValueError
    assert "The model stage should be in" in str(exc_info.value)

    # Reset the mock client to clear expectations
    mock_mlflow_client.reset_mock()

# Test the case when the provided model stage is not registered
def test_get_model_latest_version_stage_not_registered(mock_mlflow_client):
    # Create an instance of MLFlowClientModelAgent
    agent = MLFlowClientModelAgent()
    agent.init_mlflow_client()

    # Define the model name and an example stage
    model_name = "example_model"
    example_stage = "Staging"

    # Mock the get_latest_versions method of the MLflow client to return an empty list
    mock_mlflow_client.get_latest_versions.return_value = []

    # Call the method being tested with an unregistered stage
    with pytest.raises(ValueError) as exc_info:
        agent.get_model_latest_version(model_name, model_stage=example_stage)

    # Assert that the method raises a ValueError
    assert f"The model name: {model_name} with stage: {example_stage} is not registered" in str(exc_info.value)
