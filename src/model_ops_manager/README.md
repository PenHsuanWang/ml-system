# MLFlow Manager

Responsible for managing the API interaction with MLflow server. Serves as an agent to access the MLflow server and works together with the project application. You need to define the MLflow server URL in the configuration file.

## Introduction to MLFlow Agent

This subpackage is designed as an interface to interact with the MLflow server. The module structure is as follows:

```
└── mlflow_agent
    ├── __init__.py
    ├── client.py
    ├── configuration.py
    ├── mlflow_agent.py
    ├── registration.py
    ├── singleton_meta.py
    └── tracking.py
```

This package is designed using the Strategy Pattern. Several MLflow provided API functions are implemented in separate Python modules, also known as strategy classes. The `mlflow_agent.py` is the context class, and the implemented modules are the strategy classes. `mlflow_agent.py` serves as the single interface from the ML system side, while the implemented modules are responsible for the implementation of the MLflow API functions. The `mlflow_agent.py` is designed as a singleton class, and the implemented modules are also designed as singleton classes, with the singleton pattern implemented by the `singleton_meta.py` module.

## Function Implementation in Strategy Classes

### `configuration.py`

The `MLFlowConfiguration` class provides a method to set the MLflow tracking URI.

### `tracking.py`

The `MLFlowTracking` class offers methods for starting and ending tracking runs, as well as logging parameters and metrics.

### `registration.py`

The `MLFlowModelRegistry` class handles the registration of PyTorch models with MLflow.

### `client.py`

The `MLFlowClient` class in the `client.py` module interacts with MLflow's tracking server, allowing you to retrieve model versions and other related operations.

## Usage

To use the MLFlow Manager package, follow these steps:

1. Import the necessary classes and functions from the `mlflow_agent` subpackage into your project.
2. Set the MLflow tracking URI using `MLFlowConfiguration.set_tracking_uri()` to connect to the desired MLflow server.
3. Use the `MLFlowTracking` class to start and end tracking runs, as well as log parameters and metrics.
4. Utilize the `MLFlowModelRegistry` class to register PyTorch models with MLflow.
5. Employ the `MLFlowClient` class to interact with the MLflow tracking server, retrieve model versions, and perform other related tasks.

Here's an example of how to use the `MLFlowClient` class:

```python
import mlflow_agent.client as mlflow_client

# Set the MLflow tracking URI
mlflow_client.MLFlowConfiguration.set_tracking_uri("http://localhost:5001")

# Create an MLFlowClient instance
client = mlflow_client.MLFlowClient()

# Retrieve model information
client.get_model("Pytorch_Model", model_stage="Production")

# Load a model from the specified URI
model_uri = "models:/Pytorch_Model/Production"
model = mlflow.pytorch.load_model(model_uri)
```

By following these steps, you can effectively manage interactions with the MLflow server in your ML project using the MLFlow Manager package.


## API Reference

