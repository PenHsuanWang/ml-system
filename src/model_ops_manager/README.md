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
client.get_download_model_uri("Pytorch_Model", model_stage="Production")

# Load a model from the specified URI
model_uri = "models:/Pytorch_Model/Production"
model = mlflow.pytorch.load_model(model_uri)
```

By following these steps, you can effectively manage interactions with the MLflow server in your ML project using the MLFlow Manager package.


## API Reference

### `client.py`

#### Design Overview

The MLFlow Client design is structured into three main components:

1. **MLFlowClient (Abstract Base Class):**
   - `MLFlowClient` serves as the abstract base class (ABC) for the MLFlow Client hierarchy.
   - It defines a common interface for interacting with the MLflow Tracking Server.
   - It ensures a Singleton pattern for the `mlflow_client`, guaranteeing that all parts of the application share the same instance.

2. **MLFlowClientModelAgent (Composite Class):**
   - `MLFlowClientModelAgent` extends the `MLFlowClient`.
   - It represents a more specialized client focused on model-related operations.
   - This class inherits the common interface defined in `MLFlowClient` and can provide its own specialized methods for model management.

3. **MLFlowClientModelLoader (Concrete Class):**
   - `MLFlowClientModelLoader` is a concrete class that further extends `MLFlowClientModelAgent`.
   - It represents a specific use case of the model-related operations, specifically for downloading models from MLflow.
   - This class inherits both the common interface defined in `MLFlowClient` and the specialized methods provided by `MLFlowClientModelAgent`. It is dedicated to the task of model downloading.

#### Design Purpose

The design aims to achieve the following goals:

- **Consistency:** By defining a common interface in the `MLFlowClient`, it ensures that all parts of the application interact with the MLflow Tracking Server consistently.

- **Reusability:** The design allows for the reuse of the same MLFlow client instance (`mlflow_client`) throughout the application, reducing resource overhead and promoting efficient usage.

- **Specialization:** The hierarchy of classes allows for specialization. While `MLFlowClient` offers a generic interface, `MLFlowClientModelAgent` specializes in model-related operations, and `MLFlowClientModelLoader` narrows its focus to downloading models.

- **Flexibility:** The design can be extended to include more specialized clients for different aspects of MLflow, all inheriting from the common `MLFlowClient`.

#### How to Use

To use the MLFlow Client design in your application, follow these steps:

1. **Initialize MLflow Tracking URI:**
   - Before using the MLflow Client, set the tracking URI using `mlflow.set_tracking_uri()` to point to your MLflow Tracking Server.

2. **Create and Initialize the Client:**
   - Create an instance of `MLFlowClientModelLoader` or other specialized clients if needed.
   - Initialize the MLflow client instance using `init_mlflow_client()`.

3. **Interact with MLflow:**
   - Use the methods provided by the client, such as `get_download_model_uri()` to perform specific MLflow operations.

```python
# Example usage:
mlflow.set_tracking_uri("http://localhost:5011")
MLFlowClientModelLoader.init_mlflow_client()
download_uri = MLFlowClientModelLoader.get_download_model_uri(model_name="Pytorch_Model", model_stage="Production")
