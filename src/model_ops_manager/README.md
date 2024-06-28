# MLFlow Manager

Responsible for managing the API interaction with the MLflow server. Serves as an agent to access the MLflow server and works together with the project application. You need to define the MLflow server URL in the configuration file.

## Introduction to MLFlow Agent

This subpackage is designed as an interface to interact with the MLflow server. The module structure is as follows:

```text
└── mlflow_agent
    ├── __init__.py
    ├── client.py
    ├── configuration.py
    ├── mlflow_agent.py
    ├── model_downloader.py
    ├── registration.py
    ├── singleton_meta.py
    └── tracking.py
```

### Design Overview

This package uses several design patterns to ensure efficient and clear interaction with the MLflow server.

- **Singleton Pattern**: Ensures only one instance of the MLflow client is used across the application.
- **Facade Pattern**: `MLFlowAgent` acts as a facade, integrating various functionalities into a single interface.
- **Null Object Pattern**: `NullMLFlowAgent` provides a default implementation that does nothing, used when MLflow is not enabled.

### Features

- **MLFlow Client**: Implements a Singleton design pattern to manage MLFlow client instances efficiently, ensuring a consistent and thread-safe interaction with the MLFlow server across various parts of the application.
- **Model Agent**: Facilitates detailed interactions with the MLFlow server, allowing users to fetch model information, check model versions, and compose model URIs for easy retrieval and deployment.
- **Error Handling**: Rigorous error checking and handling to ensure stability and reliability, particularly in scenarios involving network communication and data integrity.
- **Configuration Management**: Provides mechanisms to configure and initialize the MLFlow tracking URI and other essential parameters dynamically, enhancing the adaptability of the agent to different environments.

## Module Descriptions

- **client.py**: Manages client interactions with the MLflow server, including initializing and managing session details.
- **configuration.py**: Handles configuration settings for the MLflow agent, reading from environment or config files.
- **mlflow_agent.py**: The central coordinator for the MLflow agent, integrating various modules into a cohesive interface.
- **model_downloader.py**: Facilitates the downloading of MLflow models, ensuring models are fetched and stored appropriately.
- **registration.py**: Provides functionalities for registering new models with the MLflow tracking server.
- **singleton_meta.py**: Implements a singleton metaclass that ensures a class has only one instance in any given Python process.
- **tracking.py**: Supports tracking ML experiments, recording metrics, parameters, and models.

## Features and Functionalities of `client.py` Module

### Singleton MLFlow Client

- **Single Instance Management**: Utilizes a Singleton pattern for the MLFlow client, ensuring only one instance is created and used across the application, enhancing consistency and reducing connection overhead.
- **Resource Optimization**: Reuses the same MLFlow client to optimize resource utilization, beneficial in resource-constrained environments.

### Initialization and Configuration

- **Pre-Initialization Checks**: Checks if the tracking URI is set before initializing the client, preventing runtime configuration errors.
- **Easy Setup**: Provides the `init_mlflow_client` method to abstract the complexity of setting up an MLFlow client, ensuring correct configuration before proceeding.

### Model Interaction Functions

- **Model URI Composition**: Offers the `compose_model_uri` method to easily retrieve URIs for model artifacts, handling the logic of determining the latest model version or using a specific version.
- **Version Checking and Validation**: Includes built-in checks to ensure that the requested model version matches the expected version, particularly important for specific stages.

### Error Handling and Feedback

- **Informative Error Messages**: Raises exceptions with clear messages for issues like unset tracking URIs or version mismatches, aiding quick resolution.
- **Robust Exception Management**: Manages exceptions related to MLFlow interactions, helping applications using the package maintain stability.

### Practical and User-Friendly API

- **High-Level Abstractions**: Methods like client initialization and model URI composition provide high-level abstractions over the MLFlow API, simplifying complex operations.
- **Integration Ease**: The module design facilitates easy integration with existing Python applications, offering clear entry points and adaptable methods.

## Usage Overview

To effectively utilize the MLFlow Manager package, you can engage with it at different levels depending on your project needs:

### Basic Usage

For general interactions with the MLflow server, such as tracking runs, logging parameters, and registering models:

```python
import mlflow_agent.client as mlflow_client

# Set the MLflow tracking URI
mlflow_client.MLFlowConfiguration.set_tracking_uri("http://localhost:5001")

# Create an MLFlowClient instance
client = mlflow_client.MLFlowClient()

# Start a new run and log parameters
with mlflow.start_run():
    mlflow.log_param("param", "value")
    mlflow.log_metric("metric", 123)

# Register a model
mlflow.pytorch.log_model(pytorch_model, "model_path")
```

### Advanced Model Interaction

For specific tasks related to model version management and URI composition:

```python
from mlflow_agent.client import MLFlowClientModelAgent

def main():
    # Initialize the MLFlow client with a specific tracking URI
    mlflow.set_tracking_uri("http://your_mlflow_server_uri")
    mlflow_client = MLFlowClientModelAgent()
    mlflow_client.init_mlflow_client()

    # Fetch the latest version of the model and compose its download URI
    model_name = "Example_Model"
    model_stage = "Production"
    model_version = mlflow_client.get_model_latest_version(model_name, model_stage)
    model_uri = mlflow_client.compose_model_uri(model_name, model_version)
    print("Model URI:", model_uri)

if __name__ == "__main__":
    main()
```

This structured approach helps users navigate the package's capabilities more efficiently and aligns with their specific needs, whether they are managing general MLflow tasks or focusing on detailed model interactions.

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
   - Use the methods provided by the client, such as `get_model_download_source_uri()` to perform specific MLflow operations.

```python
# Example usage:
mlflow.set_tracking_uri("http://localhost:5011")
MLFlowClientModelLoader.init_mlflow_client()
download_uri = MLFlowClientModelLoader.get_model_download_source_uri(model_name="Pytorch_Model", model_stage="Production")
```
