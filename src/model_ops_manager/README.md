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

### Features

- **MLFlow Client**: Implements a Singleton design pattern to manage MLFlow client instances efficiently, ensuring a consistent and thread-safe interaction with the MLFlow server across various parts of the application.
- **Model Agent**: Facilitates detailed interactions with the MLFlow server, allowing users to fetch model information, check model versions, and compose model URIs for easy retrieval and deployment.
- **Error Handling**: Rigorous error checking and handling to ensure stability and reliability, particularly in scenarios involving network communication and data integrity.
- **Configuration Management**: Provides mechanisms to configure and initialize the MLFlow tracking URI and other essential parameters dynamically, enhancing the adaptability of the agent to different environments.

### Features and Functionalities of `client.py` Module

**1. Singleton MLFlow Client**
- **Single Instance Management**: Utilizes a Singleton pattern for the MLFlow client, ensuring only one instance is created and used across the application, enhancing consistency and reducing connection overhead.
- **Resource Optimization**: Reuses the same MLFlow client to optimize resource utilization, beneficial in resource-constrained environments.

**2. Initialization and Configuration**
- **Pre-Initialization Checks**: Checks if the tracking URI is set before initializing the client, preventing runtime configuration errors.
- **Easy Setup**: Provides the `init_mlflow_client` method to abstract the complexity of setting up an MLFlow client, ensuring correct configuration before proceeding.

**3. Model Interaction Functions**
- **Model URI Composition**: Offers the `compose_model_uri` method to easily retrieve URIs for model artifacts, handling the logic of determining the latest model version or using a specific version.
- **Version Checking and Validation**: Includes built-in checks to ensure that the requested model version matches the expected version, particularly important for specific stages.

**4. Error Handling and Feedback**
- **Informative Error Messages**: Raises exceptions with clear messages for issues like unset tracking URIs or version mismatches, aiding quick resolution.
- **Robust Exception Management**: Manages exceptions related to MLFlow interactions, helping applications using the package maintain stability.

**5. Practical and User-Friendly API**
- **High-Level Abstractions**: Methods like client initialization and model URI composition provide high-level abstractions over the MLFlow API, simplifying complex operations.
- **Integration Ease**: The module design facilitates easy integration with existing Python applications, offering clear entry points and adaptable methods.

### Example Python Script for Usage Demonstration

```python
# Import necessary classes from your package
from your_package_name import MLFlowClientModelAgent

def main():
    # Set the MLFlow tracking URI
    mlflow_tracking_uri = "http://your_mlflow_server_uri"
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Initialize the MLFlow client
    mlflow_client = MLFlowClientModelAgent()
    mlflow_client.init_mlflow_client()

    # Specify the model name and the stage you are interested in
    model_name = "Example_Model"
    model_stage = "Production"  # Can be "None", "Staging", "Production", or "Archived"

    # Fetch the latest version of the model at the specified stage
    try:
        model_version = mlflow_client.get_model_latest_version(model_name, model_stage)
        print(f"Latest version for {model_name} at stage {model_stage} is: {model_version}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Compose the model URI for the latest version
    try:
        model_uri = mlflow_client.compose_model_uri(model_name, model_version)
        print(f"Model URI for version {model_version} of {model_name}: {model_uri}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

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
client.get_model_download_source_uri("Pytorch_Model", model_stage="Production")

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
