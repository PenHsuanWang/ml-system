# Machine Learning System

## Introduction

This system is designed to operate a robust machine learning service, performing both model training and inference serving. Built on top of the FastAPI framework, the service provides RESTful API endpoints for clients to manage machine learning jobs. This README will guide developers and users through the system's components and functionalities.

### High Level Machine Learning workflow

The high-level workflow of the machine learning includes the data fetching, data processing, model training, and model serving. The system is designed to be modular and extensible, allowing for easy integration of new machine learning models and datasets. The system is also designed to be scalable, allowing for multiple machine learning jobs to be run concurrently.

![ml_system_modular_design](https://hackmd.io/_uploads/SkKzUocgR.png)

The data processing and model design components are independent and can be replaced with different implementations. The system is designed to be modular and extensible, allowing for easy integration of new machine learning models and datasets. The system is also designed to be scalable, allowing for multiple machine learning jobs to be run concurrently.

This system is designed to be modular and extensible, facilitating easy integration of new machine learning models and datasets. The system is also designed to be scalable, allowing for multiple machine learning jobs to be run concurrently. Each part of the machine learning workflow can be reconfigured and replaced with a different implementation.

### High Level Module Design

The high-level module design of this project can be divided into Basic Components, Component Manager, and User Interface.

#### The basic components of the system include:
* DataFetcher : responsible for fetching data from the source. Pass the raw data to the next stage.
* DataAnalyzer: for implementing data analysis and feature engineering (EDA).
* Model: for implementing the machine learning model. Based on the requirement, the model can be implemented with different framework.
* Trainer: for implementing the training loop. The trainer will take the model and data as input and output the trained model.

#### The components' manager:
The basic components will be created based on requirements and composed by the designer, which could be complex in certain cases. Thus the component manager responsible for organizing created basic component and compose to a complete Machine Learning Entity.
Multiple components, such as data analyzers, can be created as per the specific training requirements. Put all created data analyzer components into a pool, and the manager will take care of object access from the pools and apply them to the pipeline when needed.

#### The User Interface:
The user interface is the entry point of the system. It will interact with the components manager and provide the service to the user. The user interface will provide the RESTful API for the user to interact with the system. The user interface will also provide the web interface for the user to interact with the system.

### System Overview

This Machine Learning System run on top of FastAPI backend, the new application can be added with routes and endpoints. The system is designed to be modular and extensible, allowing for easy integration of new machine learning models and datasets. The system is also designed to be scalable, allowing for multiple machine learning jobs to be run concurrently.

The entry point of server `src/server.py` to add the routes and endpoints. to add a new application, just add the router to the main application.
```python=
app = FastAPI()

# Include the router in the main FastAPI app
app.include_router(data_io_serving_app_router.router)
app.include_router(ml_training_serving_app_router.router)
app.include_router(stock_price_analyzer_serving_app_route.router)
app.include_router(mlflow_model_download_serving_app_route.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

the routes and application is implemented in `src/webapp`
```text=
src/webapp
├── data_io_serving_app.py
├── data_io_serving_app_router.py
├── ml_training_serving_app.py
├── ml_training_serving_app_router.py
...
```

The machine learning job consists of several distinct processes:
implement the modules for each process and integrate them into ml service application. each stage just need to focus on the previous stage's output and the next stage's input. Implement the adapter patter for each stage to make the system more flexible and extensible.

1. **Data Fetching (`data_io/data_fetcher`):**
   - Responsible for extracting data from the original data source.
   - Ensure data retrieval, transformation, and storage are handled efficiently.
   - Convert the fetched data into pandas Dataframe for next stage of processing.

2. **Data Processing (`ml_core/data_processor`):**
   - This module is dedicated to data cleaning, preprocessing, and feature engineering from fetched data in dataframe format.
   - Decide on the preprocessing steps, such as handling missing data, scaling, encoding categorical variables, etc.
   - Make this part of the pipeline modular and reusable.
   - Converted the processed data into numpy array for next stage of training.

3. **Data Loader (`ml_core/data_loader`):**
   - Integrate the processed data to the training loop.
   - Consider utilizing libraries like PyTorch's DataLoader or other platform's Data API.
   - Implement data augmentation if it's relevant to your problem.

4. **Model Implementation (`ml_core/model`):**
   - Define the architecture of your machine learning model.
   - Choose a framework (e.g., PyTorch) and create a modular model class.
   - Ensure your model can switch seamlessly between training and inference modes.
   - Implement mechanisms for saving and loading trained models for later use.

5. **Training Loop (`ml_core/trainer`):**
   - Design a training loop that iterates through batches of data from data loader.
   - Implement key training components like loss calculation, gradient computation, and model parameter updates.
   - Consider incorporating metrics monitoring and early stopping if needed.

### Model Artifacts
This System using MLFlow as the model artifact manage, user can set up the MLFlow Tracking URL to then setup the MLFlow Agent to interact with the MLFlow Tracking Server. The MLFlow Tracking Server will store the model artifacts and the model metadata. The MLFlow Tracking Server will also provide the RESTful API for the MLFlow Client to interact with the MLFlow Tracking Server. This project implement MLFlow Agent under `src/model_ops_manager`

### System Workflow

The high-level workflow of the system can be summarized as follows:

1. Data is fetched from the source using the `data_io/data_fetcher` module.
2. The fetched data is then processed and prepared for training with the `ml_core/data_processor`.
3. A suitable data loader is created using `ml_core/data_loader` to handle data batching.
4. The machine learning model is defined and implemented using the `ml_core/model` module.
5. The training loop is executed by the `ml_core/trainer` module.
6. The system provides mechanisms for saving and loading trained models.

### Getting Started

To use or contribute to this system, follow the steps below:
