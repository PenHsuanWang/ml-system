# Machine Learning Process Core

## Introduction to the Machine Learning Pipeline

Training an AI/ML model involves a series of stages, each critical to the development of a production-ready solution. These stages typically include data fetching, data engineering, data transformation, model preparation (e.g., designing a PyTorch neural network), executing the model training loop, model evaluation, and deployment. While these steps are essential, they can become complex and unwieldy when incorporated into a single Jupyter notebook or script.

Moreover, as machine learning projects evolve and mature, maintaining and extending these operations can pose significant challenges. Treating a machine learning project as a software product requires a structured and scalable approach.

Recognizing these challenges, our project features the Machine Learning Process Core module, which adopts Object-Oriented Programming (OOP) principles. This module reimagines each stage of the machine learning pipeline as a distinct, reusable operation. This approach simplifies the development process, enhances maintainability, and facilitates future iterations of the machine learning project. By encapsulating each stage as a discrete unit, data scientists can focus on refining individual components, enabling seamless integration into the broader machine learning workflow.

## ML processes components modules

The ML processes components modules considered the those stage afterword data fetching, that say, the module in `ml_core` package only responsible for operating collected raw data.

This package offers a collection of modules designed to streamline the model training process. The main steps involved in this process are:

1. Processing this raw data.
2. Setting up the desired model.
3. Initializing and executing the training process.

While each module can be initialized independently, the actual ML training process requires adhering to the defined workflow sequence.

![image](https://i.imgur.com/lO96Qfi.png)

* Processor preprocess the raw DataFrame and get the converted data for training. In torch nn model for example. The preprocessor will convert the DataFrame to the `np.ndarray`.
* Prepare the Model.
* Prepare the data loader for batching and feeding data to the training loop.
* Initialization the Trainer with the model and the converted training data.


## Introduction to Components Modules

---

#### 2.1 Data Processor

In the machine learning pipeline, the data_processor component plays a pivotal role as a converter responsible for transforming data obtained from the initial stages of data collection and basic cleaning. Typically, this data arrives in the form of a pandas DataFrame, which is a structured and tabular representation of the dataset.

The primary function of the data_processor is to convert this DataFrame into a more suitable format for machine learning tasks, which often involves transforming it into a numpy ndarray. This conversion is essential to ensure compatibility with various machine learning algorithms and techniques.

**API introduction**

**TimeSeriesDataProcessor**

Within our project, we provide a specialized data_processor known as `TimeSeriesDataProcessor`. This component is designed specifically for handling time series data, a common and crucial data format in many machine learning applications. The `TimeSeriesDataProcessor` is engineered to perform the following key tasks:

1. **Sliding Window Transformation:** It converts the original time series data in the DataFrame into a sequence of sliding windows, effectively segmenting the data into more manageable and informative chunks. This segmentation is particularly useful for capturing temporal patterns and dependencies in the data.

2. **Feature Scaling:** To ensure that all features within the dataset have a consistent impact on the machine learning model, the `TimeSeriesDataProcessor` also performs feature scaling. This involves standardizing or normalizing the data, bringing it to a common scale.

By utilizing the `TimeSeriesDataProcessor`, data scientists can streamline the process of preparing time series data for machine learning tasks. It simplifies the conversion from a pandas DataFrame to a structured numpy ndarray, and by incorporating sliding window transformations and feature scaling, it empowers data scientists to harness the temporal dynamics within their data effectively.


#### 2.2 Data Loader: Bridging Data and Model

In this section, we'll explore the Data Loader, a crucial part of our project that helps connect the data processed by the Data Processor to our machine learning model. Think of it as the middleman between the data and the model, ensuring everything is in the right format for our model to understand.

**Converting Data Types**

One of the Data Loader's primary jobs is to convert the processed data, which is initially in the form of a numpy ndarray, into a format that our model can work with efficiently. For example, if we're using PyTorch, our model needs the data in torch.Tensor format. The Data Loader handles this conversion seamlessly, making sure the data aligns perfectly with what our model expects.

**Data Batching and Shuffling**

But the Data Loader doesn't stop at conversion; it has another critical function. During the model's training phase, it's often helpful to process data in batches and shuffle it to improve learning. The Data Loader can take care of this too. It splits the data into manageable chunks (batches) and shuffles it, which can significantly enhance how our model learns from the data.

**Extensibility with OCP**

One neat thing about our DataLoader module is that it's designed with an eye on the future. It follows the Open-Closed Principle (OCP), which means it's open for extension. This allows us to add new features and capabilities to the Data Loader down the road, making it adaptable to our project's evolving needs.


#### 2.3 Model Trainer: Executing the Machine Learning Training Loop

In this section, we dive into the Model Trainer, a critical component responsible for executing the machine learning training loop. This code snippet handles the training of our model using the data processed and prepared by the previous stages, specifically the Data Processor and Data Loader. 

**Training the Model**

The core of this implementation lies in training the machine learning model. It provides a robust and configurable training loop, making it easy to set parameters and fine-tune the model. With this Model Trainer, you can effortlessly customize various aspects of the training process to ensure your model learns effectively from the data.

**Preparing for Deployment**

As the training loop progresses, the Model Trainer records essential metrics and performance data. These metrics are stored in MLflow, providing a comprehensive record of how the model performs during training. This step ensures that you have a detailed overview of your model's learning process, which is invaluable for further analysis and optimization.

**Continuous Improvement**

One notable feature of our Model Trainer is its adaptability. It allows for continuous improvement and model updates as new data becomes available or as the project evolves. This adaptability ensures that your model remains effective and accurate over time, aligning perfectly with the ever-changing nature of machine learning projects.

In summary, the Model Trainer executes the heart of our machine learning project, handling the training loop with ease and flexibility. It not only prepares the model for deployment but also enables continuous refinement and optimization, making it an indispensable component in our machine learning pipeline.


---

## Data Preprocessor

The `Data Preprocessor` takes the raw data frame from the Fetcher and transforms it, making it suitable for training. For instance, if you're using a PyTorch NN model, the preprocessor will convert the DataFrame into `torch.Tensor`.

```python
data_processor = TimeSeriesDataProcessor(
    input_data=apple_raw_df,
    extract_column=['Close', 'Volume'],
    training_data_ratio=0.6,
    training_window_size=TRAINING_WINDOW_SIZE,
    target_window_size=TARGET_WINDOW_SIZE
)
data_processor.preprocess_data()
```

## Model Setup

This section allows you to define and prepare the model for your ML tasks. Depending on the nature of your data and task, you can choose from various models, including LSTM, CNN, etc.

```python
model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
    "lstm",
    input_size=2,
    hidden_size=128,
    output_size=1
)
```

## Trainer Initialization and Execution

Finally, the `Trainer` is initialized with the chosen model and the converted training data. Once set, the training can commence.

```python
trainer = TrainerFactory.create_trainer(
    "torch_nn",
    model=model,
    criterion=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    device=torch.device('mps'),
    training_data=train_tensor,
    training_labels=train_target_tensor
)
trainer.run_training_loop(epochs=300)
```

### Model Evaluation

Post-training, you can evaluate the model's performance and make predictions.

```python
model.eval()

test_tensor = test_tensor.to(torch.device('mps'))
prediction = model(test_tensor).to('cpu').detach().numpy()

prediction_output = data_processor.inverse_testing_scaler(
    data=prediction,
    scaler_by_column_name='Close'
)

test_target = data_processor.inverse_testing_scaler(
    data=test_target_tensor.numpy(),
    scaler_by_column_name='Close'
)

# Compute the mean square error between prediction and test_target
mse = ((prediction_output - test_target) ** 2).mean(axis=0)
print(f"Mean square error: {mse}")
```


# Model Design

## Torch NN Model

The package `models.torch_nn_models` aims to provide a set of reusable PyTorch neural network models that can be easily integrated into larger machine learning pipelines. The design adopts an Object-Oriented Programming (OOP) approach to ensure modularity, reusability, and extensibility. 

## Package Structure

The package consists of the following Python files:

- `base_model.py`: Contains the abstract base class for all types of neural network models.
- `lstm_model.py`: Implements an LSTM-based neural network model.
- `model.py`: Includes a factory class for creating instances of different neural network models.

### Module Descriptions

#### `base_model.py`

This module defines an abstract base class (`BaseModel`) that inherits from `nn.Module`, PyTorch’s module for neural networks. The class defines the core methods that every neural network model should have:

- `forward`: An abstract method that must be overridden by subclasses to define the forward pass of the neural network.
- `get_model_hyper_parameters`: An abstract method that returns a dictionary of model hyper-parameters. This allows for greater introspection and can be used for tuning.

#### `lstm_model.py`

This module implements the LSTM-based neural network model (`LSTMModel`). The model includes:

- Two LSTM layers.
- A fully connected (linear) layer for output.

The `LSTMModel` class inherits from `BaseModel` and overrides the `forward` and `get_model_hyper_parameters` methods.

#### `model.py`

This module includes the factory class (`TorchNeuralNetworkModelFactory`) that takes a model type as a string and returns an instance of the corresponding neural network model. This allows for greater flexibility and easier integration into larger systems.

### Classes

#### BaseModel (Abstract Class)

- **Methods:**
  - `forward(self, x: torch.Tensor) -> torch.Tensor`: Abstract method to perform a forward pass through the neural network.
  - `get_model_hyper_parameters(self) -> dict`: Abstract method to get the hyper-parameters of the model.

#### LSTMModel (Concrete Class)

- **Attributes:**
  - `hidden_size`: The number of hidden units.
  - `lstm1`: The first LSTM layer.
  - `lstm2`: The second LSTM layer.
  - `fc`: The fully connected output layer.
  
- **Methods:**
  - `__init__(self, input_size, hidden_size, output_size)`: Constructor to initialize the LSTM model.
  - `forward(self, x: torch.Tensor) -> torch.Tensor`: Implements the forward pass.
  - `get_model_hyper_parameters(self) -> dict`: Retrieves the model’s hyper-parameters.

#### TorchNeuralNetworkModelFactory (Factory Class)

- **Methods:**
  - `create_torch_nn_model(model_type: str, **kwargs) -> BaseModel`: Creates and returns an instance of the specified neural network model.

### Exceptions

#### UnsupportedModelType

Custom exception raised when an unsupported model type is provided to the factory class.

## Future Scope

1. Add more types of neural network models.
2. Implement methods for model saving and loading.
3. Extend the base class to include common training and evaluation loops.
4. Enable multi-device support for models.

## Conclusion

The design provides a flexible and extensible architecture for PyTorch neural network models, making it easier to build, test, and deploy various types of neural networks.

---

This design document serves as an initial guideline for the package development and can be updated as the project evolves.
