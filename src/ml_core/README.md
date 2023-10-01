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
