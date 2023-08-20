# Machine Learning Process Core

This repository offers a collection of modules designed to streamline the model training process. The main steps involved in this process are:

1. Fetching raw data from the data source.
2. Preprocessing this raw data.
3. Setting up the desired model.
4. Initializing and executing the training process.

While each module can be initialized independently, the actual ML training process requires adhering to the defined workflow sequence.


* Fetcher get the raw DataFrame from the source and provide to the Preprocessor.
* Preprocessor preprocess the raw DataFrame and get the converted data for training. In torch nn model for example. The
  preprocessor will convert the DataFrame to the `torch.Tensor`.
* Prepare the Model.
* Initialization the Trainer with the model and the converted training data.


## DataFetcher

The `DataFetcher` serves as an interface to retrieve raw data from various data sources. 

Located in another package named `data_io`, this module acts as an adaptor, seamlessly preparing data fetching operations from different sources like Yahoo Finance, local databases, etc.

```python
fetcher = DataFetcherFactory.create_data_fetcher("yfinance")
fetcher.fetch_from_source(stock_id="AAPL", start_date="2022-01-01", end_date="2023-01-01")
apple_raw_df = fetcher.get_as_dataframe()
```

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
