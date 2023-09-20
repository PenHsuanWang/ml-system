from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.ml_core.data_processor.time_series_data_processor import TimeSeriesDataProcessor
from src.ml_core.models.torch_nn_models.model import TorchNeuralNetworkModelFactory
from src.ml_core.trainer.trainer import TrainerFactory

import torch


def test_complete_ml_training_process():
    TRAINING_WINDOW_SIZE = 60
    TARGET_WINDOW_SIZE = 1

    # Load data from yahoo finance using data fetcher
    fetcher = DataFetcherFactory.create_data_fetcher("yfinance")
    fetcher.fetch_from_source(stock_id="AAPL", start_date="2022-01-01", end_date="2023-01-01")
    apple_raw_df = fetcher.get_as_dataframe()

    # Preprocess data
    data_processor = TimeSeriesDataProcessor(extract_column=['Close', 'Volume'], training_data_ratio=0.6,
                                             training_window_size=TRAINING_WINDOW_SIZE,
                                             target_window_size=TARGET_WINDOW_SIZE)

    # it is okey to set the input df later
    data_processor.set_input_df(apple_raw_df)

    try:
        data_processor.preprocess_data()
    except RuntimeError as re:
        print(re)
        assert False

    train_tensor = data_processor.get_training_data_x()
    train_target_tensor = data_processor.get_training_target_y()

    test_tensor = data_processor.get_testing_data_x()
    test_target_tensor = data_processor.get_testing_target_y()

    print(train_tensor.shape)
    print(train_target_tensor.shape)
    # print(test_tensor.shape)

    model = TorchNeuralNetworkModelFactory.create_torch_nn_model(
        "lstm",
        input_size=2,
        hidden_size=128,
        output_size=1
    )

    trainer = TrainerFactory.create_trainer(
        "torch_nn",
        # model=model,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        device=torch.device('mps'),
        # training_data=train_tensor,
        # training_labels=train_target_tensor
    )

    trainer.set_model(model)
    trainer.set_training_tensor(train_tensor, train_target_tensor)

    try:
        trainer.run_training_loop(epochs=300)
    except RuntimeError as re:
        print(re)
        assert False

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

    # Calculating mse by prediction and test_target
    mse = ((prediction_output - test_target) ** 2).mean(axis=0)
    print(f"Mean square error: {mse}")




