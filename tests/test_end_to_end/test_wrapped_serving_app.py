from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.ml_core.data_processor.data_processor import DataProcessorFactory
import src.webapp.data_io_serving_app
import src.store.data_processor_store
import src.webapp.ml_training_serving_app
import src.webapp.ml_inference_serving_app

from torch.utils.data import DataLoader

import torch


def test_complete_ml_training_process_with_wrapped_serving_app():

    """
    applying the wrapped serving app to complete the ml training process
    test all implement web application.
    :return:
    """

    # using data_io_serving_app to prepare data fetcher

    data_fetcher_app = src.webapp.data_io_serving_app.get_app()

    # add a data fetcher, user need to create the data fetcher first on their own, and add ot to the data fetcher list
    yfinance_data_fetcher = DataFetcherFactory.create_data_fetcher("yfinance")
    data_fetcher_app.data_fetcher["yfinance"] = yfinance_data_fetcher

    # data_fetcher_app.data_fetcher["yfinance"].fetch_from_source(stock_id="AAPL", start_date="2020-01-01", end_date="2023-01-01")

    # using ml_training_serving_app to prepare data processor
    # when the ml_training_app is initialized, the class level object data_io_app is initialized as well
    # this is singleton pattern, the data_io_app is shared between ml_training_app and ml_inference_app
    ml_training_app = src.webapp.ml_training_serving_app.get_app()
    ml_training_app.set_data_fetcher(data_fetcher_name="yfinance")
    ml_training_app.fetcher_data([], {"stock_id":"AAPL", "start_date":"2020-01-01", "end_date":"2023-01-01"})
    ml_training_app.init_data_processor(data_processor_type="time_series", extract_column=['Close', 'Volume'], training_data_ratio=0.9,)
    ml_training_app.init_model(model_type="lstm", input_size=2, hidden_size=128, output_size=1)
    ml_training_app.init_trainer(trainer_type="torch_nn", learning_rate=0.01, loss_function="mse", optimizer="adam", device="mps")
    # ml_training_app.run_ml_training(epochs=50)

    model = ml_training_app.get_model()

    ml_inferencer = src.webapp.ml_inference_serving_app.get_app()
    ml_inferencer.set_model_to_serving_list(model_name="pytorch_lstm_aapl", model=model)

    data_fetcher_app.data_fetcher["yfinance"].fetch_from_source(stock_id="AAPL", start_date="2023-02-01", end_date="2023-06-30")
    df_test = data_fetcher_app.data_fetcher["yfinance"].get_as_dataframe()

    data_processor_for_test = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df_test,
        extract_column=['Close', 'Volume'],
        training_data_ratio=0.9
    )

    data_processor_for_test.preprocess_data()

    test_data_numpy = data_processor_for_test.get_training_data_x()

    output_predict = ml_inferencer.inference(model_name="pytorch_lstm_aapl", data_input=test_data_numpy)

    print(output_predict)



    # get data processor from data processor manager
    # data_processor = src.store.data_processor_manager.get_app().get_data_processor(data_processor_id="pytorch_lstm_aapl")
    # data_for_test = data_processor.inverse_testing_scaler(data=data_for_test)
    # print(data_for_test)



    # ml_inferencer.inference(model_name="pytorch_lstm_aapl", data_input)






