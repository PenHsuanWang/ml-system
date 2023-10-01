from src.webapp import ml_inference_serving_app
from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.ml_core.data_processor.data_processor import DataProcessorFactory

ml_inference_app = ml_inference_serving_app.get_app()


def setup_model_inference_serving_app(mlflow_tracking_server: str):
    ml_inference_app.setup_model_inference_serving_app(
        mlflow_tracking_server=mlflow_tracking_server
    )


def inferencer_load_and_serving_model_from_mlflow_artifact(model_name: str, model_stage: str):
    ml_inference_app.load_original_model_from_mlflow_artifact(
        model_name=model_name,
        model_stage=model_stage
    )


if __name__ == "__main__":

    # ml_inferencer have to know the mlflow tracking server address and connect to it
    setup_model_inference_serving_app(mlflow_tracking_server="http://localhost:5011")
    # register model from mlflow artifact and serving in inferencer
    inferencer_load_and_serving_model_from_mlflow_artifact(model_name="Pytorch_Model", model_stage="Production")

    # create a data fetcher to fetch data from yfinance to do model inference test
    yfinance_data_fetcher = DataFetcherFactory.create_data_fetcher("yfinance")
    yfinance_data_fetcher.fetch_from_source(stock_id="AAPL", start_date="2023-02-01", end_date="2023-06-30")
    df_test = yfinance_data_fetcher.get_as_dataframe()

    # prepare the data processor to convert data fetcher return dataframe to numpy array
    data_processor_for_test = DataProcessorFactory.create_data_processor(
        data_processor_type="time_series",
        input_data=df_test,
        extract_column=['Close', 'Volume'],
        training_data_ratio=0.9
    )
    data_processor_for_test.preprocess_data()
    test_data_numpy = data_processor_for_test.get_training_data_x()  # get numpy array for model inference

    # pass the numpy array to model inference app to do inference, get the model inference result from model inference app
    output_predict = ml_inference_app.inference(model_name="Pytorch_Model", data_input=test_data_numpy, device="mps")
    print(output_predict)
