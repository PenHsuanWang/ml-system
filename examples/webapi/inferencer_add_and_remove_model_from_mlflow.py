import requests
import torch
import io

from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.ml_core.data_processor.data_processor import DataProcessorFactory

# Define the ML inference setup functions

def setup_ml_inferencer_mlflow_agent(mlflow_tracking_server: str):
    response = requests.post("http://localhost:8000/ml_inference_manager/set_mlflow_agent",
                             json={"mlflow_tracking_server_uri": mlflow_tracking_server})
    print(response)


def set_model_from_mlflow_request(model_info: dict):
    response = requests.post("http://localhost:8000/ml_inference_manager/set_model_from_mlflow_artifact_origin_flavor",
                             json={"model_info": model_info})
    print(response)


def get_all_model_in_serving_list():
    response = requests.get("http://localhost:8000/ml_inference_manager/get_list_all_model_in_serving")
    return response.content


def run_model_inference(model_name, data, device):
    response = requests.post("http://localhost:8000/ml_inference_manager/inference",
                             json={
                                 "model_name": model_name,
                                 "data": data.tolist(),
                                 "device": device
                             })
    print(response.content)


# Add Model from MLflow Artifact Function

def add_model_from_mlflow_artifact(model_info):
    set_model_from_mlflow_request(model_info=model_info)


# Remove Model Function

def remove_model(model_name):
    response = requests.post("http://localhost:8000/ml_inference_manager/remove_model_from_serving_list",
                             json={"model_name": model_name})
    print(response.content)


if __name__ == "__main__":
    setup_ml_inferencer_mlflow_agent(mlflow_tracking_server="http://localhost:5011")

    model_info = {
        "mlflow_model_name": "Pytorch_Model",
        "model_stage": "Staging",
        "serving_model_name": "Pytorch_Model_stage"
    }

    # Add a model from Mlflow artifact to the serving app
    add_model_from_mlflow_artifact(model_info)

    model_in_serving = get_all_model_in_serving_list()

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

    run_model_inference(
        model_name="Pytorch_Model_stage",
        data=test_data_numpy,
        device="mps"
    )

    # Remove a model from the serving app
    remove_model(model_name="Pytorch_Model_stage")
