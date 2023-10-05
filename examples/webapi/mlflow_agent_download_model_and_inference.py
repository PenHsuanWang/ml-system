import requests
import torch
import io

from src.ml_core.models.torch_nn_models.lstm_model import LSTMModel
from src.data_io.data_fetcher.fetcher import DataFetcherFactory
from src.ml_core.data_processor.data_processor import DataProcessorFactory
from src.webapp import ml_inference_serving_app

ml_inference_app = ml_inference_serving_app.get_app()

# setup ml system environment and connect to mlflow tracking server
def setup_mlflow_agent(mlflow_tracking_server: str):
    # set mlflow tracking uri
    response = requests.post("http://localhost:8000/mlflow_agent/set_mlflow_tracking_uri/",
                             json={"tracking_uri": mlflow_tracking_server})
    print(response)
    # init mlflow downloader
    response = requests.get("http://localhost:8000/mlflow_agent/init_mlflow_downloader_client/")
    print(response)


def get_model_from_mlflow_agent():
    # get model from mlflow
    response = requests.post("http://localhost:8000/mlflow_agent/download_mlflow_original_model/",
                             json={
                                "model_info": {"model_name": "Pytorch_Model", "model_stage": "Production"}
                             })
    # get serialized model and recompose

    model_bytes = response.json()['model']
    model_bytes = model_bytes.encode('latin1')

    model = LSTMModel(
        input_size=2,
        hidden_size=128,
        output_size=1
    )
    model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=torch.device('cpu')))
    return model


if __name__ == "__main__":
    # ml_inferencer have to know the mlflow tracking server address and connect to it
    setup_mlflow_agent(mlflow_tracking_server="http://localhost:5011")
    # register model from mlflow artifact and serving in inferencer
    model = get_model_from_mlflow_agent()

    ml_inference_app.set_model_to_serving_list("Pytorch_Model", model)

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
    output_predict = ml_inference_app.inference(model_name="Pytorch_Model", data_input=test_data_numpy, device="cpu")
    # the model send back to clients by serializing/deserializing the model in memory, make device can not be mps?
    print(output_predict)

