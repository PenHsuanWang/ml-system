import requests
import pandas as pd
import os

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000/ml_training_manager"

# Function to make POST requests to the API
def post_request(endpoint, json_data):
    response = requests.post(f"{BASE_URL}/{endpoint}", json=json_data)
    if response.status_code != 200:
        raise Exception(f"Request to {endpoint} failed: {response.json()}")
    return response.json()

# Function to make GET requests to the API
def get_request(endpoint):
    response = requests.get(f"http://localhost:8000/{endpoint}")
    if response.status_code != 200:
        raise Exception(f"Request to {endpoint} failed: {response.json()}")
    return response.json()

# Set MLflow settings
def example_set_mlflow_settings(mlflow_tracking_uri, mlflow_tracking_username, mlflow_tracking_password):
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password

# Set MLflow model name
def example_set_mlflow_model_name(model_name):
    response = requests.post(
        f"{BASE_URL}/set_mlflow_model_name",
        json={"model_name": model_name}
    )
    print(response.json())

# Set MLflow experiment name
def example_set_mlflow_experiment_name(experiment_name):
    response = requests.post(
        f"{BASE_URL}/set_mlflow_experiment_name",
        json={"experiment_name": experiment_name}
    )
    print(response.json())

# Set MLflow run name
def example_set_mlflow_run_name(run_name):
    response = requests.post(
        f"{BASE_URL}/set_mlflow_run_name",
        json={"run_name": run_name}
    )
    print(response.json())

# Run ML training
def example_run_ml_training(epochs):
    response = requests.post(
        f"{BASE_URL}/run_ml_training",
        json={"args": [], "kwargs": {"epochs": epochs}}
    )
    print(response.json())

# Get models
def example_get_models():
    response = get_request("ml_training_manager/get_model_list")
    print(response)

# Get trainers
def example_get_trainers():
    response = get_request("ml_training_manager/get_trainer_list")
    print(response)

def main():
    TRAINING_WINDOW_SIZE = 60
    TARGET_WINDOW_SIZE = 1
    MODEL_NAME_1 = "lstm_aapl_stock_price_prediction"
    MODEL_NAME_2 = "deep_lstm_aapl_stock_price_prediction"
    EXPERIMENT_NAME = "stock_price_prediction_experiment"
    RUN_NAME = "lstm_aapl_run"
    MLFLOW_TRACKING_URI = "http://localhost:5011"
    MLFLOW_TRACKING_USERNAME = "mlflow_pwang"
    MLFLOW_TRACKING_PASSWORD = "mlflow_pwang"

    example_set_mlflow_settings(MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD)

    # Step 1: Read local CSV file and convert to DataFrame
    csv_file_path = "/home/pwang/pwang-dev/ml-system/AAPL.csv"
    df = pd.read_csv(csv_file_path)
    dataframe_payload = {
        "data": df.to_dict(orient="records"),
        "columns": df.columns.tolist()
    }

    # Step 2: Initialize the data processor from the DataFrame
    init_data_processor_response = post_request("init_data_preprocessor_from_df", {
        "data_processor_type": "time_series",
        "dataframe": dataframe_payload,
        "kwargs": {
            "extract_column": ['Close', 'Volume'],
            "training_data_ratio": 0.6,
            "training_window_size": TRAINING_WINDOW_SIZE,
            "target_window_size": TARGET_WINDOW_SIZE
        }
    })
    print(init_data_processor_response)

    # Step 3: Initialize the first model
    init_model_response_1 = post_request("init_model", {
        "model_type": "lstm",
        "model_id": "model_1",
        "kwargs": {
            "input_size": 2,
            "hidden_size": 128,
            "output_size": 1
        }
    })
    print(init_model_response_1)

    # Step 4: Initialize the second model with deeper layers
    init_model_response_2 = post_request("init_model", {
        "model_type": "deep_lstm",
        "model_id": "model_2",
        "kwargs": {
            "input_size": 2,
            "hidden_size": 256,  # Increased hidden size for a deeper model
            "output_size": 1,
            "num_layers": 3  # Assuming LSTM model supports multiple layers
        }
    })
    print(init_model_response_2)

    # Step 5: Initialize the trainer for the first model
    init_trainer_response_1 = post_request("init_trainer", {
        "trainer_type": "torch_nn",
        "trainer_id": "trainer_1",
        "kwargs": {
            "loss_function": "mse",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "device": "cpu",
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "mlflow_tracking_username": MLFLOW_TRACKING_USERNAME,
            "mlflow_tracking_password": MLFLOW_TRACKING_PASSWORD
        }
    })
    print(init_trainer_response_1)

    # Step 6: Initialize the trainer for the second model
    init_trainer_response_2 = post_request("init_trainer", {
        "trainer_type": "torch_nn",
        "trainer_id": "trainer_2",
        "kwargs": {
            "loss_function": "mse",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "device": "cpu",
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "mlflow_tracking_username": MLFLOW_TRACKING_USERNAME,
            "mlflow_tracking_password": MLFLOW_TRACKING_PASSWORD
        }
    })
    print(init_trainer_response_2)

    # Step 7: Set MLflow model name for the first model
    example_set_mlflow_model_name(MODEL_NAME_1)

    # Step 8: Set MLflow model name for the second model
    example_set_mlflow_model_name(MODEL_NAME_2)

    # Step 9: Set MLflow experiment name
    example_set_mlflow_experiment_name(EXPERIMENT_NAME)

    # Step 10: Set MLflow run name
    example_set_mlflow_run_name(RUN_NAME)

    # Step 11: Run ML training for the first model
    example_run_ml_training(epochs=20)

    # Step 12: Run ML training for the second model
    example_run_ml_training(epochs=30)  # Assuming deeper model requires more epochs

    # Step 13: Get the list of existing models
    example_get_models()

    # Step 14: Get the list of existing trainers
    example_get_trainers()

if __name__ == "__main__":
    main()
