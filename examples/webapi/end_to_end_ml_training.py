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

# Get MLflow models
def example_get_mlflow_models():
    response = get_request("mlflow/models")
    print(response)

# Get MLflow model details
def example_get_mlflow_model_details(model_name, version):
    response = get_request(f"mlflow/models/details/{model_name}/{version}")
    print(response)

def main():
    TRAINING_WINDOW_SIZE = 60
    TARGET_WINDOW_SIZE = 1
    MODEL_NAME = "lstm_aapl_stock_price_prediction"
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

    # Step 3: Initialize the model
    init_model_response = post_request("init_model", {
        "model_type": "lstm",
        "model_id": "unique_model_id",  # Ensure model_id is included
        "kwargs": {
            "input_size": 2,
            "hidden_size": 128,
            "output_size": 1
        }
    })
    print(init_model_response)

    # Step 4: Initialize the trainer
    init_trainer_response = post_request("init_trainer", {
        "trainer_type": "torch_nn",
        "trainer_id": "unique_trainer_id",  # Ensure trainer_id is included
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
    print(init_trainer_response)

    # Step 5: Set MLflow model name
    example_set_mlflow_model_name(MODEL_NAME)

    # Step 6: Set MLflow experiment name
    example_set_mlflow_experiment_name(EXPERIMENT_NAME)

    # Step 7: Set MLflow run name
    example_set_mlflow_run_name(RUN_NAME)

    # Step 8: Run ML training
    example_run_ml_training(epochs=20)

    # Step 9: Get MLflow models
    example_get_mlflow_models()

    # Step 10: Get MLflow model details
    example_get_mlflow_model_details(MODEL_NAME, 3)

if __name__ == "__main__":
    main()
