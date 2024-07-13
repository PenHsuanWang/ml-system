import requests
import pandas as pd
import os

# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"

# Function to make POST requests to the API
def post_request(endpoint, json_data):
    response = requests.post(f"{BASE_URL}/{endpoint}", json=json_data)
    if response.status_code != 200:
        raise Exception(f"Request to {endpoint} failed: {response.json()}")
    return response.json()

# Function to make GET requests to the API
def get_request(endpoint):
    response = requests.get(f"{BASE_URL}/{endpoint}")
    if response.status_code != 200:
        raise Exception(f"Request to {endpoint} failed: {response.json()}")
    return response.json()

# Function to make PUT requests to the API
def put_request(endpoint, json_data):
    response = requests.put(f"{BASE_URL}/{endpoint}", json=json_data)
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
    response = post_request(
        "ml_training_manager/set_mlflow_model_name",
        {"model_name": model_name}
    )
    print(response)

# Set MLflow experiment name
def example_set_mlflow_experiment_name(experiment_name):
    response = post_request(
        "ml_training_manager/set_mlflow_experiment_name",
        {"experiment_name": experiment_name}
    )
    print(response)

# Set MLflow run name
def example_set_mlflow_run_name(run_name):
    response = post_request(
        "ml_training_manager/set_mlflow_run_name",
        {"run_name": run_name}
    )
    print(response)

# Run ML training
def example_run_ml_training(epochs):
    try:
        response = post_request(
            "ml_training_manager/run_ml_training",
            {"args": [], "kwargs": {"epochs": epochs}}
        )
        print(response)
    except Exception as e:
        print(f"Failed to run ML training: {e}")

# Get MLflow models
def example_get_mlflow_models():
    response = get_request("ml_training_manager/list_models")
    print(response)

# Get MLflow model details
def example_get_mlflow_model_details(model_id):
    response = get_request(f"ml_training_manager/get_model/{model_id}")
    print("Model Details:")
    print(response)
    parameters = response.get("parameters", {})
    print("Model Parameters:")
    for param, value in parameters.items():
        print(f"  {param}: {value}")

def example_get_trainer(trainer_id):
    response = get_request(f"ml_training_manager/get_trainer/{trainer_id}")
    print("Trainer Details:")
    print(response)
    parameters = response.get("parameters", {})
    print("Trainer Parameters:")
    for param, value in parameters.items():
        print(f"  {param}: {value}")

def example_get_data_processor(data_processor_id):
    response = get_request(f"ml_training_manager/get_data_processor/{data_processor_id}")
    print("Data Processor Details:")
    print(response)
    parameters = response.get("parameters", {})
    print("Data Processor Parameters:")
    for param, value in parameters.items():
        print(f"  {param}: {value}")

# List trainers
def example_list_trainers():
    response = get_request("ml_training_manager/list_trainers")
    print(response)

# List data processors
def example_list_data_processors():
    response = get_request("ml_training_manager/list_data_processors")
    print(response)

# Update model
def example_update_model(model_id, new_params):
    response = put_request(
        f"ml_training_manager/update_model/{model_id}",
        {"params": new_params}
    )
    print(response)
    print("Updated model configuration:", response.get("updated_model"))

# Update trainer
def example_update_trainer(trainer_id, new_params):
    response = put_request(
        f"ml_training_manager/update_trainer/{trainer_id}",
        {"params": new_params}
    )
    print(response)
    print("Updated model configuration:", response.get("updated_trainer"))

# Update data processor
def example_update_data_processor(data_processor_id, new_params):
    response = put_request(
        f"ml_training_manager/update_data_processor/{data_processor_id}",
        {"params": new_params}
    )
    print(response)
    # Verify data processor shape after update
    try:
        data_processor_info = get_request(f"ml_training_manager/get_data_processor/{data_processor_id}")
        print(f"Updated data processor info: {data_processor_info}")
    except Exception as e:
        print(f"Failed to retrieve updated data processor: {e}")

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
    init_data_processor_response = post_request("ml_training_manager/init_data_processor_from_df", {
        "data_processor_id": "example_data_processor_id",
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

    # Step 3: Initialize the model with multiple LSTM layers
    init_model_response = post_request("ml_training_manager/init_model", {
        "model_type": "lstm",
        "model_id": "unique_model_id",
        "model_name": "unique_model_id",
        "kwargs": {
            "input_size": 2,
            "hidden_layer_sizes": [128, 64, 32],  # Specify multiple hidden layers
            "output_size": 1
        }
    })

    print(init_model_response)

    # Step 4: Initialize the trainer
    init_trainer_response = post_request("ml_training_manager/init_trainer", {
        "trainer_type": "torch_nn",
        "trainer_id": "unique_trainer_id",  # Ensure trainer_id is included
        "kwargs": {
            "loss_function": "mse",
            "optimizer": "adam",
            "learning_rate": "0.001",  # Ensure learning_rate is a string
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

    # Step 9: List models
    example_get_mlflow_models()

    # Step 10: List trainers
    example_list_trainers()

    # Step 11: List data processors
    example_list_data_processors()

    # Step 12: Get details of a specific model, trainer, and data processor
    example_get_mlflow_model_details("unique_model_id")
    example_get_trainer("unique_trainer_id")
    example_get_data_processor("example_data_processor_id")

    # Step 13: Update trainer
    example_update_trainer("unique_trainer_id", {"learning_rate": "0.002"})

    # Step 14: Update data processor
    example_update_data_processor("example_data_processor_id", {"new_param": "new_value"})

    # Step 15: List data processors again to confirm the update
    example_list_data_processors()

    # Step 16: Retrain using the updated trainer without re-initializing
    example_run_ml_training(epochs=10)

if __name__ == "__main__":
    main()
