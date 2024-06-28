from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import pandas as pd

# Definition of FastAPI router
router = APIRouter()


# Define the Pydantic model for the request body
class TrainingDataRequest(BaseModel):
    data: str = Field(..., description="JSON string of the DataFrame to be used for training.")


@router.post("/receive")
async def receive_training_data(request: TrainingDataRequest) -> dict:
    """
    Receive training data as a JSON string and convert it into a pandas DataFrame for further processing.

    This endpoint is designed to accept preprocessed training data in JSON format. The received data is then
    converted into a pandas DataFrame, which can be utilized for training machine learning models. The endpoint
    assumes the JSON string is in 'split' format. After conversion, a placeholder function represents the point
    where the DataFrame would be passed to the ML training pipeline. In case of any conversion errors, an HTTP
    exception is raised with appropriate details.

    :param request: A request object containing a JSON string of the DataFrame.
    :return: A dictionary with a message indicating successful receipt and processing of the data.
    :raises HTTPException: If there is an error in processing the training data.
    """
    try:
        # Convert the JSON to a pandas DataFrame
        # Assume that the JSON string in request.data is in 'split' format
        training_data = pd.read_json(request.data, orient='split')

        # Placeholder: Replace with actual logic to queue or start training process
        # This is where you would typically hand off the DataFrame to your ML training pipeline
        print("Received DataFrame for training:", training_data.head())

        # After processing the DataFrame, send a success message
        return {"message": "Received training data successfully."}
    except ValueError as e:
        # If an error occurs, raise an HTTPException with status code 400
        raise HTTPException(status_code=400, detail=f"Error processing training data: {e}")
