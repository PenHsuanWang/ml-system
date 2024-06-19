"""
Developing the serving app to export model training setting to REST api endpoint
"""

import os

from fastapi import APIRouter, Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from src.webapp.ml_training_serving_app import get_app, MLTrainingServingApp

# Definition of FastAPI router
router = APIRouter()

# Define the pydantic model for request body

class SetDataFetcherBody(BaseModel):
    data_fetcher_name: str
    args: list
    kwargs: dict


class FetchDataFromSourceBody(BaseModel):
    args: list
    kwargs: dict


class DataFramePayload(BaseModel):
    data: List[Dict]
    columns: List[str]


class InitDataProcessorFromDFBody(BaseModel):
    data_processor_id: str
    data_processor_type: str
    dataframe: DataFramePayload
    kwargs: dict


class InitDataProcessorBody(BaseModel):
    data_processor_id: str
    data_processor_type: str
    args: list
    kwargs: dict


class InitModelBody(BaseModel):
    model_type: str
    model_id: str
    kwargs: dict


class InitTrainerBody(BaseModel):
    trainer_type: str
    trainer_id: str
    kwargs: dict


class RunMLTrainingBody(BaseModel):
    args: list
    kwargs: dict


class SetMLflowModelNameBody(BaseModel):
    model_name: str


class SetMLflowExperimentNameBody(BaseModel):
    experiment_name: str


class SetMLflowRunNameBody(BaseModel):
    run_name: str


class UpdateModelParams(BaseModel):
    params: Dict[str, Any]


class UpdateTrainerParams(BaseModel):
    params: Dict[str, Any]


class UpdateDataProcessorParams(BaseModel):
    params: Dict[str, Any]


# Define the REST api endpoint
@router.post("/ml_training_manager/set_data_fetcher")
def set_data_fetcher(
        request: SetDataFetcherBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Set data fetcher
    :param ml_trainer_app:
    :param request: SetDataFetcherBody
    :return: JSONResponse
    """
    data_fetcher_name = request.data_fetcher_name
    args = request.args
    kwargs = request.kwargs

    ml_trainer_app.set_data_fetcher(data_fetcher_name)

    return {"message": f"Data Fetcher named: {data_fetcher_name} set successfully"}


@router.post("/ml_training_manager/fetch_data_from_source")
def fetch_data_from_source(
        request: FetchDataFromSourceBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Fetch data from source
    :param ml_trainer_app:
    :param request: FetchDataFromSourceBody
    :return: JSONResponse
    """
    kwargs = request.kwargs

    ml_trainer_app.fetcher_data(
        request.args,
        request.kwargs
    )

    return {"message": f"Fetched data successfully"}


@router.post("/ml_training_manager/init_data_preprocessor_from_df")
def init_data_preprocessor_from_df(
        request: InitDataProcessorFromDFBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Init data preprocessor from a DataFrame
    :param ml_trainer_app:
    :param request: InitDataProcessorFromDFBody
    :return: JSONResponse
    """
    try:
        data_processor_id = request.data_processor_id
        data_processor_type = request.data_processor_type
        dataframe_json = request.dataframe.dict()
        kwargs = request.kwargs

        ml_trainer_app.init_data_processor_from_df(data_processor_id, data_processor_type, dataframe_json, **kwargs)

        return {"message": "Init data preprocessor from DataFrame successfully"}

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=422,
            content={"message": "Init data preprocessor from DataFrame failed"}
        )


@router.post("/ml_training_manager/init_data_preprocessor")
def init_data_preprocessor(
        request: InitDataProcessorBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Init data preprocessor
    :param ml_trainer_app:
    :param request: InitDataProcessorBody
    :return: JSONResponse
    """
    try:
        data_processor_id = request.data_processor_id
        data_processor_type = request.data_processor_type
        kwargs = request.kwargs

        ml_trainer_app.init_data_processor(data_processor_id, data_processor_type, **kwargs)

        return {"message": f"Init data preprocessor successfully"}

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=422,
            content={"message": "Init data preprocessor failed"}
        )


@router.post("/ml_training_manager/init_model")
def init_model(
        request: InitModelBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Init model
    :param ml_trainer_app:
    :param request: InitModelBody
    :return: JSONResponse
    """
    model_type = request.model_type
    model_id = request.model_id
    kwargs = request.kwargs

    if not ml_trainer_app.init_model(model_type, model_id, **kwargs):
        return JSONResponse(
            status_code=422,
            content={"message": "Failed to initialize model"}
        )

    return {"message": f"Init model successfully"}


@router.post("/ml_training_manager/init_trainer")
def init_trainer(
        request: InitTrainerBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Init trainer
    :param ml_trainer_app:
    :param request: InitTrainerBody
    :return: JSONResponse
    """
    trainer_type = request.trainer_type
    trainer_id = request.trainer_id
    kwargs = request.kwargs

    if not ml_trainer_app.init_trainer(trainer_type, trainer_id, **kwargs):
        return JSONResponse(
            status_code=422,
            content={"message": "Failed to initialize trainer"}
        )

    return {"message": f"Init trainer successfully"}


@router.post("/ml_training_manager/set_mlflow_model_name")
def set_mlflow_model_name(
        request: SetMLflowModelNameBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Set MLflow model name
    :param ml_trainer_app:
    :param request: SetMLflowModelNameBody
    :return: JSONResponse
    """
    model_name = request.model_name

    if not ml_trainer_app.set_mlflow_model_name(model_name):
        return JSONResponse(
            status_code=422,
            content={"message": "Failed to set MLflow model name"}
        )

    return {"message": f"Set MLflow model name to {model_name} successfully"}


@router.post("/ml_training_manager/set_mlflow_experiment_name")
def set_mlflow_experiment_name(
        request: SetMLflowExperimentNameBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Set MLflow experiment name
    :param ml_trainer_app:
    :param request: SetMLflowExperimentNameBody
    :return: JSONResponse
    """
    experiment_name = request.experiment_name

    if not ml_trainer_app.set_mlflow_experiment_name(experiment_name):
        return JSONResponse(
            status_code=422,
            content={"message": "Failed to set MLflow experiment name"}
        )

    return {"message": f"Set MLflow experiment name to {experiment_name} successfully"}


@router.post("/ml_training_manager/set_mlflow_run_name")
def set_mlflow_run_name(
        request: SetMLflowRunNameBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Set MLflow run name
    :param ml_trainer_app:
    :param request: SetMLflowRunNameBody
    :return: JSONResponse
    """
    run_name = request.run_name

    if not ml_trainer_app.set_mlflow_run_name(run_name):
        return JSONResponse(
            status_code=422,
            content={"message": "Failed to set MLflow run name"}
        )

    return {"message": f"Set MLflow run name to {run_name} successfully"}


@router.post("/ml_training_manager/run_ml_training")
def run_ml_training(
        request: RunMLTrainingBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Run ml training
    :param ml_trainer_app:
    :param request: RunMLTrainingBody
    :return: JSONResponse
    """
    print(f"Received run_ml_training request: {request}")

    epochs = request.kwargs["epochs"]

    if not ml_trainer_app.run_ml_training(epochs):
        print("run_ml_training failed.")
        return JSONResponse(
            status_code=422,
            content={"message": "Failed to run ML training"}
        )

    print("run_ml_training succeeded.")
    return {"message": "Run ML training successfully"}


@router.get("/ml_training_manager/get_data_processor/{data_processor_id}")
def get_data_processor(data_processor_id: str, ml_trainer_app: MLTrainingServingApp = Depends(get_app)):
    """
    Get data processor by ID
    :param data_processor_id: Data processor ID
    :param ml_trainer_app: MLTrainingServingApp
    :return: JSONResponse
    """
    data_processor = ml_trainer_app.get_data_processor(data_processor_id)
    if data_processor:
        # Log the data processor details for debugging
        print(f"Retrieved data processor {data_processor_id}: {data_processor}")
        return {"data_processor": str(data_processor)}
    return JSONResponse(
        status_code=404,
        content={"message": "Data processor not found"}
    )


@router.get("/ml_training_manager/get_trainer/{trainer_id}")
def get_trainer(
        trainer_id: str,
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Get trainer by ID
    :param trainer_id: Trainer ID
    :param ml_trainer_app: MLTrainingServingApp
    :return: JSONResponse
    """
    trainer = ml_trainer_app.get_trainer(trainer_id)
    if trainer:
        return {"trainer": str(trainer)}
    return JSONResponse(
        status_code=404,
        content={"message": "Trainer not found"}
    )


@router.get("/ml_training_manager/get_model/{model_id}")
def get_model(
        model_id: str,
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Get model by ID
    :param model_id: Model ID
    :param ml_trainer_app: MLTrainingServingApp
    :return: JSONResponse
    """
    model = ml_trainer_app.get_model(model_id)
    if model:
        return {"model": str(model)}
    return JSONResponse(
        status_code=404,
        content={"message": "Model not found"}
    )

# New endpoints

@router.get("/ml_training_manager/list_models")
def list_models(ml_trainer_app: MLTrainingServingApp = Depends(get_app)):
    """
    List all models
    :param ml_trainer_app: MLTrainingServingApp
    :return: JSONResponse
    """
    models = ml_trainer_app.list_models()
    return {"models": models}


@router.get("/ml_training_manager/list_trainers")
def list_trainers(ml_trainer_app: MLTrainingServingApp = Depends(get_app)):
    """
    List all trainers
    :param ml_trainer_app: MLTrainingServingApp
    :return: JSONResponse
    """
    trainers = ml_trainer_app.list_trainers()
    return {"trainers": trainers}


@router.get("/ml_training_manager/list_data_processors")
def list_data_processors(ml_trainer_app: MLTrainingServingApp = Depends(get_app)):
    """
    List all data processors
    :param ml_trainer_app: MLTrainingServingApp
    :return: JSONResponse
    """
    data_processors = ml_trainer_app.list_data_processors()
    return {"data_processors": data_processors}


@router.put("/ml_training_manager/update_model/{model_id}")
def update_model(model_id: str, update_params: UpdateModelParams, ml_trainer_app: MLTrainingServingApp = Depends(get_app)):
    """
    Update model parameters and return the updated configuration.
    :param ml_trainer_app: MLTrainingServingApp
    :param model_id: ID of the model to update
    :param update_params: New parameters for the model
    :return: JSONResponse
    """
    if ml_trainer_app.update_model(model_id, update_params.params):
        updated_model = ml_trainer_app.get_model(model_id)
        return {"message": f"Model {model_id} updated successfully", "updated_model": str(updated_model)}
    return JSONResponse(status_code=422, content={"message": f"Failed to update model {model_id}"})


@router.put("/ml_training_manager/update_trainer/{trainer_id}")
def update_trainer(trainer_id: str, update_params: UpdateTrainerParams, ml_trainer_app: MLTrainingServingApp = Depends(get_app)):
    """
    Update trainer parameters and return the updated configuration.
    :param ml_trainer_app: MLTrainingServingApp
    :param trainer_id: ID of the trainer to update
    :param update_params: New parameters for the trainer
    :return: JSONResponse
    """
    if ml_trainer_app.update_trainer(trainer_id, update_params.params):
        updated_trainer = ml_trainer_app.get_trainer(trainer_id)
        return {"message": f"Trainer {trainer_id} updated successfully", "updated_trainer": str(updated_trainer)}
    return JSONResponse(status_code=422, content={"message": f"Failed to update trainer {trainer_id}"})


@router.put("/ml_training_manager/update_data_processor/{data_processor_id}")
def update_data_processor(data_processor_id: str, update_params: UpdateDataProcessorParams, ml_trainer_app: MLTrainingServingApp = Depends(get_app)):
    """
    Update data processor parameters and return the updated configuration.
    :param ml_trainer_app: MLTrainingServingApp
    :param data_processor_id: ID of the data processor to update
    :param update_params: New parameters for the data processor
    :return: JSONResponse
    """
    if ml_trainer_app.update_data_processor(data_processor_id, update_params.params):
        updated_data_processor = ml_trainer_app.get_data_processor(data_processor_id)
        return {"message": f"Data Processor {data_processor_id} updated successfully", "updated_data_processor": str(updated_data_processor)}
    return JSONResponse(status_code=422, content={"message": f"Failed to update data processor {data_processor_id}"})
