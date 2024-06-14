"""
Developing the serving app to export model training setting to REST api endpoint
"""

import os

from fastapi import APIRouter, Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

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
    data_processor_type: str
    dataframe: DataFramePayload
    kwargs: dict


class InitDataProcessorBody(BaseModel):
    data_processor_type: str
    args: list
    kwargs: dict


class InitModelBody(BaseModel):
    model_type: str
    args: list
    kwargs: dict


class InitTrainerBody(BaseModel):
    trainer_type: str
    args: list
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
        data_processor_type = request.data_processor_type
        dataframe_json = request.dataframe.dict()
        kwargs = request.kwargs

        ml_trainer_app.init_data_processor_from_df(data_processor_type, dataframe_json, **kwargs)

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
        data_processor_type = request.data_processor_type
        kwargs = request.kwargs

        ml_trainer_app.init_data_processor(data_processor_type, **kwargs)

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
    kwargs = request.kwargs

    ml_trainer_app.init_model(model_type, **kwargs)

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
    kwargs = request.kwargs

    ml_trainer_app.init_trainer(trainer_type, **kwargs)

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
    epochs = request.kwargs["epochs"]

    ml_trainer_app.run_ml_training(epochs)

    return {"message": f"Run ml training successfully"}

