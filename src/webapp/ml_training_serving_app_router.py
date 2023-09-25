"""
Developing the serving app to export model training setting to REST api endpoint
"""

import os

from fastapi import APIRouter, Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
    Init data processor
    :param ml_trainer_app:
    :param request: InitDataProcessorBody
    :return: JSONResponse
    """
    kwargs = request.kwargs

    ml_trainer_app.fetcher_data(
        request.args,
        request.kwargs
    )

    return {"message": f"Init data processor successfully"}


@router.post("/ml_training_manager/init_data_preprocessor")
def init_data_preprocessor(
        request: InitDataProcessorBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Init data preprocessor
    :param ml_trainer_app:
    :param request: SetDataFetcherBody
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
    :param request: SetDataFetcherBody
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
    Setup trainer
    :param ml_trainer_app:
    :param request: InitDataProcessorBody
    :return: JSONResponse
    """
    trainer_type = request.trainer_type
    kwargs = request.kwargs

    ml_trainer_app.init_trainer(trainer_type, **kwargs)

    return {"message": f"Init trainer successfully"}


@router.post("/ml_training_manager/run_ml_training")
def run_ml_training(
        request: RunMLTrainingBody = Body(...),
        ml_trainer_app: MLTrainingServingApp = Depends(get_app)
):
    """
    Run ml training
    :param ml_trainer_app:
    :param request: InitDataProcessorBody
    :return: JSONResponse
    """
    epochs = request.kwargs["epochs"]

    ml_trainer_app.run_ml_training(epochs)

    return {"message": f"Run ml training successfully"}

