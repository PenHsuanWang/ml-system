"""
Developing the serving app to expose mlflow client model downloader ro REST api endpoint
"""

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.webapp.mlflow_model_download_serving_app import get_app, MLFlowModelDownloadServingApp

router = APIRouter()


class SetMLFlowTrackingUriBody(BaseModel):
    tracking_uri: str


class DownloadMLFlowPyfuncModelBody(BaseModel):
    model_name: str
    model_version: int
    model_stage: str


class DownloadMLFlowOriginalModelBody(BaseModel):
    model_name: str
    model_version: int
    model_stage: str


# Define the router for the api endpoint
@router.post("/mlflow_agent/set_mlflow_tracking_uri")
def set_mlflow_tracking_uri(
        request: SetMLFlowTrackingUriBody = Body(...),
        mlflow_model_downloader_app: MLFlowModelDownloadServingApp = Depends(get_app)
):
    """
    Set the mlflow tracking uri
    :param mlflow_model_downloader_app:
    :param request: SetMLFlowTrackingUriBody
    :return: JSONResponse
    """
    tracking_uri = request.tracking_uri
    mlflow_model_downloader_app.set_mlflow_tracking_uri(tracking_uri)
    return JSONResponse(
        status_code=200,
        content={
            "message": "mlflow tracking uri is set"
        }
    )


@router.get("/mlflow_agent/init_mlflow_downloader_client")
def init_mlflow_downloader_client(
        mlflow_model_downloader_app: MLFlowModelDownloadServingApp = Depends(get_app)
):
    """
    Initialize the mlflow downloader client
    :return: JSONResponse
    """
    try:
        mlflow_model_downloader_app.init_mlflow_downloader_client()
        return JSONResponse(
            status_code=200,
            content={
                "message": "mlflow downloader client is initialized"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"mlflow downloader client initialization failed, error: {e}"
            }
        )


@router.post("/mlflow_agent/download_mlflow_pyfunc_model")
def download_mlflow_pyfunc_model(
        request: DownloadMLFlowPyfuncModelBody = Body(...),
        mlflow_model_downloader_app: MLFlowModelDownloadServingApp = Depends(get_app)
):
    """
    Download the mlflow pyfunc model
    :param mlflow_model_downloader_app:
    :param request: DownloadMLFlowPyfuncModelBody
    :return: JSONResponse
    """
    model_name = request.model_name
    model_version = request.model_version
    model_stage = request.model_stage

    try:
        model = mlflow_model_downloader_app.download_mlflow_pyfunc_model(model_name, model_version, model_stage)
        return JSONResponse(
            status_code=200,
            content={
                "message": "mlflow pyfunc model is downloaded",
                "model": model
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"mlflow pyfunc model download failed, error: {e}"
            }
        )


@router.post("/mlflow_agent/download_mlflow_original_model")
def download_mlflow_original_model(
        request: DownloadMLFlowOriginalModelBody = Body(...),
        mlflow_model_downloader_app: MLFlowModelDownloadServingApp = Depends(get_app)
):
    """
    Download the mlflow original model
    :param mlflow_model_downloader_app:
    :param request: DownloadMLFlowOriginalModelBody
    :return: JSONResponse
    """
    model_name = request.model_name
    model_version = request.model_version
    model_stage = request.model_stage

    try:
        model = mlflow_model_downloader_app.download_mlflow_original_model(model_name, model_version, model_stage)
        return JSONResponse(
            status_code=200,
            content={
                "message": "mlflow original model is downloaded",
                "model": model
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"mlflow original model download failed, error: {e}"
            }
        )

