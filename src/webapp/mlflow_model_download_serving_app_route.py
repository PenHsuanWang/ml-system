"""
Developing the serving app to expose mlflow client model downloader ro REST api endpoint
"""
import io
import torch

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.webapp.mlflow_model_download_serving_app import get_app, MLFlowModelDownloadServingApp

router = APIRouter()


class SetMLFlowTrackingUriBody(BaseModel):
    tracking_uri: str


class DownloadMLFlowPyfuncModelBody(BaseModel):
    model_info: dict


class DownloadMLFlowOriginalModelBody(BaseModel):
    model_info: dict


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

    kwargs = request.model_info
    model_name = kwargs.get("model_name", None)
    model_version = kwargs.get("model_version", None)
    model_stage = kwargs.get("model_stage", None)

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

    kwargs = request.model_info
    model_name = kwargs.get("model_name", None)
    model_version = kwargs.get("model_version", None)
    model_stage = kwargs.get("model_stage", None)

    try:
        model = mlflow_model_downloader_app.download_mlflow_original_model(model_name, model_version, model_stage)

        print(type(model))

        print("fetch model from mlflow original model downloader")

        # serialize the model in memory to return to client
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        model_bytes = buffer.getvalue()

        return JSONResponse(
            status_code=200,
            content={
                "message": "mlflow original model is downloaded",
                "model": model_bytes.decode('latin1')
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"mlflow original model download failed, error: {e}"
            }
        )

