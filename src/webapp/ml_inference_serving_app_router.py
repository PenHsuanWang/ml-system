"""
Developing the serving app to export model inference setting to REST api endpoint
"""
import numpy as np

from fastapi import APIRouter, Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.webapp.ml_inference_serving_app import get_app, MlInferenceServingApp

# Definition of FastAPI router
router = APIRouter()


# Define the pydantic model for request body
class SetMlflowAgentBody(BaseModel):
    mlflow_tracking_server_uri: str


class SetModelFromMlflowArtifactBody(BaseModel):
    model_info: dict


class SetModelBody(BaseModel):
    model_name: str
    model_bytes: str


class RemoveModelFromServingListBody(BaseModel):
    model_name: str


class InferencerBody(BaseModel):
    model_name: str
    data: list
    device: str


# Define the REST api endpoint
@router.post("/ml_inference_manager/set_mlflow_agent")
def set_mlflow_agent_tracking_server_uri(
        request: SetMlflowAgentBody = Body(...),
        ml_inference_serving_app: MlInferenceServingApp = Depends(get_app)
):
    """
    Set mlflow agent tracking server uri
    :param request: request body
    :param ml_inference_serving_app: ml inference serving app
    :return: response
    """
    mlflow_tracking_server_uri = request.mlflow_tracking_server_uri
    ml_inference_serving_app.setup_mlflow_agent(mlflow_tracking_server=mlflow_tracking_server_uri)
    return JSONResponse(content={"message": "success"})


@router.get("/ml_inference_manager/get_list_all_model_in_serving")
def get_list_all_model_in_serving(
        ml_inference_serving_app: MlInferenceServingApp = Depends(get_app)
):
    """
    return the list of model name in serving
    :param ml_inference_serving_app:
    :return:
    """
    return JSONResponse(content={"message": f"all models in serving list: {ml_inference_serving_app.list_all_model_in_serving()}"})


@router.post("/ml_inference_manager/set_model_from_mlflow_artifact_origin_flavor")
def set_model_from_mlflow_artifact_origin_flavor(
        request: SetModelFromMlflowArtifactBody = Body(...),
        ml_inference_serving_app: MlInferenceServingApp = Depends(get_app)
):
    """
    Set model from mlflow artifact origin flavor
    :param request: request body
    :param ml_inference_serving_app: ml inference serving app
    :return: response
    """
    model_info = request.model_info
    model_name = model_info.get("model_name", None)
    model_version = model_info.get("model_version", None)
    model_stage = model_info.get("model_stage", None)
    if model_name is None:
        return JSONResponse(content={"message": "model name is not provided"})
    if ml_inference_serving_app.set_model_from_mlflow_artifact_origin_flavor(
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage
    ):
        return JSONResponse(content={"message": "success"})
    else:
        return JSONResponse(content={"message": f"Fail to setup model {model_info} from mlflow artifact server"})


@router.post("/ml_inference_manager/set_model_from_mlflow_artifact_pyfunc")
def set_model_from_mlflow_artifact_pyfunc_flavor(
        request: SetModelFromMlflowArtifactBody = Body(...),
        ml_inference_serving_app: MlInferenceServingApp = Depends(get_app)
):
    """
    Set model from mlflow artifact pyfunc flavor
    :param request: request body
    :param ml_inference_serving_app: ml inference serving app
    :return: response
    """
    model_info = request.model_info
    model_name = model_info.get("model_name", None)
    model_version = model_info.get("model_version", None)
    model_stage = model_info.get("model_stage", None)
    if model_name is None:
        return JSONResponse(content={"message": "model name is not provided"})
    if ml_inference_serving_app.set_model_from_mlflow_artifact_pyfunc(
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage
    ):
        return JSONResponse(content={"message": "success"})
    else:
        return JSONResponse(content={"message": f"Fail to setup model {model_info} from mlflow artifact server"})


@router.post("/ml_inference_manager/set_model")
def set_model_from_system(
        request: SetModelBody = Body(...),
        ml_inference_serving_app: MlInferenceServingApp = Depends(get_app)
):
    """
    Set model from system
    :param request: request body
    :param ml_inference_serving_app: ml inference serving app
    :return: response
    """
    model_name = request.model_name
    model_bytes = request.model_bytes
    if model_name is None:
        return JSONResponse(content={"message": "model name is not provided"})

    # TODO: reconstruction of model

    if ml_inference_serving_app.set_model_to_serving_list(
        model_name=model_name,
        model=model_bytes
    ):
        return JSONResponse(content={"message": "success"})
    else:
        return JSONResponse(content={"message": f"Fail to setup model {model_name}, check the model object and serialization process"})


@router.post("/ml_inference_manager/remove_model_from_serving_list")
def remove_model_from_serving_list(
        request: RemoveModelFromServingListBody = Body(...),
        ml_inference_serving_app: MlInferenceServingApp = Depends(get_app)
):
    """
    Remove model from serving list
    :param request: request body
    :param ml_inference_serving_app: ml inference serving app
    :return: response
    """
    model_name = request.model_name
    if model_name is None:
        return JSONResponse(content={"message": "model name is not provided"})
    if ml_inference_serving_app.remove_model_from_serving_list(model_name=model_name):
        return JSONResponse(content={"message": "success"})
    else:
        return JSONResponse(content={"message": f"Fail to delete model {model_name} from serving list, you can check the model {model_name} is exist by invoke endpoint `/ml_inference_manager/get_list_all_model_in_serving`"})


@router.post("/ml_inference_manager/inference")
def inference_rest_api_endpoint(
        request: InferencerBody = Body(...),
        ml_inference_serving_app: MlInferenceServingApp = Depends(get_app)
):
    """
    Inference function to get model inference result
    :param request: request body
    :param ml_inference_serving_app: ml inference serving app
    :return: response
    """
    model_name = request.model_name
    data = np.array(request.data)
    device = request.device
    if model_name is None:
        return JSONResponse(content={"message": "model name is not provided"})
    if data is None:
        return JSONResponse(content={"message": "data is not provided"})
    if device is None:
        return JSONResponse(content={"message": "device is not provided"})
    output_predict = ml_inference_serving_app.inference(model_name=model_name, data_input=data, device=device)
    return JSONResponse(content={"message": "success", "output": output_predict.tolist()})


