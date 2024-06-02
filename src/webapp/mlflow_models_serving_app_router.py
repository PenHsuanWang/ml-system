from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from src.webapp.mlflow_models_serving_app import get_mlflow_models_service, MLFlowModelsService
from typing import List, Dict, Optional
from pydantic import BaseModel

router = APIRouter()


class ModelVersionDetail(BaseModel):
    """
    A Pydantic model representing details of a specific version of an MLFlow model.

    Attributes:
        version (str): The version number of the model.
        stage (str): The stage of the model (e.g., Production, Staging, Archived).
        description (Optional[str]): A description of the model version, providing additional context about its purpose or changes.
    """
    version: str
    stage: str
    description: Optional[str]


class MLFlowModelDetail(BaseModel):
    """
    A Pydantic model representing a registered MLFlow model with details on its various versions.

    This model is used to aggregate information about different versions of a model registered in MLFlow,
    including their stages and descriptions. It is primarily used in API responses where a comprehensive overview
    of a model and its versions is required.

    Attributes:
        name (str): The name of the registered model.
        latest_versions (List[ModelVersionDetail]): A list of `ModelVersionDetail` instances,
            each representing a version of the model with its respective stage and optional description.
    """
    name: str
    latest_versions: List[ModelVersionDetail]



@router.get("/mlflow/models", response_model=List[MLFlowModelDetail], responses={200: {"description": "List of all MLFlow models"}})
def get_mlflow_models(service: MLFlowModelsService = Depends(get_mlflow_models_service)):
    """
    Endpoint to list all registered MLFlow models.
    Returns detailed information about each model including name, versions, stages, and descriptions.

    :param service: The MLFlowModelsService instance.
    :type service: MLFlowModelsService
    :return: A list of all registered models.
    :rtype: List[MLFlowModelDetail]
    :raises HTTPExpression: If an error occurs while fetching the list of models.
    """
    try:
        raw_models = service.list_models()
        models = [
            MLFlowModelDetail(
                name=model['name'],
                latest_versions=[
                    ModelVersionDetail(version=ver['version'], stage=ver['stage'], description=ver.get('description'))
                    for ver in model['latest_versions']
                ]
            ) for model in raw_models
        ]
        return models
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Internal server error", "error": str(e)})


@router.get("/mlflow/models/compare/{model_name1}/{version1}/{model_name2}/{version2}", response_model=Dict[str, Dict[str, Dict[str, str]]])
def compare_mlflow_models(model_name1: str, version1: int, model_name2: str, version2: int, service: MLFlowModelsService = Depends(get_mlflow_models_service)):
    """
    Endpoint to compare two MLFlow models based on their names and versions.
    Returns a comparison of the models' parameters, metrics, and architecture.

    :param model_name1: The name of the first model.
    :type model_name1: str
    :param version1: The version of the first model.
    :type version1: int
    :param model_name2: The name of the second model.
    :type model_name2: str
    :param version2: The version of the second model.
    :type version2: int
    :param service: The MLFlowModelsService instance.
    :type service: MLFlowModelsService
    :return: A dictionary containing the comparison of parameters, metrics, and architecture of the two models.
    :rtype: dict
    :raises HTTPException: If an error occurs while fetching the comparison of models.
    """
    try:
        comparison_result = service.get_model_comparison(model_name1, version1, model_name2, version2)
        return comparison_result
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Internal server error", "error": str(e)})
