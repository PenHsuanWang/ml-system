from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from src.webapp.mlflow_models_serving_app import get_mlflow_models_service, MLFlowModelsService
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

router = APIRouter()


class ModelVersionDetail(BaseModel):
    version: str
    stage: str
    description: Optional[str]


class ModelDetail(BaseModel):
    name: str
    version: int
    details: Dict[str, Any]


class MLFlowModelDetail(BaseModel):
    name: str
    latest_versions: List[ModelVersionDetail]


class ComparisonDetail(BaseModel):
    model1: str
    model2: str


class ComparisonResult(BaseModel):
    parameters: Dict[str, ComparisonDetail]
    metrics: Dict[str, ComparisonDetail]
    training_data_info: Optional[Dict[str, ComparisonDetail]] = None
    architecture: ComparisonDetail


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


@router.get("/mlflow/models/details/{model_name}/{version}", response_model=ModelDetail)
def get_mlflow_model_details(model_name: str, version: int, service: MLFlowModelsService = Depends(get_mlflow_models_service)):
    """
    Endpoint to get the details of a specific MLFlow model version.
    Returns detailed information about the model including parameters, metrics, and other metadata.

    :param model_name: The name of the model.
    :type model_name: str
    :param version: The version of the model.
    :type version: int
    :param service: The MLFlowModelsService instance.
    :type service: MLFlowModelsService
    :return: The details of the model version.
    :rtype: dict
    :raises HTTPException: If an error occurs while fetching the model details.
    """
    try:
        model_details = service.get_model_details(model_name, version)
        return ModelDetail(name=model_name, version=version, details=model_details)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Internal server error", "error": str(e)})


@router.get("/mlflow/models/compare/{model_name1}/{version1}/{model_name2}/{version2}", response_model=ComparisonResult)
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
