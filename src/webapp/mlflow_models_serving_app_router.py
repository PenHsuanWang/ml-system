from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from src.webapp.mlflow_models_serving_app import get_mlflow_models_service, MLFlowModelsService
from typing import List, Dict
from pydantic import BaseModel


router = APIRouter()


class MLFlowModelDetail(BaseModel):
    name: str
    version: List[str]
    stage: List[str]
    description: List[str]


@router.get("/mlflow/models", response_model=List[MLFlowModelDetail], responses={200: {"description": "List of all MLFlow models"}})
def get_mlflow_models(service: MLFlowModelsService = Depends(get_mlflow_models_service)):
    """
    Endpoint to list all registered MLFlow models.
    Returns detailed information about each model including name, versions, stages, and descriptions.
    """
    try:
        models = service.list_models()
        return models
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": str(e.detail)})
    except Exception as e:
        # Catch-all for any other unexpected errors
        return JSONResponse(status_code=500, content={"message": "Internal server error", "error": str(e)})

