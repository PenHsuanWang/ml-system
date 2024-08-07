import threading
import typing

import shap
import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from src.model_ops_manager.mlflow_agent.mlflow_agent import MLFlowAgent
from typing import Dict, Union, Optional

class MLFlowModelsService:
    """
    The MLFlowModelsService is a singleton class to manage models and perform model inference.
    """

    _app = None
    _app_lock = threading.Lock()
    _mlflow_agent = MLFlowAgent()

    def __new__(cls):
        """
        Create a new instance of the class or return the existing singleton instance.
        This method is thread-safe.
        :return: The singleton instance of the class.
        :rtype: MLFlowModelsService
        """
        with cls._app_lock:
            if cls._app is None:
                cls._app = super(MLFlowModelsService, cls).__new__(cls)
                cls._app._initialized = False  # Initialize the _initialized variable here
        return cls._app

    def __init__(self):
        pass

    @classmethod
    def setup_mlflow_agent(cls, *args, **kwargs) -> None:
        """
        Set up the MLFlow agent with the provided tracking server URI.
        :param mlflow_tracking_server: The MLFlow tracking server URI.
        """
        mlflow_tracking_server = kwargs.get("mlflow_tracking_server", None)
        if mlflow_tracking_server is None:
            raise ValueError("MLflow tracking server is not provided")

        cls._mlflow_agent.set_tracking_uri(mlflow_tracking_server)
        cls._mlflow_agent.init_mlflow_client()

    def list_models(self) -> list:
        """
        List all registered models.
        This method uses the MLFlow client to fetch a list of all registered models from the MLFlow server.
        :return: A list of all registered models.
        :rtype: list
        :raises HTTPException: If an error occurs while fetching the list of models.
        """
        try:
            return self._mlflow_agent.list_all_registered_models()
        except MlflowException as me:
            raise HTTPException(status_code=500, detail=f"Failed to fetch models from MLflow: {str(me)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_model_details(self, model_name: str, version: int) -> dict:
        """
        Get the details of a specific model version.
        :param model_name: The name of the model.
        :type model_name: str
        :param version: The version of the model.
        :type version: int
        :return: The details of the model version.
        :rtype: dict
        :raises HTTPException: If an error occurs while fetching the model details.
        """
        try:
            return self._mlflow_agent.get_model_details(model_name, version)
        except MlflowException as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch model details: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_model_comparison(self, model_name1: str, version1: int, model_name2: str, version2: int) -> Dict[str, Union[str, float, None]]:
        """
        Compare two models based on their names and versions.
        This method fetches the details of two models from the MLFlow server using their names and versions.
        It then compares the parameters, metrics, and architecture of the two models.
        :param model_name1: The name of the first model.
        :type model_name1: str
        :param version1: The version of the first model.
        :type version1: int
        :param model_name2: The name of the second model.
        :type model_name2: str
        :param version2: The version of the second model.
        :type version2: int
        :return: A dictionary containing the comparison of parameters, metrics, and architecture of the two models.
        :rtype: ComparisonResult
        :raises HTTPException: If an error occurs while comparing the models.
        """
        try:
            details1 = self._mlflow_agent.get_model_details(model_name1, version1)
            details2 = self._mlflow_agent.get_model_details(model_name2, version2)

            print("Debug: Model details for model 1", details1)
            print("Debug: Model details for model 2", details2)

            details1["training_data_info"] = self.ensure_dict(details1.get("training_data_info", {}))
            details2["training_data_info"] = self.ensure_dict(details2.get("training_data_info", {}))

            filled_details1 = self.fill_missing_data(details1)
            filled_details2 = self.fill_missing_data(details2)

            comparison_result = {
                "parameters": self.compare_dicts(filled_details1.get("parameters", {}), filled_details2.get("parameters", {})),
                "metrics": self.compare_dicts(filled_details1.get("metrics", {}), filled_details2.get("metrics", {})),
                "training_data_info": self.compare_dicts(filled_details1.get("training_data_info", {}), filled_details2.get("training_data_info", {})),
                "architecture": self.compare_values(filled_details1.get("architecture", ""), filled_details2.get("architecture", ""))
            }

            print("Debug: Comparison Result", comparison_result)

            return comparison_result
        except MlflowException as e:
            raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def ensure_dict(value) -> dict:
        """
        Ensure that the value is a dictionary.
        If the value is a string, wrap it in a dictionary with a key 'info'.
        :param value: The value to ensure is a dictionary.
        :return: A dictionary.
        """
        if isinstance(value, dict):
            return value
        return {"info": value}

    @staticmethod
    def fill_missing_data(details: dict) -> dict:
        """
        Fill missing data in the details dictionary with default values.
        :param details: The original details dictionary.
        :return: The details dictionary with missing data filled.
        """
        default_values = {
            "parameters": {},
            "metrics": {},
            "training_data_info": {},
            "architecture": ""
        }
        for key in default_values:
            if key not in details or details[key] is None:
                details[key] = default_values[key]
            elif isinstance(details[key], dict):
                # Fill missing keys in nested dictionaries
                for nested_key in default_values[key]:
                    if nested_key not in details[key]:
                        details[key][nested_key] = default_values[key][nested_key]
        return details

    @staticmethod
    def compare_dicts(dict1: Dict[str, Union[str, float]], dict2: Dict[str, Union[str, float]]) -> Dict[str, Union[str, float, None]]:
        """
        Compare two dictionaries.
        This method compares two dictionaries and returns a new dictionary with the keys from both dictionaries.
        For each key, the new dictionary contains a sub-dictionary with the values from the first and second dictionary.
        :param dict1: The first dictionary.
        :type dict1: dict
        :param dict2: The second dictionary.
        :type dict2: dict
        :return: A dictionary containing the comparison of the two dictionaries.
        :rtype: dict
        """
        comparison = {}
        for key in set(dict1.keys()).union(dict2.keys()):
            comparison[key] = {
                "model1": dict1.get(key) if dict1.get(key) is not None else '',
                "model2": dict2.get(key) if dict2.get(key) is not None else ''
            }
        return comparison

    @staticmethod
    def compare_values(value1: str, value2: str) -> Dict[str, Union[str, float, None]]:
        """
        Compare two values.
        This method compares two values and returns a string if they are the same or a dictionary if they are different.
        :param value1: The first value.
        :param value2: The second value.
        :return: A ComparisonDetail object with the values for model1 and model2.
        :rtype: dict
        """
        return {
            "model1": value1 if value1 is not None else '',
            "model2": value2 if value2 is not None else ''
        }

    def explain_model(self, model_name: str, version: int, X: typing.Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        Generate explanations for a model.
        This method retrieves the specified model from MLFlow using the custom methods,
        generates SHAP values, feature importances, or partial dependence plots for the model,
        and returns these explanations.
        :param model_name: The name of the model.
        :type model_name: str
        :param version: The version of the model.
        :type version: int
        :param X: The input data for which to compute the SHAP values.
        :type X: pandas.DataFrame or numpy.ndarray
        :return: The generated model explanations.
        :rtype: dict
        :raises HTTPException: If the model cannot be explained.
        """
        try:
            model_uri = self._mlflow_agent.get_model_download_source_uri(model_name, model_version=version)
            model = self._mlflow_agent.load_original_model(model_uri)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X should be a pandas DataFrame or numpy ndarray")

        # Convert input data to the appropriate format for SHAP
        X_values = X.values if isinstance(X, pd.DataFrame) else X

        # Calculate SHAP values
        try:
            if isinstance(model, RandomForestRegressor):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, torch.nn.Module):  # Assuming model is a PyTorch neural network
                model.eval()  # Ensure the model is in evaluation mode
                explainer = shap.DeepExplainer(model, torch.from_numpy(X_values).float())
            else:
                explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X_values)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to compute SHAP values: {str(e)}")

        # Get feature importances
        importances = getattr(model, "feature_importances_", None)

        # Generate partial dependence plots (only applicable for tree-based models)
        pdp_results = None
        if isinstance(model, RandomForestRegressor):
            try:
                pdp_results = partial_dependence(model, X, np.arange(X.shape[1]))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to compute partial dependence plots: {str(e)}")

        return {
            'shap_values': shap_values,
            'feature_importances': importances,
            'pdp': pdp_results
        }


def get_mlflow_models_service() -> MLFlowModelsService:
    """
    Get the singleton instance of the MLFlowModelsService class.
    This function returns the singleton instance of the MLFlowModelsService class.
    :return: The singleton instance of the MLFlowModelsService class.
    :rtype: MLFlowModelsService
    """
    return MLFlowModelsService()
