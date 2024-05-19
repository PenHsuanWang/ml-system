import threading
import shap
import numpy as np
import pandas as pd
from fastapi import HTTPException
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from src.model_ops_manager.mlflow_agent.model_downloader import MLFlowClientModelLoader

class MLFlowModelsService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        Create a new instance of the class or return the existing singleton instance.

        This method is thread-safe.

        :return: The singleton instance of the class.
        :rtype: MLFlowModelsService
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MLFlowModelsService, cls).__new__(cls)
                cls._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the instance.

        This method initializes the MLFlow client if the instance has not been initialized.
        """
        if not self._initialized:
            self.client = MLFlowClientModelLoader
            self.client.init_mlflow_client()
            self._initialized = True

    def list_models(self):
        """
        List all registered models.

        This method uses the MLFlow client to fetch a list of all registered models from the MLFlow server.

        :return: A list of all registered models.
        :rtype: list
        :raises HTTPException: If an error occurs while fetching the list of models.
        """
        try:
            return self.client.list_all_registered_models()
        except MlflowException as me:
            raise HTTPException(status_code=500, detail=f"Failed to fetch models from MLflow: {str(me)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_model_comparison(self, model_name1: str, version1: int, model_name2: str, version2: int):
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
        :rtype: dict
        """
        try:
            details1 = self.client.get_model_details(model_name1, version1)
            details2 = self.client.get_model_details(model_name2, version2)
            return {
                "comparison": {
                    "parameters": self.compare_dicts(details1["parameters"], details2["parameters"]),
                    "metrics": self.compare_dicts(details1["metrics"], details2["metrics"]),
                    "architecture": self.compare_values(details1["architecture"], details2["architecture"])
                }
            }
        except MlflowException as e:
            raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")

    @staticmethod
    def compare_dicts(dict1, dict2):
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
        if not all(isinstance(d, dict) for d in [dict1, dict2]):
            raise ValueError("Both inputs must be dictionaries.")
        keys = set(dict1.keys()).union(dict2.keys())
        result = {}
        for key in keys:
            result[key] = {
                "model1": dict1.get(key, "Not available"),
                "model2": dict2.get(key, "Not available")
            }
        return result

    @staticmethod
    def compare_values(val1, val2):
        """
        Compare two values.

        This method compares two values and returns a string if they are the same or a dictionary if they are different.

        :param val1: The first value.
        :param val2: The second value.
        :return: A string if the values are the same, or a dictionary with the values if they are different.
        :rtype: str or dict
        """
        if not isinstance(val1, (str, int, float)) or not isinstance(val2, (str, int, float)):
            raise ValueError("Both values must be of type str, int, or float.")
        return "Same" if val1 == val2 else {"model1": val1, "model2": val2}

    def explain_model(self, model_name: str, version: int, X):
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
            model_uri = self.client.get_model_download_source_uri(model_name, model_version=version)
            model = self.client.load_original_model(model_uri)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X should be a pandas DataFrame or numpy ndarray")

        # Calculate SHAP values
        try:
            if isinstance(model, RandomForestRegressor):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to compute SHAP values: {str(e)}")

        # Get feature importances
        importances = getattr(model, "feature_importances_", None)

        # Generate partial dependence plots
        try:
            pdp_results = partial_dependence(model, X, np.arange(X.shape[1]))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to compute partial dependence plots: {str(e)}")

        return {
            'shap_values': shap_values,
            'feature_importances': importances,
            'pdp': pdp_results
        }

def get_mlflow_models_service():
    """
    Get the singleton instance of the MLFlowModelsService class.

    This function returns the singleton instance of the MLFlowModelsService class.

    :return: The singleton instance of the MLFlowModelsService class.
    :rtype: MLFlowModelsService
    """
    return MLFlowModelsService()
