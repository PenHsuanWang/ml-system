import threading
from fastapi import HTTPException
from src.model_ops_manager.mlflow_agent.client import MLFlowClientModelAgent


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
            self.client = MLFlowClientModelAgent
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
        details1 = self.client.get_model_details(model_name1, version1)
        details2 = self.client.get_model_details(model_name2, version2)
        return {
            "comparison": {
                "parameters": self.compare_dicts(details1["parameters"], details2["parameters"]),
                "metrics": self.compare_dicts(details1["metrics"], details2["metrics"]),
                "architecture": self.compare_values(details1["architecture"], details2["architecture"])
            }
        }

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
        return "Same" if val1 == val2 else {"model1": val1, "model2": val2}


def get_mlflow_models_service():
    """
    Get the singleton instance of the MLFlowModelsService class.

    This function returns the singleton instance of the MLFlowModelsService class.

    :return: The singleton instance of the MLFlowModelsService class.
    :rtype: MLFlowModelsService
    """
    return MLFlowModelsService()
