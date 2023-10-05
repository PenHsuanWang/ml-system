import mlflow


class MLFlowConfiguration:

    _mlflow_connect_tracking_server_uri = None

    @classmethod
    def set_tracking_uri(cls, tracking_uri):
        """
        The method to set the MLFlow tracking uri
        :param tracking_uri:
        :return:
        """
        cls._mlflow_connect_tracking_server_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    @property
    def mlflow_connect_tracking_server_uri(self):
        return self._mlflow_connect_tracking_server_uri
