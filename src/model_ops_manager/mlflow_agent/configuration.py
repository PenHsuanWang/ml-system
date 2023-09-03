import mlflow


class MLFlowConfiguration:

    @staticmethod
    def set_tracking_uri(tracking_uri):
        """
        The method to set the MLFlow tracking uri
        :param tracking_uri:
        :return:
        """
        mlflow.set_tracking_uri(tracking_uri)



