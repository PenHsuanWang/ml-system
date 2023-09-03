import mlflow
from typing import Any, Optional


class MLFlowTracking:

    @staticmethod
    def start_run(experiment_name: str, run_name: str, *args, **kwargs) -> None:
        """
        The method to start the MLFlow tracking run
        :param experiment_name:
        :param run_name:
        :return:
        """
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name, *args, **kwargs)

    @staticmethod
    def end_run() -> None:
        """
        The method to end the MLFlow tracking run
        :return:
        """
        mlflow.end_run()

    @staticmethod
    def log_param(key: str, value: Any) -> None:
        """
        The method to log the parameter to MLFlow
        :param key:
        :param value:
        :return:
        """
        mlflow.log_param(key, value)

    @staticmethod
    def log_params_many(params: dict) -> None:
        """
        The method to log the parameters to MLFlow
        :param params:
        :return:
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    @staticmethod
    def log_metric(key: str, value: float) -> None:
        """
        The method to log a metric key value pair to MLFlow
        :param key:
        :param value:
        :return:
        """
        mlflow.log_metric(key, value)

    @staticmethod
    def log_metrics_many(metrics: dict[str, float], step: Optional[int]) -> None:
        """
        The method to log many metrics in dictionary format to MLFlow
        :param metrics: dictionary of metrics
        :param step: A single integer step at which to log the specified Metrics. If unspecified, each metric is logged
        at step zero.
        :return:
        """
        mlflow.log_metrics(metrics, step=step)



