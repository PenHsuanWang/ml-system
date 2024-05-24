import time
from threading import Thread, Event
from typing import Dict, Callable, List
from mlflow import MlflowClient
from sklearn.metrics import f1_score, recall_score, precision_score

class ModelPerformanceMonitor:
    """
    A class to monitor the performance of a deployed model at regular intervals and log metrics to MLflow.

    :param model_name: Name of the model to monitor.
    :param model_version: Version of the model to monitor.
    :param fetch_predictions_fn: Function to fetch predictions and ground truth labels.
    :param interval: Time interval (in seconds) to log metrics.
    :param alert_thresholds: Dictionary of metric thresholds to trigger alerts.
    """
    def __init__(
        self,
        model_name: str,
        model_version: int,
        fetch_predictions_fn: Callable[[], Dict[str, List[float]]],
        interval: int = 60,
        alert_thresholds: Dict[str, float] = None
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.interval = interval
        self.fetch_predictions_fn = fetch_predictions_fn
        self.client = MlflowClient()
        self.alert_thresholds = alert_thresholds or {}
        self.stop_event = Event()
        self.thread = Thread(target=self.monitor_performance)
        self.thread.daemon = True

    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        self.thread.start()

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread gracefully."""
        self.stop_event.set()
        self.thread.join()

    def monitor_performance(self) -> None:
        """Continuously fetch and log performance metrics at defined intervals."""
        while not self.stop_event.is_set():
            try:
                data = self.fetch_predictions_fn()
                metrics = self.calculate_metrics(data['y_true'], data['y_pred'])
                self.log_metrics(metrics)
                self.check_alerts(metrics)
            except Exception as e:
                print(f"Error in monitoring performance: {e}")
            time.sleep(self.interval)

    def calculate_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics such as F1 score, recall, and precision.

        :param y_true: Ground truth labels.
        :param y_pred: Predicted labels.
        :return: Dictionary containing metric names and values.
        """
        return {
            'f1_score': f1_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'precision': precision_score(y_true, y_pred, average='macro')
        }

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log the fetched metrics to MLflow.

        :param metrics: Dictionary containing metric names and values.
        """
        for key, value in metrics.items():
            self.client.log_metric(self.model_name, key, value, self.model_version)

    def check_alerts(self, metrics: Dict[str, float]) -> None:
        """
        Check if any metric falls below the predefined thresholds and trigger alerts.

        :param metrics: Dictionary containing metric names and values.
        """
        for metric, threshold in self.alert_thresholds.items():
            if metrics.get(metric, float('inf')) < threshold:
                self.trigger_alert(metric, metrics[metric])

    def trigger_alert(self, metric: str, value: float) -> None:
        """
        Trigger an alert for the specified metric.

        :param metric: Name of the metric.
        :param value: Value of the metric.
        """
        # Implement alerting mechanism, e.g., sending an email or a message to a monitoring system.
        print(f"Alert: {metric} has fallen below the threshold. Current value: {value}")

# Example usage:
if __name__ == "__main__":
    def fetch_predictions():
        # This function should return a dictionary with keys 'y_true' and 'y_pred'
        # Replace with actual implementation
        return {'y_true': [0, 1, 1, 0], 'y_pred': [0, 1, 0, 0]}

    monitor = ModelPerformanceMonitor(
        model_name="MyModel",
        model_version=1,
        fetch_predictions_fn=fetch_predictions,
        interval=30,
        alert_thresholds={'f1_score': 0.75}
    )

    monitor.start_monitoring()
    # Run monitoring for a while and then stop
    time.sleep(120)
    monitor.stop_monitoring()
