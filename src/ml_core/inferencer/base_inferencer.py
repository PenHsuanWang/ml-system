
import torch


class BaseInferencer:
    """Base class for all inferencers."""

    def __init__(self, model=None):
        """Initialize the inferencer."""
        self._model = model

    # parsing the model type
    def _parse_model_type(self):
        print(f"Model type: {type(self._model)}")

    def load_model(self, model_path):
        """Load the model from the given path."""
        raise NotImplementedError

    def predict(self, input_data, device):
        """Predict the output for the given input data."""
        raise NotImplementedError


