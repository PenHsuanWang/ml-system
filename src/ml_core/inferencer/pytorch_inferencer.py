import numpy as np
import torch

from src.ml_core.inferencer.base_inferencer import BaseInferencer


class PytorchModelInferencer(BaseInferencer):
    """
    the inferencer responsible for passing input data to torch nn model
    """

    def __init__(self, model):
        super().__init__(model)

    def predict(self, input_data: np.ndarray, device) -> np.ndarray:
        """
        predict the output for the given input data
        :param input_data: input ndarray to run model inference
        :param device: device to run the model inference
        :return:
        """
        self._parse_model_type()

        input_data = torch.from_numpy(input_data).float().to(device)

        output = self._model(input_data).to("cpu")

        # convert the output to numpy array
        output = output.detach().numpy()

        return output
