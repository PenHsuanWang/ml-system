from src.ml_core.models.torch_nn_models.lstm_model import LSTMModel


class TorchNeuralNetworkModelFactory:
    """
    Factory class for creating torch neural network models
    """

    @staticmethod
    def create_torch_nn_model(model_type: str, **kwargs):
        """
        Create a torch neural network models instance
        :param model_type: Model type
        :return: Torch neural network models instance
        """
        if model_type == "lstm":
            return LSTMModel(**kwargs)
        else:
            raise Exception("Model type not supported")

