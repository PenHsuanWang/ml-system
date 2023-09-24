from src.ml_core.inferencer.base_inferencer import BaseInferencer


class PytorchModelInferencer(BaseInferencer):
    """
    the inferencer responsible for passing input data to torch nn model
    """

    def __init__(self, model):
        super().__init__(model)

    def predict(self, input_data):
        """
        predict the output for the given input data
        :param input_data:
        :return:
        """
        self._parse_model_type()

        output = self._model(input_data)

        return output
