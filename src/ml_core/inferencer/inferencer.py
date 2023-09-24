from src.ml_core.inferencer.pytorch_inferencer import PytorchModelInferencer

class InferencerFactory:
    """
    the factory class for inferencer, based on difference model flavor to provide corrresponding inferencer
    """

    @staticmethod
    def create_inferencer(model_flavor: str, **kwargs):
        """
        create a inferencer instance
        :param model_flavor: model flavor
        :return: inferencer instance
        """
        if model_flavor == "pytorch":
            return PytorchModelInferencer(**kwargs)
        else:
            raise Exception("Model flavor not supported")
