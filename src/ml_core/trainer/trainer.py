from src.ml_core.trainer.torch_nn_trainer import TorchNeuralNetworkTrainer


class TrainerFactory:

    @staticmethod
    def create_trainer(trainer_type: str, **kwargs):
        """
        create a trainer instance
        :param trainer_type: trainer type
        :return: trainer instance
        """
        if trainer_type == "torch_nn":
            return TorchNeuralNetworkTrainer(**kwargs)
        else:
            raise Exception("Trainer type not supported")
