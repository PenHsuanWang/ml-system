from src.ml_core.models.torch_nn_models.base_model import BaseModel

import torch
import torch.nn as nn


class LSTMModel(BaseModel):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        LSTM model
        :param input_size: The dimensionality of the input at each time step
        :param hidden_size: The number of features in the hidden state h
        :param output_size: The dimensionality of the output at each time step
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the LSTM model
        :param x: The input tensor
        :return: The output tensor after forward pass
        """
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def get_model_hyper_parameters(self) -> dict:
        """
        get the model hyper-parameters
        :return:
        """

        model_hyper_parameters = {}

        for name, param in self.state_dict().items():
            layer_type = name.split('.')[0]  # e.g., "conv1", "fc1", etc.
            # if "fc" in layer_type or "conv" in layer_type:
            # print(f"Layer: {name}, Shape: {param.shape}")
            model_hyper_parameters[name] = param.shape

        return model_hyper_parameters


