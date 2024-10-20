from src.ml_core.models.torch_nn_models.base_model import BaseModel

import torch
import torch.nn as nn
import logging

# Configure logging
logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):

    def __init__(self, input_size: int, hidden_layer_sizes: list, output_size: int):
        """
        LSTM model with arbitrary number of layers
        :param input_size: The dimensionality of the input at each time step
        :param hidden_layer_sizes: A list with the number of features in the hidden state h for each LSTM layer
        :param output_size: The dimensionality of the output at each time step
        """
        super(LSTMModel, self).__init__()

        if not hidden_layer_sizes:
            raise RuntimeError("hidden_layer_sizes must contain at least one layer size")

        self.input_size = input_size  # Added input_size attribute
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lstm_layers = nn.ModuleList()

        # Create the first LSTM layer with the input size
        self.lstm_layers.append(nn.LSTM(input_size, hidden_layer_sizes[0], batch_first=True))

        # Create additional LSTM layers
        for i in range(1, len(hidden_layer_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True))

        # Fully connected layer
        self.fc = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the LSTM model
        :param x: The input tensor
        :return: The output tensor after forward pass
        """
        for lstm_layer in self.lstm_layers:
            h0 = torch.zeros(1, x.size(0), lstm_layer.hidden_size).to(x.device)
            c0 = torch.zeros(1, x.size(0), lstm_layer.hidden_size).to(x.device)
            x, _ = lstm_layer(x, (h0, c0))

        out = self.fc(x[:, -1, :])
        return out

    def get_model_hyper_parameters(self) -> dict:
        """
        Get the model hyper-parameters
        :return: Model hyper-parameters as a dictionary
        """
        model_hyper_parameters = {
            "input_size": self.input_size,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "output_size": self.fc.out_features
        }
        return model_hyper_parameters

    def to_dict(self):
        """
        Serialize the LSTMModel object to a dictionary.
        """
        return {
            'input_size': self.input_size,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'output_size': self.fc.out_features
        }