from src.ml_core.models.torch_nn_models.base_model import BaseModel

import torch
import torch.nn as nn


class LSTMModel(BaseModel):

    def __init__(self, input_size, hidden_size, output_size):
        """
        LSTM model
        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

