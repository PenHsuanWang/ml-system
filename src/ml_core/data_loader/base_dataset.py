# this is the base dataset build on top of the pytorch dataset class.
# to make a module to be able to load local data and pass to the pytorch dataset for training loop.

from typing import Any

import torch
from torch.utils.data import Dataset

import numpy as np

class TimeSeriesDataset(Dataset):
    """
    The generic dataset class responsible for loading numerical data as input dataset
    for example, the python list, numpy array, and event torch tensor is acceptable.
    """
    def __init__(self, data_x: Any, data_y: Any, transform=None):
        """
        input data can be python list, numpy array, and torch tensor.
        :param data: Any of python list, numpy array, and torch tensor.
        :param transform:
        """
        assert len(data_x) == len(data_y), "Input data (X) and target data (Y) must have the same length."
        self.data_x = torch.from_numpy(data_x).float()
        self.data_y = torch.from_numpy(data_y).float()
        self.transform = transform

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, id):
        sample_x = self.data_x[id]
        sample_y = self.data_y[id]
        if self.transform:
            sample_x = self.transform(sample_x)
            sample_y = self.transform(sample_y)
        return sample_x, sample_y


class PandasDataframeDataset(Dataset):
    """
    The dataset responsible for loading pandas dataframe as input dataset.
    Intering the dataframe, the dataset will load the data from the dataframe.
    """
    def __init__(self, dataframe, transform=None):
        """
        :param dataframe: pandas dataframe
        :param transform:
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    dataset = TimeSeriesDataset(x, y)
    data_laoder = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    for i, data in enumerate(data_laoder):
        print(data)
        x, y = data
        print(x)
        print(y)




