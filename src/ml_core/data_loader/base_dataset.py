# this is the base dataset build on top of the pytorch dataset class.
# to make a module to be able to load local data and pass to the pytorch dataset for training loop.

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset class for the pytorch dataset
    """

    def __init__(self, data, transform=None):
        """
        Args:
            data (list): A list of data samples. Each sample can be of any data type.
            transform (callable, optional): A function/transform to apply to the data. Default is None.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample



