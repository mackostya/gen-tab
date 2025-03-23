import numpy as np
import h5py
import logging
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
from src.dataset.data_transformer import DataTransformer


class HousePricingDataset(Dataset):
    def __init__(self, file_path: str, data_transformer: DataTransformer):
        """
        Initializes the HousePricingDataset object by loading the dataset from the specified HDF5 file and setting up
        feature configurations and transformations.

        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            data_transformer (Optional[DataTransformer]): A data transformer object for transforming the data. Default is None.
        """
        logging.debug("Loading the csv file")
        self._file_name = file_path
        self._data = pd.read_csv(self._file_name)
        self._numerical_features = self._data.select_dtypes(include=[np.number]).columns.tolist()
        if "SalePrice" in self._numerical_features:
            self._numerical_features.remove("SalePrice")
        if "Id" in self._numerical_features:
            self._numerical_features.remove("Id")
        self._data_numerical = self._data[self._numerical_features]
        self._data_transformer = data_transformer
        self.input_dim = len(self._numerical_features)

    def fit_transformer(self):
        """
        Transforms the data using the specified data transformer.

        Args:
            data_transformer (DataTransformer): The data transformer object.
        """
        self._data_transformer.fit(self._data_numerical)

    def transform(self):
        """
        Transforms the data using the specified data transformer.

        Args:
            data_transformer (DataTransformer): The data transformer object.
        """
        self._data_numerical = self._data_transformer.transform(self._data_numerical)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._data_numerical)

    def __getitem__(self, idx):
        """
        Gets the item at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: The X value, label, and y value at the specified index.
        """
        return self._data_numerical[idx]
