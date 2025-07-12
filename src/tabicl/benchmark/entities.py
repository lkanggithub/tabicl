import numpy as np
import pandas as pd


class Dataset:
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_name: str,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.target_name = target_name

    def num_unique_train_data_y(self) -> int:
        y = self.get_train_data_y()
        return len(np.unique(y))

    def get_train_data_x(self) -> np.ndarray:
        columns = list(self.train_data.columns)
        columns.remove(self.target_name)
        return self.train_data[columns].values

    def get_train_data_y(self) -> np.ndarray:
        return self.train_data[self.target_name].values

    def get_test_data_x(self) -> np.ndarray:
        columns = list(self.test_data.columns)
        columns.remove(self.target_name)
        return self.test_data[columns].values

    def get_test_data_y(self) -> np.ndarray:
        return self.test_data[self.target_name].values
