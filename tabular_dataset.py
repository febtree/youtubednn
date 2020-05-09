import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):

    def __init__(self, data, cat_cols=None, output_col=None):
        """
        data: pandas dataframe
        cat_cols: categorical column name list, should be label-encoded beforehand
        output_col: output column name
        """
        self.n = data.shape[0]
        # target data
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [ col for col in data.columns if col not in (self.cat_cols+[output_col]) ]

        # continous features data
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        # categorical features data
        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
        one sample of training data
        """
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]

