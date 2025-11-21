import os
import torch
import pandas as pd
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from datasets import load_from_disk, concatenate_datasets

class dataset(Dataset):
    def __init__(self, root_df, n_features):
        print(f"Setting up dataset with n_features = {n_features}")
        
        self.df = root_df.copy()
        self.data = self.df.to_numpy()
        self.features , self.labels = (torch.from_numpy(self.data[:, : n_features]),
                                       torch.from_numpy(self.data[:, n_features:]))

        
    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx,:]
        
    def __len__(self):
        return len(self.data)

    def get_dataset(self):
        return self.data

    def get_df(self):
        return self.df

    def get_via_scores(self):
        return self.df['viability score']

class dataset_classifier(Dataset):
    def __init__(self, root_df, n_features):
        
        self.df = root_df.copy()
        self.df = self.df.drop(columns = ['patient', 'gene 1', 'gene 2', 'gene 1 ensembl', 'gene 2 ensembl'])

        def bool_to_float(x):
            if x:
                return 1.0
            return 0.0

        if self.df["SL"].dtype == np.dtype('bool'):
            self.df["SL"] = list(map(bool_to_float, list(self.df["SL"])))
        self.data = self.df.to_numpy().astype(np.float)

        print(f"Making dataloader from shape {self.data.shape} with {n_features}")

        self.features , self.labels = (torch.from_numpy(self.data[:, : n_features]),
                                       torch.from_numpy(self.data[:, n_features:]))

        
    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx, :]
        
    def __len__(self):
        return len(self.data)

    def get_dataset(self):
        return self.data

    def get_df(self):
        return self.df
    

class dataset_unlabeled(Dataset):
    def __init__(self, root_df):
        
        self.df = root_df.copy()
        self.data = self.df.to_numpy()
        self.features , self.names = (torch.from_numpy(self.data), list(self.df.index))

        
    def __getitem__(self, idx):
        return self.features[idx, :], self.names[idx]
        
    def __len__(self):
        return len(self.data)

    def get_dataset(self):
        return self.data

    def get_df(self):
        return self.df