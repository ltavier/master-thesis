import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import json


class ParquetDataset(Dataset):
    """
    mode='window' → return single-window samples (x1: [seq_length, num_charac], y: scalar, date)
    mode='month'  → return full-month batches (x1: [N_m, seq_length, num_charac], y: [N_m], month)
    """
    def __init__(self,
                 dataframe,
                 features_json,
                 device='cpu',
                 seq_len = 60,
                 mode='window'):
        assert mode in ('window', 'month')
        self.device = device
        self.mode   = mode
        self.seq_len = seq_len

        # 1) Load parquet(s)
        df = dataframe

        # If month mode, sort & group by end_date
        if mode == 'month':
            df = df.sort_values('end_date').reset_index(drop=True)
            self.months  = []
            self.indices = []
            for mon, sub in df.groupby('end_date'):
                self.months.append(mon)
                self.indices.append(sub.index.values)

        # 2) Pull out raw arrays (flattened X1, Y, and dates if window)
        self.X1_arr = df['X1'].values      # each is a 1D list of length (seq_length * num_charac)
        self.Y_arr  = df['Y'].values
        if mode == 'window':
            self.dates = df['end_date'].values

        # 3) Load JSON metadata to figure out seq_length & num_charac
        with open(features_json, 'r') as f:
            metadata = json.load(f)
        char_list       = metadata['characteristics']
        self.num_charac = len(char_list)
        # compute seq_length from any X1 entry
        total_len       = len(self.X1_arr[0])
        assert total_len % self.num_charac == 0
        self.seq_length = total_len // self.num_charac

    def __len__(self):
        return len(self.Y_arr) if self.mode == 'window' else len(self.months)

    def __getitem__(self, idx):
        if self.mode == 'window':
            # single window
            flat = self.X1_arr[idx]
            x1   = torch.tensor(flat,
                                dtype=torch.float32,
                                device=self.device)
            x1   = x1.view(self.seq_length, self.num_charac)
            if self.seq_len < 60:
                x1 = x1[60-self.seq_len:,:]
            y    = torch.tensor(self.Y_arr[idx],
                                dtype=torch.float32,
                                device=self.device)
            date = self.dates[idx]
            return x1, y, date

        else:
            # full-month batch
            inds = self.indices[idx]
            rawX = [self.X1_arr[i] for i in inds]
            rawY = [self.Y_arr[i]  for i in inds]

            X1 = torch.tensor(np.stack(rawX),
                              dtype=torch.float32,
                              device=self.device)
            # shape → (N_m, seq_length, num_charac)
            X1 = X1.view(len(inds), self.seq_length, self.num_charac)
            if self.seq_len < 60:
                X1 = X1[:,60-self.seq_len:,:]

            Y = torch.tensor(rawY,
                             dtype=torch.float32,
                             device=self.device)
            month = self.months[idx]
            return X1, Y, month

    def get_loader(self, batch_size=None, shuffle=None, num_workers=5):
        """
        For window mode, you probably want batch_size>1 and shuffle=True.
        For month mode, use batch_size=1 & shuffle=False.
        """
        if self.mode == 'window':
            assert batch_size is not None and shuffle is not None
            return DataLoader(self,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=(self.device!='cpu'))
        else:
            return DataLoader(self,
                              batch_size=1,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=(self.device!='cpu'))