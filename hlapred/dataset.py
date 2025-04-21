import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from hlapred.utils import get_encoding, SEQUENCE_KEY


class AntibodyDataset(Dataset):

    def __init__(self, inputs):
        """
        p, p_mask: (b,30)
        k_mask: (b,22)
        a: (b,12,35)
        a_mask: (b,12)
        w: (b,1)
        y: (b,1)
        """
        if isinstance(inputs, pd.DataFrame):
            self.p, self.p_mask, self.k_mask, self.a, self.a_mask, self.y, self.w = get_encoding(inputs)
            self.pro_len = inputs[SEQUENCE_KEY].apply(lambda x: len(x)).values.reshape(-1,1)
        else:
            self.p = inputs['p']
            self.p_mask = inputs['p_mask']
            self.k_mask = inputs['k_mask']
            self.a = inputs['a']
            self.a_mask = inputs['a_mask']
            self.y = inputs['y']
            self.w = inputs['w']
            self.pro_len = inputs['pro_len']

        self.p = torch.tensor(self.p, dtype=torch.long)
        self.p_mask = torch.tensor(self.p_mask, dtype=torch.bool)
        self.k_mask = torch.tensor(self.k_mask, dtype=torch.bool)
        self.a = torch.tensor(self.a, dtype=torch.long)
        self.a_mask = torch.tensor(self.a_mask, dtype=torch.bool)
        self.y = torch.tensor(self.y, dtype=torch.float)
        self.w = torch.tensor(self.w, dtype=torch.float)

    def __len__(self):
        return self.p.shape[0]

    def __getitem__(self, idx):
        return self.p[idx], self.p_mask[idx], self.k_mask[idx], self.a[idx], self.a_mask[idx], self.y[idx], self.w[idx]



def AntibodyDataset_breaker(inputs, batch_size, shuffle=True, drop_last=False):
    """
    inputs: pandas.DataFrame
        p, p_mask: (b,30)
        k_mask: (b,22)
        a: (b,12,35)
        a_mask: (b,12)
        y: (b,1)
    """
    if isinstance(inputs, pd.DataFrame):
        N = inputs.shape[0]
        inputs.index = range(N)
        data_idx = np.arange(N)

        while True:
            if shuffle:
                np.random.shuffle(data_idx)

            split = 0
            while (split + 1) * batch_size <= N:
                # Output a batch
                batch_idx = data_idx[split *batch_size:(split + 1) *
                                                batch_size]
                yield AntibodyDataset(inputs.iloc[batch_idx,:].copy())
                split += 1

            # Deal with the part smaller than a batch_size
            left_len = N % batch_size
            if left_len != 0 and not drop_last:
                batch_idx = data_idx[split * batch_size:]
                yield AntibodyDataset(inputs.iloc[batch_idx,:].copy())

    else:
        raise NotImplementedError








