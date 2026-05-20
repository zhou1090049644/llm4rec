import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os

import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader, DistributedSampler
import numpy as np
import pyarrow.orc as orc
import pyarrow as pa

class CustomNpzFile(data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.embeddings = []
        self.item_ids = []

        for file in self.file_list:
            npz_data = np.load(file, allow_pickle=True)
            embeddings = npz_data['embs']
            item_id = npz_data['ids']

            if item_id.ndim == 1:
                item_id = item_id.reshape(-1, 1)

            self.embeddings.append(embeddings)
            self.item_ids.append(item_id)
            print(f'Processed {file}')

        self.embeddings = np.vstack(self.embeddings)
        self.item_ids = np.vstack(self.item_ids)
        self.item_ids = self.item_ids.squeeze(axis=1)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        item_id = self.item_ids[index]
        
        tensor_emb = torch.as_tensor(emb, dtype=torch.float32)
        
        item_id = np.asarray(item_id).astype(np.int64)
        tensor_item_id = torch.as_tensor(item_id, dtype=torch.int64)
        
        return tensor_item_id, tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
