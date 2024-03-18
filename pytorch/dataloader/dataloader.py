import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random
from einops import rearrange
    
class DeepMIMOSampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, files_dir = '/mnt/d/DeepMIMO_datasets/O1_3p5/samples/', 
        ) -> None:
        super().__init__()
        self.files_dir = files_dir
        self.data = None
        self.build()
        
    def build(self):
        self.data = np.load(self.files_dir + 'filepath.npy')
        print('file walk complete')
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        cur_data =np.load(os.path.join(self.files_dir, self.data[idx]+'.npy'))
        
        # cur_data = cur_data / (np.abs(cur_data).max(keepdims = True) + 1e-9)
        
        cur_data = cur_data / np.abs(cur_data).mean(keepdims = True)
        
        
        # cur_data_abs = np.abs(cur_data)
        # cur_data = (cur_data - cur_data.mean(keepdims = True)) / cur_data_abs.std(keepdims = True)
        
        cur_data = np.stack([cur_data.real, cur_data.imag], axis = -1)
        cur_data = torch.from_numpy(cur_data)#.bfloat16()
        
        return cur_data