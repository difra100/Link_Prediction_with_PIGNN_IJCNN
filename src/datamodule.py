from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import os

# num_cpu_cores = 1#os.cpu_count()

from src.config import *
from src.utils import *

class DataModuleLP(pl.LightningDataModule):

    def __init__(self, train_set, val_set, test_set, mode):
        super().__init__()
        self.mode = mode  # "hp" or "test"
        self.train_set, self.val_set, self.test_set = train_set, val_set, test_set

    def setup(self, stage=None):
        if stage == 'fit':
            return
        
    def train_dataloader(self, *args, **kwargs):
        return DataLoader([self.train_set], shuffle=False)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader([self.val_set], shuffle=False)
    
    def test_dataloader(self, *args, **kwargs):
        if self.mode == 'hp':
            return DataLoader([self.val_set], shuffle=False)
        elif self.mode == 'test':
            return DataLoader([self.test_set], shuffle=False)
    


    