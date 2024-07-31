import os
import sys
import lightning as L
import torch
from torch.utils.data import DataLoader

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.general_utils import load_dataset


def create_dataloader(data, batch_size, is_train=True):
    return DataLoader(data,
                      num_workers=4,
                      batch_size=batch_size,
                      shuffle=is_train,
                      drop_last=True,
                      pin_memory=True
                      )

class SpriteDataModule(L.LightningDataModule):
    def __init__(self, dir_path: str, batch_size: int):
        super().__init__()
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None

    def setup(self, stage: str) -> None:
        self.train_data, self.val_data = load_dataset(self.dir_path)

    def train_dataloader(self) -> DataLoader:
        return create_dataloader(self.train_data, self.batch_size, is_train=True)

    def val_dataloader(self) -> DataLoader:
        return create_dataloader(self.val_data, self.batch_size, is_train=False)
