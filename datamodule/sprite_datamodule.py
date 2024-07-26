import os
import sys
import lightning as L
from torch.utils.data import DataLoader

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.general_utils import load_dataset


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
        return DataLoader(self.train_data,
                          num_workers=4,
                          batch_size=self.batch_size,  # 128
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data,
                          num_workers=4,
                          batch_size=self.batch_size,  # 128
                          shuffle=False,
                          drop_last=True,
                          pin_memory=True
                          )
