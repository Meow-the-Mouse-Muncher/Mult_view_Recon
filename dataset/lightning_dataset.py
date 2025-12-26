import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from typing import Optional


class RefocusDataset(Dataset):
    """从 HDF5 文件加载 GT/occ 数据的数据集类。"""
    def __init__(self, h5_files: list, split: str = 'train'):
        self.h5_files = h5_files
        self.split = split
        self.samples = []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                # 假设每个 HDF5 有 'gt' 和 'occ' 数据集
                gt_data = f['gt'][:]
                occ_data = f['occ'][:]
                # 这里可以添加分割逻辑，比如按比例划分 train/val
                # 简单起见，所有数据都用
                self.samples.append((gt_data, occ_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt, occ = self.samples[idx]
        # gt: (1, 3, H, W), occ: (1, C*3, H, W)
        gt = torch.from_numpy(gt).float().squeeze(0)  # -> (3, H, W)
        occ = torch.from_numpy(occ).float().squeeze(0)  # -> (C*3, H, W)
        return gt, occ


class RefocusDataModule(L.LightningDataModule):
    """PyTorch Lightning 数据模块，用于管理 Refocus 数据集."""
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.h5_files = self._find_h5_files()

    def _find_h5_files(self):
        """递归查找所有 .h5 文件。"""
        h5_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
        return h5_files

    def setup(self, stage: Optional[str] = None):
        """设置数据集划分"""
        if stage == 'fit' or stage is None:
            # # 简单划分：前 80% 为 train，后 20% 为 val
            # train_size = int(0.8 * len(self.h5_files))
            self.train_dataset = RefocusDataset(self.h5_files, split='train')
            # self.val_dataset = RefocusDataset(self.h5_files[train_size:], split='val')
        elif stage == 'test':
            self.test_dataset = RefocusDataset(self.h5_files, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)