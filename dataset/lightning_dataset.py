import os
import h5py
import torch
import random
from torch.utils.data import Dataset, DataLoader
import lightning as L
from typing import Optional


class RefocusDataset(Dataset):
    """从 HDF5 文件加载 GT/occ 数据的数据集类。"""
    def __init__(self, h5_files: list, split: str = 'train'):
        self.h5_files = h5_files
        self.split = split

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_file = self.h5_files[idx]
        with h5py.File(h5_file, 'r') as f:
            gt = torch.from_numpy(f['gt'][:]).float()/255      
            occ = torch.from_numpy(f['occ'][:]).float()/255
            occ_mid = torch.from_numpy(f['view'][:]).float()/255
        
        # 数据增强：随机翻转
        if self.split == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                gt = torch.flip(gt, dims=[2])      # [C, H, W] -> W 是 dim 2
                occ = torch.flip(occ, dims=[2])
                occ_mid = torch.flip(occ_mid, dims=[2])
            
            # 随机垂直翻转
            if random.random() > 0.5:
                gt = torch.flip(gt, dims=[1])      # [C, H, W] -> H 是 dim 1
                occ = torch.flip(occ, dims=[1])
                occ_mid = torch.flip(occ_mid, dims=[1])

        return gt, occ, occ_mid

class RefocusDataModule(L.LightningDataModule):
    """PyTorch Lightning 数据模块，用于管理 Refocus 数据集."""
    def __init__(self, data_dir: str, model: str = None, batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        # 如果指定了 model(mode)，则只读取对应目录下的数据
        if model:
            self.data_dir = os.path.join(data_dir, model)
        else:
            self.data_dir = data_dir
            
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.h5_files = self._find_h5_files()

    def _find_h5_files(self):
        """递归查找所有 .h5 文件。"""
        h5_files = []
        if not os.path.exists(self.data_dir):
            return h5_files
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
        h5_files.sort() # 排序保证稳定性
        return h5_files

    def setup(self, stage: Optional[str] = None):
        """设置数据集划分"""
        if stage == 'fit' or stage is None:
            # 为了保证划分的随机性且可重复，先根据种子打乱列表
            # 注意：这里使用局部 Random 实例以不影响全局随机状态
            shuffled_files = self.h5_files.copy()
            random.Random(42).shuffle(shuffled_files)
            
            # 划分：前 90% 为 train，后 10% 为 val
            train_size = int(0.9 * len(shuffled_files))
            self.train_dataset = RefocusDataset(shuffled_files[:train_size], split='train')
            self.val_dataset = RefocusDataset(shuffled_files[train_size:], split='val')
        elif stage == 'test':
            self.test_dataset = RefocusDataset(self.h5_files, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)