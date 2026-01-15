import os
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.ray_utils import generate_rays, compute_pts3d
from utils.warp_utils import get_sampling_grid
import lightning as L
from typing import Optional

class LFDataset(Dataset):
    """从 HDF5 文件加载 LF 数据的数据集类。"""
    def __init__(self, h5_files: list, split: str = 'train', n_rays: int = 4096, val_chunk_size: int = 4096):
        self.h5_files = h5_files
        self.split = split
        self.n_rays = n_rays
        self.val_chunk_size = val_chunk_size # 使用传入的参数
        
        # === 验证集切片配置 ===
        self.val_meta = [] # 存储切块索引信息: (file_idx, start, end)

        # 只有非训练集才需要预计算切片
        if self.split != 'train':
            self._precompute_val_chunks()

    def _precompute_val_chunks(self):
        """预计算验证集的切块索引，将每张大图拆成多个小块"""
        print(f"[{self.split}] 正在预处理切块信息...")
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                # 只读形状，不读数据，速度很快
                H, W = f['GT/depth'].shape
                total_pixels = H * W
                
            # 将一张图拆分成多个 chunk
            for start in range(0, total_pixels, self.val_chunk_size):
                end = min(start + self.val_chunk_size, total_pixels)
                # 记录这块数据属于哪张图、从哪开始、到哪结束
                self.val_meta.append({
                    "file_idx": file_idx,
                    "start": start,
                    "end": end,
                    "H": H, "W": W,
                    "is_last_chunk": (end == total_pixels) # 标记是否是该图的最后一块
                })
        print(f"[{self.split}] 预处理完成: {len(self.h5_files)} 张图像被拆分为 {len(self.val_meta)} 个验证块。")

    def __len__(self):
        # 训练集长度 = 图片数 (随机采样不拆分)
        if self.split == 'train':
            return len(self.h5_files)
        # 验证集长度 = 切块总数 (所以验证步数会变多)
        else:
            return len(self.val_meta)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self._get_train_item(self.h5_files[idx])
        else:
            return self._get_val_chunk(idx)

    # ==========================
    #      训练逻辑 (抽离)
    # ==========================
    def _get_train_item(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            # 1. 读取基础数据
            gt_rgb = torch.from_numpy(f['GT/rgb'][:]).float() / 255.0
            gt_pose = torch.from_numpy(f['GT/pose'][:]).float()
            gt_depth = torch.from_numpy(f['GT/depth'][:]).float()
            
            # Ref Data
            occ_rgb = torch.from_numpy(f['occ_ref/rgb'][:]).float() / 255.0
            occ_poses = torch.from_numpy(f['occ_ref/poses'][:]).float()
            K = torch.from_numpy(f['world/K'][:]).float()
            
            H, W = gt_depth.shape[:2]
            
            # 2. 随机光线采样
            total_pixels = H * W
            select_inds = torch.randperm(total_pixels)[:self.n_rays]
            y = select_inds // W
            x = select_inds % W
            
            # 3. 采样数据
            sampled_rgb = gt_rgb[y, x]
            sampled_depth = gt_depth[y, x]
            
            # 4. 生成射线和 3D 点
            pixel_coords = torch.stack([x, y], dim=-1).float()
            gt_rays_o, gt_rays_d = generate_rays(H, W, K, gt_pose, coords=pixel_coords)
            pts_3d = compute_pts3d(H, W, K, gt_pose, sampled_depth, coords=pixel_coords)
            
            # 5. Grid 和 Ref 几何
            sampling_grid = get_sampling_grid(pts_3d, occ_poses, K, H, W)
            
            ref_cam_pos = occ_poses[:, :3, 3].unsqueeze(1).expand(-1, self.n_rays, 3)
            diff = pts_3d.unsqueeze(0) - ref_cam_pos
            occ_rays_d = torch.nn.functional.normalize(diff, p=2, dim=-1)

            return {
                'gt_rgb': sampled_rgb,           
                'gt_rays_o': gt_rays_o,          
                'gt_rays_d': gt_rays_d,          
                'pts_3d': pts_3d,                
                'occ_rgb': occ_rgb.permute(0, 3, 1, 2), 
                'sampling_grid': sampling_grid,
                'occ_rays_o': ref_cam_pos,       
                'occ_rays_d': occ_rays_d,        
                'K': K
            }

    # ==========================
    #    验证/测试切块逻辑 (新增)
    # ==========================
    def _get_val_chunk(self, idx):
        meta = self.val_meta[idx]
        file_idx = meta['file_idx']
        start = meta['start']
        end = meta['end']
        current_chunk_size = end - start
        
        h5_file = self.h5_files[file_idx]
        
        with h5py.File(h5_file, 'r') as f:
            gt_depth = torch.from_numpy(f['GT/depth'][:]).float()
            H, W = gt_depth.shape
            
            # 1. 生成全图坐标并切取当前 chunk
            #    (先生成全图坐标更方便对齐，显存占用极小)
            #    注意 meshgrid indexing='xy' 还是 'ij' 要看 utils 里的实现，通常是 xy 对应 u,v
            pixel_coords_full = torch.stack(torch.meshgrid(
                torch.arange(W), torch.arange(H), indexing='xy'
            ), dim=-1).reshape(-1, 2).float() # [H*W, 2]
            
            chunk_coords = pixel_coords_full[start:end] # [chunk, 2]
            
            # 2. 读取其他元数据
            K = torch.from_numpy(f['world/K'][:]).float()
            gt_pose = torch.from_numpy(f['GT/pose'][:]).float()
            occ_rgb = torch.from_numpy(f['occ_ref/rgb'][:]).float() / 255.0
            occ_poses = torch.from_numpy(f['occ_ref/poses'][:]).float()
            
            # 3. 根据 chunk 坐标生成射线
            #    这样只生成这几千条射线，显存无压力
            gt_rays_o, gt_rays_d = generate_rays(H, W, K, gt_pose, coords=chunk_coords)
            
            # 4. 深度与 3D 点
            flat_depth = gt_depth.reshape(-1)
            chunk_depth = flat_depth[start:end]
            pts_3d = compute_pts3d(H, W, K, gt_pose, chunk_depth, coords=chunk_coords)
            
            # 5. 生成 Sampling Grid
            sampling_grid = get_sampling_grid(pts_3d, occ_poses, K, H, W)
            
            # 6. Ref 几何
            ref_cam_pos = occ_poses[:, :3, 3].unsqueeze(1).expand(-1, current_chunk_size, 3)
            diff = pts_3d.unsqueeze(0) - ref_cam_pos
            occ_rays_d = torch.nn.functional.normalize(diff, p=2, dim=-1)
            
            # 7. GT RGB 切片
            gt_rgb_full = torch.from_numpy(f['GT/rgb'][:]).float() / 255.0
            gt_rgb_chunk = gt_rgb_full.reshape(-1, 3)[start:end]
            
            # 8. 读取中心图 (仅最后一块需要，节省IO)
            occ_center_rgb = torch.zeros(1)
            if meta['is_last_chunk']:
                # 尝试读取，兼容老数据
                if 'occ_center' in f:
                    occ_center_rgb = torch.from_numpy(f['occ_center/rgb'][:]).float() / 255.0

        return {
            'gt_rgb': gt_rgb_chunk,         # [chunk, 3]
            'gt_rays_o': gt_rays_o,         
            'gt_rays_d': gt_rays_d,         
            'pts_3d': pts_3d,               
            'occ_rgb': occ_rgb.permute(0, 3, 1, 2), 
            'sampling_grid': sampling_grid,         
            'occ_rays_o': ref_cam_pos,              
            'occ_rays_d': occ_rays_d,               
            'K': K,
            # 仅返回索引信息，不返回 Center 全图
            'meta_file_idx': torch.tensor(file_idx),
            'meta_is_last': torch.tensor(meta['is_last_chunk']),
            'meta_start': torch.tensor(start), 
        }

class LFDataModule(L.LightningDataModule):
    """PyTorch Lightning 数据模块，用于管理 LF 数据集."""
    def __init__(self, data_dir: str, train_data_dir: Optional[str] = None, test_data_dir: Optional[str] = None, model: str = None, batch_size: int = 4, num_workers: int = 4, n_rays: int = 4096, val_chunk_size: int = 4096):
        super().__init__()
        
        # 1. 确定基础目录
        # 如果从外部传入了特定路径，就用传入的；否则基于 data_dir 自动推断
        _train_base = train_data_dir if train_data_dir else os.path.join(data_dir, "train_data")
        _test_base = test_data_dir if test_data_dir else os.path.join(data_dir, "test_data")

        # 2. 如果指定了 model (即 mode, 如 "rot_arc")，则进入子目录
        # 这样避免读取到其他 mode 的数据
        if model:
            self.train_data_dir = os.path.join(_train_base, model)
            self.test_data_dir = os.path.join(_test_base, model)
        else:
            self.train_data_dir = _train_base
            self.test_data_dir = _test_base
            
        print(f"Dataset Config:")
        print(f"  - Train Dir: {self.train_data_dir}")
        print(f"  - Test  Dir: {self.test_data_dir}")
            
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_rays = n_rays
        self.val_chunk_size = val_chunk_size
        # 移除 self.h5_files，因为现在按 stage 动态查找

    def _find_h5_files_in_dir(self, dir_path: str):
        """在指定目录递归查找所有 .h5 文件。"""
        h5_files = []
        if not os.path.exists(dir_path):
            return h5_files
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
        h5_files.sort()  # 排序保证稳定性
        return h5_files

    def setup(self, stage: Optional[str] = None):
        """设置数据集划分"""
        if stage == 'fit' or stage is None:
            # 使用 train_data_dir 查找训练和验证文件
            train_val_files = self._find_h5_files_in_dir(self.train_data_dir)
            # 为了保证划分的随机性且可重复，先根据种子打乱列表
            shuffled_files = train_val_files.copy()
            random.Random(42).shuffle(shuffled_files)
            
            # 划分：前 99.5% 为 train，后 0.5% 为 val
            train_size = int(0.995 * len(shuffled_files))
            self.train_dataset = LFDataset(shuffled_files[:train_size], split='train', n_rays=self.n_rays)
            self.val_dataset = LFDataset(shuffled_files[train_size:], split='val', n_rays=self.n_rays, val_chunk_size=self.val_chunk_size)
        elif stage == 'test':
            # 使用 test_data_dir 查找测试文件
            test_files = self._find_h5_files_in_dir(self.test_data_dir)
            self.test_dataset = LFDataset(test_files, split='test', n_rays=self.n_rays, val_chunk_size=self.val_chunk_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # 验证集 batch_size=1, shuffle=True! 这里预览看到更多的数据
        # 因为我们已经手动 chunk 了，每个 Item 就是一个 Batch
        return DataLoader(self.val_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)