import os
import h5py
import torch
import random
from torch.utils.data import Dataset, DataLoader
from utils.ray_utils import generate_rays, compute_pts3d
from utils.warp_utils import get_sampling_grid
import lightning as L
from typing import Optional
class LFDataset(Dataset):
    """从 HDF5 文件加载 LF 数据的数据集类。"""
    def __init__(self, h5_files: list, split: str = 'train', n_rays: int = 4096):
        self.h5_files = h5_files
        self.split = split
        self.n_rays = n_rays

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_file = self.h5_files[idx]
        with h5py.File(h5_file, 'r') as f:
            # 1. 读取基础数据 (Keep everything in Float32 for training)
            # GT Data
            gt_rgb = torch.from_numpy(f['GT/rgb'][:]).float() / 255.0 # [H, W, 3] -> will be sampled
            gt_pose = torch.from_numpy(f['GT/pose'][:]).float()
            gt_depth = torch.from_numpy(f['GT/depth'][:]).float() # [H, W]
            
            # Ref Data
            occ_rgb = torch.from_numpy(f['occ_ref/rgb'][:]).float() / 255.0  # [N, H, W, 3]
            occ_poses = torch.from_numpy(f['occ_ref/poses'][:]).float()      # [N, 4, 4]
            occ_center_rgb = torch.from_numpy(f['occ_center/rgb'][:]).float() / 255.0 # [H, W, 3] (Not used here)
            # Camera
            K = torch.from_numpy(f['world/K'][:]).float()
            
            H, W = gt_depth.shape[:2]

            if self.split == 'train':
                # --- 2. 随机光线采样 ---
                total_pixels = H * W
                select_inds = torch.randperm(total_pixels)[:self.n_rays]
                y = select_inds // W
                x = select_inds % W
                
                # --- 3. GT 侧数据准备 ---
                # Sample Pixel Colors & Depth
                sampled_rgb = gt_rgb[y, x]       # [n_rays, 3]
                sampled_depth = gt_depth[y, x]   # [n_rays]
                
                # Compute Rays & 3D Points
                # ray_utils 内部会自动加 0.5，这里传入整数索引即可
                pixel_coords = torch.stack([x, y], dim=-1).float() # [n_rays, 2]
                gt_rays_o, gt_rays_d = generate_rays(H, W, K, gt_pose, coords=pixel_coords)
                pts_3d = compute_pts3d(H, W, K, gt_pose, sampled_depth, coords=pixel_coords) # [n_rays, 3]
                
                # --- 4. Ref 侧数据准备 (核心: Sampling Grid) ---
                # Goal: Find (u, v) in Ref images corresponding to pts_3d
                # 使用 warp_utils 中的工具函数封装计算逻辑，保持 Dataset 简洁
                sampling_grid = get_sampling_grid(pts_3d, occ_poses, K, H, W) # [N, n_rays, 2]
                
                # --- 5. Ref 几何信息 ---
                # Ray direction from Ref Center -> Point
                ref_cam_pos = occ_poses[:, :3, 3].unsqueeze(1).expand(-1, self.n_rays, 3) # [N, n_rays, 3]
                diff = pts_3d.unsqueeze(0) - ref_cam_pos # [1, n_rays, 3] - [N, n_rays, 3] -> [N, n_rays, 3]
                occ_rays_d = torch.nn.functional.normalize(diff, p=2, dim=-1)

                batch = {
                    'gt_rgb': sampled_rgb,           # [n_rays, 3]
                    'gt_rays_o': gt_rays_o,          # [n_rays, 3]
                    'gt_rays_d': gt_rays_d,          # [n_rays, 3]
                    'pts_3d': pts_3d,                # [n_rays, 3]
                    
                    # Ref Inputs
                    # occ_rgb: [N, 3, H, W] for CNN extraction
                    'occ_rgb': occ_rgb.permute(0, 3, 1, 2), 
                    # sampling_grid: [N, n_rays, 2] for sampling features/colors
                    'sampling_grid': sampling_grid,
                    
                    'occ_rays_o': ref_cam_pos,       # [N, n_rays, 3]
                    'occ_rays_d': occ_rays_d,        # [N, n_rays, 3]
                    'K': K
                }

            else:
                # Validation/Test logic: Use full image
                # 为了防止显存爆炸，我们不在 Dataset 里全图采样，而是只提供必要信息，
                # 让 Model 在 validation_step 自行分块 (chunk) 处理或全图处理。
                # 但考虑到逻辑简单性，这里先返回全图采样点，在 LightningModule 做 chunk。
                
                # 生成网格坐标 [H, W, 2] -> [H*W, 2]
                pixel_coords = torch.stack(torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy'
                ), dim=-1).reshape(-1, 2).float() # [H*W, 2]
                
                # --- GT Information ---
                flat_depth = gt_depth.reshape(-1) # [H*W]
                
                # 生成射线
                gt_rays_o, gt_rays_d = generate_rays(H, W, K, gt_pose, coords=pixel_coords) # [H*W, 3]
                
                # Compute 3D points
                pts_3d = compute_pts3d(H, W, K, gt_pose, flat_depth, coords=pixel_coords) # [H*W, 3]
                
                # Compute Sampling Grid for Ref Images
                sampling_grid = get_sampling_grid(pts_3d, occ_poses, K, H, W) # [N, H*W, 2]
                
                # Ref 几何信息
                ref_cam_pos = occ_poses[:, :3, 3].unsqueeze(1).expand(-1, H*W, 3) # [N, H*W, 3]
                diff = pts_3d.unsqueeze(0) - ref_cam_pos 
                occ_rays_d = torch.nn.functional.normalize(diff, p=2, dim=-1)

                batch = {
                    'gt_rgb': gt_rgb.reshape(-1, 3),    # [H*W, 3]
                    'gt_rays_o': gt_rays_o,             # [H*W, 3] (Previously Missing!)
                    'gt_rays_d': gt_rays_d,             # [H*W, 3] (Previously Missing!)
                    'pts_3d': pts_3d,                   # [H*W, 3]
                    
                    'H': torch.tensor(H), # 存为 Tensor 方便 batch 解包
                    'W': torch.tensor(W),
                    'occ_center_rgb': occ_center_rgb, # [H, W, 3] (Not used here)
                    
                    'occ_rgb': occ_rgb.permute(0, 3, 1, 2), # [N, 3, H, W]
                    'sampling_grid': sampling_grid,         # [N, H*W, 2]
                    'occ_rays_o': ref_cam_pos,              # [N, H*W, 3]
                    'occ_rays_d': occ_rays_d,               # [N, H*W, 3]
                    'K': K
                }
            
        return batch

class LFDataModule(L.LightningDataModule):
    """PyTorch Lightning 数据模块，用于管理 LF 数据集."""
    def __init__(self, data_dir: str, model: str = None, batch_size: int = 4, num_workers: int = 4, n_rays: int = 4096):
        super().__init__()
        # 如果指定了 model(mode)，则只读取对应目录下的数据
        if model:
            self.data_dir = os.path.join(data_dir, model)
        else:
            self.data_dir = data_dir
            
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_rays = n_rays
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
            self.train_dataset = LFDataset(shuffled_files[:train_size], split='train', n_rays=self.n_rays)
            self.val_dataset = LFDataset(shuffled_files[train_size:], split='val', n_rays=self.n_rays)
        elif stage == 'test':
            self.test_dataset = LFDataset(self.h5_files, split='test', n_rays=self.n_rays)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)