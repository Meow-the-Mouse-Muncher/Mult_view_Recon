import sys
import os
import re
from typing import List, Tuple
from functools import partial
from utils.warp_utils import (
    load_render_data, 
    precompute_transforms, 
    refocus_image_gpu
)
from utils.ray_utils import (
    generate_rays, compute_pts3d
)
import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm

def find_sequence_pairs(root_dir: str) -> List[Tuple[str, str, str]]:
    """
    根据新的命名规则整理 GT 和 OCC 序列对，并保留子文件夹结构
    命名规则: rel_path/scene_xxx_Target_xxx_height_xxx_ang_xxx_occ/GT
    返回: [(rel_prefix, gt_dir, occ_dir), ...]
    """
    pattern = re.compile(r"(scene_\d+_Target_\d+_height_\d+_ang_\d+)_(GT|occ)$", re.IGNORECASE)
    all_pairs = []
    
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} does not exist.")
        return []

    # 遍历子文件夹，如 fix_line/
    for sub_dir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        
        scene_groups = {}
        # 遍历子文件夹内的序列目录
        for d in os.listdir(sub_path):
            match = pattern.search(d)
            if match:
                prefix = match.group(1)
                typ = match.group(2).lower()
                full_path = os.path.join(sub_path, d)
                scene_groups.setdefault(prefix, {})[typ] = full_path
        
        for prefix, paths in scene_groups.items():
            if 'gt' in paths and 'occ' in paths:
                # rel_prefix 包含子文件夹名，例如 "fix_line/scene_007_..."
                rel_prefix = os.path.join(sub_dir, prefix)
                all_pairs.append((rel_prefix, paths['gt'], paths['occ']))
    return all_pairs

def process_to_h5(root_dir, save_dir):
    """
    将数据处理并保存为 HDF5
    """
    os.makedirs(save_dir, exist_ok=True)
    pairs = find_sequence_pairs(root_dir)
    print(f"Found {len(pairs)} pairs. Starting H5 conversion...")

    for rel_prefix, gt_dir, occ_dir in tqdm(pairs, desc="H5 Processing"):
        data = load_render_data(gt_dir, occ_dir)
        if data is None: continue
        
        K = data['K']
        h, w = data['gt_target']['image'].shape[:2]
        
        gt_rgb = data['gt_target']['image'][None, ...]
        gt_depth = data['gt_target']['depth']
        gt_pose = data['gt_target']['pose']
        
        # GPU 计算光线和 3D 点
        K_torch = torch.from_numpy(K).float().cuda()
        gt_pose_torch = torch.from_numpy(gt_pose).float().cuda()
        gt_depth_torch = torch.from_numpy(gt_depth).float().cuda()
        
        # [1, 4, 4]
        gt_rays_o_torch, gt_rays_d_torch = generate_rays(h, w, K_torch, gt_pose_torch[None, ...])
        gt_pts3d_torch = compute_pts3d(h, w, K_torch, gt_pose_torch, gt_depth_torch)
        
        # 转回 CPU 存 H5
        gt_rays_o = gt_rays_o_torch.cpu().numpy()
        gt_rays_d = gt_rays_d_torch.cpu().numpy()
        gt_pts3d = gt_pts3d_torch.cpu().numpy()[None, ...]
        
        ref_imgs = data['occ_ref']['images'] 
        ref_poses = data['occ_ref']['poses'] 
        
        # GPU 批量 warp 图像
        if len(ref_imgs) > 0:
            occ_warped_rgb = refocus_image_gpu(
                src_imgs=ref_imgs,
                center_depth=gt_depth,
                src_poses=ref_poses,
                center_pose=gt_pose,
                K=K,
                device='cuda'
            )
            
            # GPU 计算 Ref Rays
            ref_poses_torch = torch.from_numpy(ref_poses).float().cuda()
            ref_rays_o_torch, ref_rays_d_torch = generate_rays(h, w, K_torch, ref_poses_torch)
            ref_rays_o = ref_rays_o_torch.cpu().numpy()
            ref_rays_d = ref_rays_d_torch.cpu().numpy()
            
        else:
            occ_warped_rgb = np.array([])
            ref_rays_o = np.array([])
            ref_rays_d = np.array([])

        occ_target_img = data['occ_target']['image'][None, ...]

        # 4. 保存到 H5，确保子文件夹存在
        h5_path = os.path.join(save_dir, f"{rel_prefix}.h5")
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        
        try:
            with h5py.File(h5_path, 'w') as f:
                # --- GT 组 ---
                gt_g = f.create_group('GT')
                gt_g.create_dataset('rgb', data=gt_rgb) # 1 H W 3
                gt_g.create_dataset('rays_o', data=gt_rays_o) # 1 H W 3
                gt_g.create_dataset('rays_d', data=gt_rays_d) # 1 H W 3

                # --- occ_ref 组 (对齐后的参考序列) ---
                ref_g = f.create_group('occ_ref')
                ref_g.create_dataset('rgb', data=occ_warped_rgb)   # 32 H W 3
                ref_g.create_dataset('rays_o', data=ref_rays_o)    # 32 H W 3
                ref_g.create_dataset('rays_d', data=ref_rays_d)    # 32 H W 3

                # --- occ_center 组 (中心帧) ---
                center_g = f.create_group('occ_center')
                center_g.create_dataset('rgb', data=occ_target_img) # 1 H W 3

                # --- world 组 ---
                world_g = f.create_group('world')
                world_g.create_dataset('pts3d', data=gt_pts3d) # 1 H W 3
                world_g.create_dataset('K', data=K)
                
        except Exception as e:
            print(f"Error saving {h5_path}: {e}")

if __name__ == "__main__":
    DATA_ROOT = "/home_ssd/sjy/UE5_Project/PCGBiomeForestPoplar/Saved/MovieRenders/sparse_data"
    SAVE_DIR = "/home_ssd/sjy/Active_cam_recon/data"
    process_to_h5(DATA_ROOT, SAVE_DIR)
