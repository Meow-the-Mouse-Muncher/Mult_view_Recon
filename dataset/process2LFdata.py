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
from utils.ray_utils import generate_rays, compute_pts3d
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
        
        gt_rgb = data['gt_target']['image']
        gt_depth = data['gt_target']['depth']
        gt_pose = data['gt_target']['pose']
        
        gt_rays_o, gt_rays_d = generate_rays(h, w, K, gt_pose[None, ...])
        gt_pts3d = compute_pts3d(h, w, K, gt_pose, gt_depth)
        
        ref_imgs = data['occ_ref']['images'] 
        ref_poses = data['occ_ref']['poses'] 
        ref_rays_o, ref_rays_d = generate_rays(h, w, K, ref_poses)
        
        shared_data, transforms = precompute_transforms(ref_poses, gt_pose, K)
        warped_list = []
        for i in range(len(ref_imgs)):
            warped = refocus_image_gpu(
                src_img=ref_imgs[i],
                src_depth=None,
                center_depth=gt_depth,
                frame_transform=transforms[i],
                shared_data=shared_data
            )
            warped_list.append(warped)
        occ_warped_rgb = np.stack(warped_list, axis=0) 

        occ_target_img = data['occ_target']['image']

        # 4. 保存到 H5，确保子文件夹存在
        h5_path = os.path.join(save_dir, f"{rel_prefix}.h5")
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        
        try:
            with h5py.File(h5_path, 'w') as f:
                # --- GT 组 ---
                gt_g = f.create_group('GT')
                gt_g.create_dataset('rgb', data=gt_rgb)
                gt_g.create_dataset('rays_o', data=gt_rays_o.squeeze(0))
                gt_g.create_dataset('rays_d', data=gt_rays_d.squeeze(0))

                # --- occ_ref 组 (对齐后的参考序列) ---
                ref_g = f.create_group('occ_ref')
                ref_g.create_dataset('rgb', data=occ_warped_rgb)
                ref_g.create_dataset('rays_o', data=ref_rays_o)
                ref_g.create_dataset('rays_d', data=ref_rays_d)

                # --- occ_center 组 (中心帧) ---
                center_g = f.create_group('occ_center')
                center_g.create_dataset('rgb', data=occ_target_img)

                # --- world 组 ---
                world_g = f.create_group('world')
                world_g.create_dataset('pts3d', data=gt_pts3d)
                world_g.create_dataset('K', data=K)
                
        except Exception as e:
            print(f"Error saving {h5_path}: {e}")

if __name__ == "__main__":
    DATA_ROOT = "/home_ssd/sjy/UE5_Project/PCGBiomeForestPoplar/Saved/MovieRenders/sparse_data"
    SAVE_DIR = "/home_ssd/sjy/Active_cam_recon/data"
    process_to_h5(DATA_ROOT, SAVE_DIR)
