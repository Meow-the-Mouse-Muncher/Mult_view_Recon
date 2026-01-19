import sys
import os
import re
from typing import List, Tuple
from functools import partial
from utils.warp_utils import load_render_data
import numpy as np
import cv2
import h5py
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
    只保存图像、深度图和位姿，光线在训练时动态生成
    """
    os.makedirs(save_dir, exist_ok=True)
    pairs = find_sequence_pairs(root_dir)
    print(f"Found {len(pairs)} pairs. Starting H5 conversion...")

    for rel_prefix, gt_dir, occ_dir in tqdm(pairs, desc="H5 Processing"):
        # 0. 检查目标文件是否已存在
        h5_path = os.path.join(save_dir, f"{rel_prefix}.h5")
        if os.path.exists(h5_path):
            # print(f"Skipping {h5_path} (exists)")
            continue

        data = load_render_data(gt_dir, occ_dir)
        if data is None: continue
        
        K = data['K']
        h, w = data['gt_target']['image'].shape[:2]
        
        gt_rgb = data['gt_target']['image']
        gt_depth = data['gt_target']['depth'] # 保存 Depth，shape [H, W]
        gt_pose = data['gt_target']['pose']   # 保存 Pose, shape [4, 4]
        
        # 不再生成 rays 和 pts3d
        
        ref_imgs = data['occ_ref']['images'] 
        ref_poses = data['occ_ref']['poses'] 
        
        # 移除预处理阶段的 Warping/Refocus，改为训练时在线采样
        # 这样节省存储空间，且避免二次插值
        
        occ_target_img = data['occ_target']['image']

        # 4. 保存到 H5
        h5_path = os.path.join(save_dir, f"{rel_prefix}.h5")
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        
        try:
            with h5py.File(h5_path, 'w') as f:
                # --- GT 组 ---
                gt_g = f.create_group('GT')
                gt_g.create_dataset('rgb', data=gt_rgb) 
                gt_g.create_dataset('pose', data=gt_pose)   # 4 4 (float64)
                gt_g.create_dataset('depth', data=gt_depth) # H W (float64)

                # --- occ_ref 组 ---
                ref_g = f.create_group('occ_ref')
                # 保存原始 RGB 图像序列 (Original Images)，训练时使用 Grid Sample 获取特征
                ref_g.create_dataset('rgb', data=ref_imgs)
                # 不存 rays，只存 poses
                ref_g.create_dataset('poses', data=ref_poses) 

                # --- occ_center 组 (中心帧) ---
                center_g = f.create_group('occ_center')
                center_g.create_dataset('rgb', data=occ_target_img) 

                # --- world 组 ---
                world_g = f.create_group('world')
                # 不存 dense pts3d，如果需要可用 depth + K + pose 恢复
                world_g.create_dataset('K', data=K)
                
        except Exception as e:
            print(f"Error saving {h5_path}: {e}")
            # 如果失败删除半成品
            if os.path.exists(h5_path):
                os.remove(h5_path)

if __name__ == "__main__":
    DATA_ROOT = "/home_ssd/sjy/UE5_Project/PCGBiomeForestPoplar/Saved/MovieRenders/test_data"
    SAVE_DIR = "/home_ssd/sjy/Active_cam_recon/data/test_data" # 建议换个目录或者增加后缀以区分
    process_to_h5(DATA_ROOT, SAVE_DIR)
