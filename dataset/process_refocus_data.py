import os
import re
from typing import List, Dict, Optional

import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm




def create_h5_pairs(root: str, out_dir: str, overwrite: bool = False, downsample_factor: int = 1) -> int:
    """为每对 GT/occ 文件夹创建单独的 HDF5 文件，按子文件夹分组。"""
    pattern = re.compile(r"scene_(\d+).*Target_?(\d+).*height_?(\d+)_?(GT|occ)$", re.IGNORECASE)
    written = 0
    
    # 首先收集所有需要处理的pairs
    all_pairs = []
    for sub_dir in os.listdir(root):
        sub_path = os.path.join(root, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        out_sub_dir = os.path.join(out_dir, sub_dir)
        os.makedirs(out_sub_dir, exist_ok=True)
        pairs = {}
        for dirpath, dirnames, _ in os.walk(sub_path):
            for d in dirnames:
                m = pattern.search(d)
                if not m:
                    continue
                s, t, h, typ = m.group(1), m.group(2), m.group(3), m.group(4).lower()
                key = f"scene_{s}_T{t}_H{h}"
                full = os.path.join(dirpath, d)
                pairs.setdefault(key, {})[typ] = full
        
        for key, info in sorted(pairs.items()):
            if 'gt' not in info or 'occ' not in info:
                continue
            all_pairs.append((key, info, out_sub_dir))
    
    # 使用进度条处理所有pairs
    for key, info, out_sub_dir in tqdm(all_pairs, desc="Processing H5 files", unit="file"):
        gt_rgb = os.path.join(info['gt'], 'rgb')
        occ_rgb = os.path.join(info['occ'], 'rgb')
        gt_files = sorted([os.path.join(gt_rgb, x) for x in os.listdir(gt_rgb) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        occ_files = sorted([os.path.join(occ_rgb, x) for x in os.listdir(occ_rgb) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        if len(gt_files) != len(occ_files):
            continue
        imgs_gt, imgs_occ = [], []
        for g, o in zip(gt_files, occ_files):
            ig = cv2.imread(g, cv2.IMREAD_UNCHANGED)
            io = cv2.imread(o, cv2.IMREAD_UNCHANGED)
            if ig is None or io is None:
                continue
            def fix(img):
                if img.ndim == 2:
                    img = img[..., None]
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                if downsample_factor > 1:
                    img = cv2.resize(img, (img.shape[1] // downsample_factor, img.shape[0] // downsample_factor), interpolation=cv2.INTER_LINEAR)
                return img
            imgs_gt.append(fix(ig))
            imgs_occ.append(fix(io))
        if not imgs_gt or not imgs_occ:
            continue
        # GT 只保留中间那张图
        mid_idx = len(imgs_gt) // 2
        gt_img = imgs_gt[mid_idx]
        
        # occ 丢弃中间那张图，作为模型的输入
        sparse_imgs = imgs_occ[:mid_idx] + imgs_occ[mid_idx+1:]
        
        # view 取参考视角
        view_img = imgs_occ[mid_idx]  # 保持你原来的逻辑

        if not sparse_imgs:
            continue
            
        # 1. 处理 GT: (H, W, 3) -> (3, H, W)
        arr_gt = gt_img.transpose(2, 0, 1)
        # 2. 处理 参考视角: (H, W, 3) -> (3, H, W)
        arr_view = view_img.transpose(2, 0, 1)
        # 3. 处理多视角输入 (occ):
        # 先在通道维度拼接: (H, W, 3) * N -> (H, W, N*3)
        # 再转置为 CHW: (N*3, H, W)
        arr_occ = np.concatenate(sparse_imgs, axis=-1).transpose(2, 0, 1)
        out_h5 = os.path.join(out_sub_dir, f"{key}.h5")
        mode = 'w' if overwrite else 'x'
        try:
            with h5py.File(out_h5, mode) as f:
                f.create_dataset('gt', data=arr_gt)
                f.create_dataset('occ', data=arr_occ)
                f.create_dataset('view', data=arr_view)  
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset('filenames', data=np.array([os.path.basename(x) for x in gt_files], dtype=object), dtype=dt)
            written += 1
        except Exception as e:
            tqdm.write(f"Error processing {key}: {e}")
            continue
    
    return written


if __name__ == "__main__":
    root_dir = "/home_ssd/sjy/UE5_Project/PCGBiomeForestPoplar/refocus_data"
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    downsample_factor = 2  # 设置下采样倍数
    
    print(f"开始处理数据，下采样倍数: {downsample_factor}")
    print(f"输入目录: {root_dir}")
    print(f"输出目录: {out_dir}")
    
    written = create_h5_pairs(root_dir, out_dir, overwrite=True, downsample_factor=downsample_factor)
    
    print(f"\n✅ 处理完成!")
    print(f"成功写入 {written} 个 HDF5 文件到 {out_dir}")
    print(f"下采样倍数: {downsample_factor}")