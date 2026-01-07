import os
import re
from typing import List, Dict, Optional, Tuple
import multiprocessing
from functools import partial

import numpy as np
import cv2
import h5py
import torch
from tqdm import tqdm


def process_single_pair(args) -> int:
    """处理单个场景对的逻辑，用于多进程调用。"""
    # 解包参数
    key, info, out_sub_dir, overwrite, downsample_factor = args
    
    out_h5 = os.path.join(out_sub_dir, f"{key}.h5")
    
    # 检查是否跳过
    if not overwrite and os.path.exists(out_h5):
        return 0  # 已经存在且不覆盖，跳过

    gt_rgb = os.path.join(info['gt'], 'rgb')
    occ_rgb = os.path.join(info['occ'], 'rgb')
    
    # 获取文件列表
    try:
        gt_files = sorted([os.path.join(gt_rgb, x) for x in os.listdir(gt_rgb) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        occ_files = sorted([os.path.join(occ_rgb, x) for x in os.listdir(occ_rgb) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    except FileNotFoundError:
        return 0

    if len(gt_files) != len(occ_files):
        return 0

    imgs_gt, imgs_occ = [], []
    for g, o in zip(gt_files, occ_files):
        ig = cv2.imread(g, cv2.IMREAD_UNCHANGED)
        io = cv2.imread(o, cv2.IMREAD_UNCHANGED)
        if ig is None or io is None:
            continue
        
        # 内部辅助函数直接展开
        if ig.ndim == 2: ig = ig[..., None]
        elif ig.shape[2] == 3: ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        elif ig.shape[2] == 4: ig = cv2.cvtColor(ig, cv2.COLOR_BGRA2RGBA)
        
        if io.ndim == 2: io = io[..., None]
        elif io.shape[2] == 3: io = cv2.cvtColor(io, cv2.COLOR_BGR2RGB)
        elif io.shape[2] == 4: io = cv2.cvtColor(io, cv2.COLOR_BGRA2RGBA)

        if downsample_factor > 1:
            ig = cv2.resize(ig, (ig.shape[1] // downsample_factor, ig.shape[0] // downsample_factor), interpolation=cv2.INTER_LINEAR)
            io = cv2.resize(io, (io.shape[1] // downsample_factor, io.shape[0] // downsample_factor), interpolation=cv2.INTER_LINEAR)
            
        imgs_gt.append(ig)
        imgs_occ.append(io)

    if not imgs_gt or not imgs_occ:
        return 0

    # GT 只保留中间那张图
    mid_idx = len(imgs_gt) // 2
    gt_img = imgs_gt[mid_idx]
    
    # occ 丢弃中间那张图，作为模型的输入
    sparse_imgs = imgs_occ[:mid_idx] + imgs_occ[mid_idx+1:]
    
    # view 取参考视角
    view_img = imgs_occ[mid_idx]

    if not sparse_imgs:
        return 0
        
    # --- 维度处理 (直接 CHW) ---
    # 1. 处理 GT: (H, W, 3) -> (3, H, W)
    arr_gt = gt_img.transpose(2, 0, 1)
    
    # 2. 处理 参考视角: (H, W, 3) -> (3, H, W)
    arr_view = view_img.transpose(2, 0, 1)
    
    # 3. 处理多视角输入 (occ):
    # 先在通道维度拼接: (H, W, 3) * N -> (H, W, N*3)
    # 再转置为 CHW: (N*3, H, W)
    arr_occ = np.concatenate(sparse_imgs, axis=-1).transpose(2, 0, 1)

    # 写入 HDF5
    try:
        mode = 'w' if overwrite else 'x'
        with h5py.File(out_h5, mode) as f:
            f.create_dataset('gt', data=arr_gt) 
            f.create_dataset('occ', data=arr_occ)
            f.create_dataset('view', data=arr_view)  
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('filenames', data=np.array([os.path.basename(x) for x in gt_files], dtype=object), dtype=dt)
        return 1
    except Exception as e:
        # 在多进程中尽量避免 print，除非出错
        print(f"Error processing {key}: {e}")
        return 0


def create_h5_pairs(root: str, out_dir: str, overwrite: bool = False, downsample_factor: int = 1, num_workers: int = 8) -> int:
    """主函数：收集任务并进行多进程分发。"""
    pattern = re.compile(r"scene_(\d+).*Target_?(\d+).*height_?(\d+)_?(GT|occ)$", re.IGNORECASE)
    
    # 1. 扫描目录收集任务 (串行，因为很快)
    print("正在扫描目录...")
    all_pairs_info = []
    
    for sub_dir in os.listdir(root):
        sub_path = os.path.join(root, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        
        # 确保输出子目录存在
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
        
        for key, info in pairs.items():
            if 'gt' not in info or 'occ' not in info:
                continue
            # 打包参数供多进程使用
            all_pairs_info.append((key, info, out_sub_dir, overwrite, downsample_factor))
    
    print(f"找到 {len(all_pairs_info)} 个任务。开始多进程处理 (核心数: {num_workers})...")

    # 2. 多进程处理
    written_count = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 可以让完成的任务立返，配合 tqdm 显示进度
        for result in tqdm(pool.imap_unordered(process_single_pair, all_pairs_info), total=len(all_pairs_info), unit="file"):
            written_count += result
            
    return written_count


if __name__ == "__main__":
    root_dir = "/home_ssd/sjy/UE5_Project/PCGBiomeForestPoplar/refocus_data"
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    downsample_factor = 2
    
    # 根据你的 CPU 核心数自动调整，或者手动指定
    num_cpu = min(multiprocessing.cpu_count(), 16)
    
    print(f"开始处理数据，下采样倍数: {downsample_factor}")
    print(f"输入目录: {root_dir}")
    print(f"输出目录: {out_dir}")
    
    written = create_h5_pairs(root_dir, out_dir, overwrite=True, downsample_factor=downsample_factor, num_workers=num_cpu)
    
    print(f"\n✅ 处理完成!")
    print(f"成功写入 {written} 个 HDF5 文件到 {out_dir}")