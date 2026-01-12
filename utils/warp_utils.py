"""
Refocus images using inverse geometric transformation with GPU acceleration and GT depth map.
Decoupled version for flexible dataset processing.
"""

import json
import numpy as np
import os
import cv2
import re
import torch
from typing import Tuple, List, Dict, Optional
import tqdm

def load_depth(depth_path: str) -> Optional[np.ndarray]:
    """读取深度图文件 (单位 cm)"""
    if not os.path.exists(depth_path):
        return None
    try:
        # EXR 文件读取，通常深度在第一个通道
        return cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
    except Exception as e:
        print(f"Error reading depth file {depth_path}: {e}")
        return None

def load_pose_info(sequence_dir: str) -> Optional[Dict]:
    """从序列目录加载相机内参和位姿信息"""
    transforms_file = os.path.join(sequence_dir, "pose", "transforms.json")
    if not os.path.exists(transforms_file):
        return None
        
    with open(transforms_file, 'r') as f:
        data = json.load(f)
        
    K = np.array([
        [data['fl_x'], 0, data['cx']],
        [0, data['fl_y'], data['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    
    frames = data['frames']
    poses = [np.array(f['transform_matrix'], dtype=np.float32) for f in frames]
    center_idx = len(frames) // 2
    
    return {
        'K': K,
        'poses': poses,
        'center_idx': center_idx,
        'fl_x': data['fl_x'],
        'fl_y': data['fl_y'],
        'cx': data['cx'],
        'cy': data['cy']
    }

def load_GT_data(gt_dir: str) -> Optional[Dict]:
    """提取 GT 序列的信息"""
    info = load_pose_info(gt_dir)
    if info is None: return None
    
    center_idx = info['center_idx']
    # GT 我们只关心中心参考帧
    res = {
        'K': info['K'],
        'center_pose': info['poses'][center_idx],
        'center_depth_path': os.path.join(gt_dir, 'depth', f"{center_idx:04d}.exr"),
        'center_rgb_path': os.path.join(gt_dir, 'rgb', f"{center_idx:04d}.png"),
        'center_idx': center_idx
    }
    return res

def load_occ_data(occ_dir: str) -> Optional[Dict]:
    """提取 OCC 序列的信息"""
    info = load_pose_info(occ_dir)
    if info is None: return None
    
    rgb_dir = os.path.join(occ_dir, 'rgb')
    depth_dir = os.path.join(occ_dir, 'depth')
    
    # 获取有序的文件列表
    rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) 
                         if f.lower().endswith('.exr')])
    
    res = {
        'K': info['K'],
        'poses': info['poses'],
        'rgb_files': rgb_files,
        'depth_files': depth_files,
        'center_idx': info['center_idx']
    }
    return res

def precompute_transforms(poses: List[np.ndarray], center_pose: np.ndarray, K: np.ndarray):
    """预计算所有帧的变换矩阵"""
    K_inv = np.linalg.inv(K)
    R_center = center_pose[:3, :3]
    T_center = center_pose[:3, 3:4]
    
    shared_data = (K, K_inv)
    frame_transforms = []
    
    for pose in poses:
        R_src = pose[:3, :3]
        T_src = pose[:3, 3:4]
        
        # Calculate relative transform: Center -> Source (inverse direction)
        R_c2s = R_src.T @ R_center
        T_c2s = R_src.T @ (T_center - T_src)
        
        frame_transforms.append((R_c2s, T_c2s))
    
    return shared_data, frame_transforms

def refocus_image_gpu(src_img, src_depth, center_depth, frame_transform, shared_data, device='cuda'):
    """GPU加速的逆向重聚焦图像处理"""
    R_c2s, T_c2s = frame_transform
    K, K_inv = shared_data
    h, w = src_img.shape[:2]
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    src_tensor = torch.from_numpy(src_img).float().to(device)
    depth_m = -torch.from_numpy(center_depth).float().to(device)  # m, negative for -Z
    K_tensor = torch.from_numpy(K).float().to(device)
    K_inv_tensor = torch.from_numpy(K_inv).float().to(device)
    R_c2s_tensor = torch.from_numpy(R_c2s).float().to(device)
    T_c2s_tensor = torch.from_numpy(T_c2s).float().to(device)
    
    # Create pixel coordinates for center camera
    u, v = torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy')
    ones = torch.ones_like(u)
    pixels = torch.stack([u.flatten(), v.flatten(), ones.flatten()], dim=0).float()
    
    # Unproject to rays in center camera coordinate system
    rays_center = (K_inv_tensor @ pixels).reshape(3, h, w)  # [3, H, W]
    
    # Apply per-pixel depth and transform to source camera
    # points_3d_center = rays_center * depth
    points_3d_center = rays_center * depth_m.unsqueeze(0)
    
    # Transform to source camera: R_c2s @ points_3d_center + T_c2s
    transformed_points = R_c2s_tensor @ points_3d_center.reshape(3, -1) + T_c2s_tensor
    transformed_points = transformed_points.reshape(3, h, w)
    
    # Project to source camera image plane
    projected = K_tensor @ transformed_points.reshape(3, -1)
    projected = projected.reshape(3, h, w)
    
    # Perspective division
    z = projected[2, :, :] + 1e-6
    x_coords = projected[0, :, :] / z
    y_coords = projected[1, :, :] / z
     
    # Use GPU grid sampling for remapping
    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2.0 * x_coords / (w - 1) - 1.0
    y_norm = 2.0 * y_coords / (h - 1) - 1.0
    
    # Create sampling grid
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    src_tensor_norm = src_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # [1, 3, H, W]
    # Sample color and depth from source
    sampled_color = torch.nn.functional.grid_sample(
        src_tensor_norm, grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    result = sampled_color.squeeze(0).permute(1, 2, 0) * 255.0  # [H, W, 3]
    return result.cpu().numpy().astype(np.uint8)



def recenter_poses(poses):
    """Recenter poses according to the original NeRF code.
    Input: poses [N, 4, 4]
    """
    poses_ = poses.copy()
    c2w = poses_avg(poses) # [3, 4]
    
    # 构建 4x4 的平均位姿矩阵并求逆
    c2w_homo = np.eye(4)
    c2w_homo[:3, :4] = c2w
    inv_c2w = np.linalg.inv(c2w_homo)
    
    # 将所有位姿变换到平均位姿坐标系下: inv(c2w_avg) @ poses
    # 注意：这里 poses 是 [N, 4, 4]，可以直接矩阵乘法
    poses_avg_space = inv_c2w @ poses_ # [N, 4, 4]
    return poses_avg_space

def poses_avg(poses):
    """Average poses. Input: [N, 4, 4]"""
    # 获取所有相机的中心点
    center = poses[:, :3, 3].mean(0)
    # 获取 Z 轴 (forward) 的平均方向
    vec2 = normalize(poses[:, :3, 2].sum(0))
    # 获取 Y 轴 (up) 的平均方向
    up = poses[:, :3, 1].sum(0)
    # 构造新的观察矩阵
    c2w = viewmatrix(vec2, up, center)
    return c2w

def normalize(x):
    """Normalization helper function."""
    return x / (np.linalg.norm(x) + 1e-10)

def viewmatrix(z, up, pos):
    """Construct lookat view matrix."""
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def load_render_data(gt_dir: str, occ_dir: str) -> Optional[Dict]:
    """
    高度解耦的加载函数，直接返回数值：
    1. K: 相机内参 [3, 3]
    2. occ_ref: 去除中心帧后的图像序列和位姿 [N-1, H, W, 3], [N-1, 4, 4]
    3. gt_target: 中心帧的 GT 图像和深度
    4. occ_target: 中心帧的 OCC 图像
    """
    # 内部辅助函数
    def read_rgb(path):
        if not os.path.exists(path): return None
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def get_paths(d):
        rgb_d = os.path.join(d, 'rgb')
        dep_d = os.path.join(d, 'depth')
        rgbs = sorted([os.path.join(rgb_d, f) for f in os.listdir(rgb_d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        deps = sorted([os.path.join(dep_d, f) for f in os.listdir(dep_d) if f.lower().endswith('.exr')])
        return rgbs, deps

    # 1. 加载元数据和路径
    occ_info = load_pose_info(occ_dir)
    if not occ_info: return None
    
    K = occ_info['K']
    poses = np.array(occ_info['poses']) # 转换成 [N, 4, 4] NumPy 数组
    center_idx = occ_info['center_idx']
    
    occ_rgb_paths, _ = get_paths(occ_dir)
    gt_rgb_paths, gt_dep_paths = get_paths(gt_dir)

    # 2. 处理深度图和缩放因子
    gt_center_dep = load_depth(gt_dep_paths[center_idx])/100 # cm -> m
    if gt_center_dep is None:
        return None
        
    mindep = gt_center_dep.min() 
    scale = 2.0 / (mindep + 1e-6)
    
    # 对位姿进行缩放 (平移部分)
    poses[:, :3, 3] *= scale
    # 对深度图进行缩放
    gt_center_dep *= scale
    
    # 3. 对位姿进行中心化 (将平均坐标系对齐到原点)
    poses = recenter_poses(poses)

    # 4. 加载图像数据
    # --- 处理 gt_target (中心帧) ---
    gt_center_img = read_rgb(gt_rgb_paths[center_idx])
    
    # --- 处理 occ_target (中心帧) ---
    occ_center_img = read_rgb(occ_rgb_paths[center_idx])

    # --- 处理 occ_ref (加载非中心帧序列) ---
    occ_imgs_list = []
    occ_poses_list = []
    for i in range(len(occ_rgb_paths)):
        if i == center_idx:
            continue
        img = read_rgb(occ_rgb_paths[i])
        if img is not None:
            occ_imgs_list.append(img)
            occ_poses_list.append(poses[i])

    # 转换列表为 NumPy 数组进行聚合
    occ_ref_images = np.stack(occ_imgs_list, axis=0) if occ_imgs_list else np.array([])
    occ_ref_poses = np.stack(occ_poses_list, axis=0) if occ_poses_list else np.array([])

    return {
        'K': K,
        'occ_ref': {
            'images': occ_ref_images,  # [N-1, H, W, 3]
            'poses': occ_ref_poses,    # [N-1, 4, 4]
        },
        'gt_target': {
            'image': gt_center_img,    # [H, W, 3]
            'depth': gt_center_dep,    # [H, W]
            'pose': poses[center_idx]  # [4, 4]
        },
        'occ_target': {
            'image': occ_center_img,   # [H, W, 3]
            'pose': poses[center_idx]  # [4, 4]
        }
    }
