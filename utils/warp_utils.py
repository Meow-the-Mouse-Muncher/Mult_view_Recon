"""
Refocus images using inverse geometric transformation with GPU acceleration and GT depth map.
Decoupled version for flexible dataset processing.
"""

import json
import numpy as np
import os
import cv2
import re
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import torch
from typing import Tuple, List, Dict, Optional
import tqdm

def load_depth(depth_path: str) -> Optional[np.ndarray]:
    """读取深度图文件 (单位 cm), 强制转换为 float64"""
    if not os.path.exists(depth_path):
        return None
    try:
        # EXR 文件读取，通常深度在第一个通道
        d = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        return d.astype(np.float64)
    except Exception as e:
        print(f"Error reading depth file {depth_path}: {e}")
        return None

def load_pose_info(sequence_dir: str) -> Optional[Dict]:
    """从序列目录加载相机内参和位姿信息 (使用 float64)"""
    transforms_file = os.path.join(sequence_dir, "pose", "transforms.json")
    if not os.path.exists(transforms_file):
        return None
        
    with open(transforms_file, 'r') as f:
        data = json.load(f)
        
    K = np.array([
        [data['fl_x'], 0, data['cx']],
        [0, data['fl_y'], data['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    
    frames = data['frames']
    poses = [np.array(f['transform_matrix'], dtype=np.float64) for f in frames]
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

def refocus_image_gpu(src_imgs, center_depth, src_poses, center_pose, K, device='cuda'):
    """
    GPU Batch Warping: 将一批 src_imgs 投影到 center_pose 的视角
    如果输入是单张图 src_imgs [H, W, 3], src_poses [4, 4], 会自动升维处理
    """
    # 兼容单张图片输入 (处理之前已有的调用方式)
    if src_imgs.ndim == 3:
        src_imgs = src_imgs[None, ...] # [H, W, 3] -> [1, H, W, 3]
    if src_poses.ndim == 2:
         src_poses = src_poses[None, ...] # [4, 4] -> [1, 4, 4]

    if len(src_imgs) == 0:
        return np.array([])
        
    N, H, W, _ = src_imgs.shape
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 1. 移动数据到 GPU (使用 double 精度)
    src_tensor = torch.from_numpy(src_imgs).double().permute(0, 3, 1, 2).to(device) # [N, 3, H, W]
    src_tensor /= 255.0
    
    K_tensor = torch.from_numpy(K).double().to(device)
    K_inv = torch.inverse(K_tensor)
    
    center_R = torch.from_numpy(center_pose[:3, :3]).double().to(device)
    center_T = torch.from_numpy(center_pose[:3, 3]).double().to(device)
    
    src_poses_torch = torch.from_numpy(src_poses).double().to(device)
    src_R = src_poses_torch[:, :3, :3] # [N, 3, 3]
    src_T = src_poses_torch[:, :3, 3]  # [N, 3]
    
    depth_m = -torch.from_numpy(center_depth).double().to(device) # [H, W]
    
    # 2. 计算相对变换 R_c2s, T_c2s
    # [N, 3, 3]
    rel_R = torch.bmm(src_R.transpose(1, 2), center_R.unsqueeze(0).expand(N, -1, -1))
    
    # [N, 3, 1]
    diff_T = (center_T - src_T).unsqueeze(-1) # [N, 3, 1]
    rel_T = torch.bmm(src_R.transpose(1, 2), diff_T) # [N, 3, 1]
    
    # 3. 反向投影 Center Rays (一次计算)
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    # Add 0.5 to align with pixel center
    pixels = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)], dim=-1).double() # [H, W, 3]
    pixels = pixels.reshape(-1, 3).T # [3, HW]
    
    rays_camera = K_inv @ pixels # [3, HW]
    points_3d = rays_camera * depth_m.reshape(1, -1) # [3, HW]
    
    # 4. Batch 变换
    # [N, 3, 3] @ [1, 3, HW] -> [N, 3, HW]
    points_src = torch.bmm(rel_R, points_3d.unsqueeze(0).expand(N, -1, -1)) + rel_T
    
    # 5. 投影回 Source 图像
    # [1, 3, 3] @ [N, 3, HW] -> [N, 3, HW]
    proj_src = torch.bmm(K_tensor.unsqueeze(0).expand(N, -1, -1), points_src)
    
    z = proj_src[:, 2:3, :] + 1e-6
    uv_src = proj_src[:, :2, :] / z # [N, 2, HW]
    
    # 6. Grid Sample
    # Normalize to [-1, 1]
    uv_src[:, 0] = 2.0 * uv_src[:, 0] / (W - 1) - 1.0
    uv_src[:, 1] = 2.0 * uv_src[:, 1] / (H - 1) - 1.0
    
    grid = uv_src.permute(0, 2, 1).reshape(N, H, W, 2)
    
    warped = torch.nn.functional.grid_sample(
        src_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    
    res = (warped.permute(0, 2, 3, 1) * 255.0).cpu().numpy().astype(np.uint8)
    
    # 如果原始输入是单张，则返回单张 (去掉 batch 维度)
    return res[0] if res.shape[0] == 1 else res

def get_sampling_grid(pts_3d: torch.Tensor, ref_poses: torch.Tensor, K: torch.Tensor, H: int, W: int):
    """
    计算 3D 点在参考相机平面的投影坐标 (Sampling Grid)。
    用于 Dataset 中生成 grid_sample 所需的网格。
    Args:
        pts_3d: [n_rays, 3] 世界坐标系下的 3D 点
        ref_poses: [N, 4, 4] 参考相机位姿 (c2w)
        K: [3, 3] 相机内参
        H, W: 图像尺寸
    Returns:
        sampling_grid: [N, n_rays, 2] 归一化坐标 [-1, 1], 用于 grid_sample, 最后一个维度是 (x, y)
    """
    N_ref = ref_poses.shape[0]
    n_rays = pts_3d.shape[0]
    
    # R_w2c = R_c2w^T, T_w2c = -R_c2w^T * T_c2w
    R_w2c = ref_poses[:, :3, :3].transpose(1, 2) # [N, 3, 3]
    T_w2c = -torch.bmm(R_w2c, ref_poses[:, :3, 3:4]) # [N, 3, 1]
    
    # [n_rays, 3] -> [N, 3, n_rays]
    pts_expanded = pts_3d.unsqueeze(0).expand(N_ref, -1, -1).transpose(1, 2)
    
    # 投影到相机坐标系 P_cam = R * P_world + T
    pts_cam = torch.bmm(R_w2c, pts_expanded) + T_w2c # [N, 3, n_rays]
    
    # 投影到像素平面 P_uv = K * P_cam (Homogeneous)
    K_expanded = K.unsqueeze(0).expand(N_ref, -1, -1)
    pts_uv = torch.bmm(K_expanded, pts_cam) 
    
    # Perspective Divide
    z = pts_uv[:, 2:3, :] + 1e-6
    uv = pts_uv[:, :2, :] / z # [N, 2, n_rays]
    
    # Normalize to [-1, 1] for grid_sample(align_corners=True)
    # 假设内参 K 是基于 Input Image 的 (0,0) 为 Pixel(0,0) 的左上角
    # 因此 Pixel(0,0) 的中心是 (0.5, 0.5)
    # align_corners=True 时: -1 对应 Index 0, 1 对应 Index W-1
    # 我们希望 projected u=0.5 (Index 0 Center) -> -1
    #              u=W-0.5 (Index W-1 Center) -> 1
    
    # 公式: 2 * (u - 0.5) / (W - 1) - 1
    # u=0.5 => 2*0 - 1 = -1 (Correct)
    # u=W-0.5 => 2*(W-1)/(W-1) - 1 = 1 (Correct)
    
    # uv[0] is u (x), uv[1] is v (y)
    grid_u = 2.0 * (uv[:, 0, :] - 0.5) / (W - 1) - 1.0
    grid_v = 2.0 * (uv[:, 1, :] - 0.5) / (H - 1) - 1.0
    
    sampling_grid = torch.stack([grid_u, grid_v], dim=-1) # [N, n_rays, 2]
    return sampling_grid


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
    gt_center_dep = load_depth(gt_dep_paths[center_idx])
    gt_center_dep /= 100.0 # cm -> m
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
