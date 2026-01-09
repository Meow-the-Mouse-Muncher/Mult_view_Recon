"""
Refocus images using inverse geometric transformation with GPU acceleration and GT depth map
"""

import json
import numpy as np
import sys
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch

def load_depth(depth_path):
    """读取深度图文件，单位是cm"""
    if not os.path.exists(depth_path):
        return None
    
    try:
        return cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
    except Exception as e:
        print(f"Error reading depth file {depth_path}: {e}")
        return None

def precompute_transforms(poses, center_pose, K):
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
    """GPU加速的逆向重聚焦图像处理，使用GT深度图和深度剔除"""
    R_c2s, T_c2s = frame_transform
    K, K_inv = shared_data
    h, w = src_img.shape[:2]
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    src_tensor = torch.from_numpy(src_img).float().to(device)
    # z_measured = torch.from_numpy(src_depth).float().to(device) / 100.0  # cm->m
    depth_m = -torch.from_numpy(center_depth).float().to(device) / 100.0  # cm->m, negative for -Z
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

    4



    x_coords = projected[0, :, :] / z
    y_coords = projected[1, :, :] / z
    
    # Get projected depth (Z_proj) - distance from source camera
    # z_proj = -transformed_points[2, :, :]   
    
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
    
    # Depth : Z_proj < Z_measured  这种情况应该保留：树林遮住了目标物
    #         Z_proj > Z_measured  这种情况应该剔除：侧视图的遮挡覆盖了目标物

    # depth_mask = z_proj < z_measured  # Keep pixels where projected depth >= measured depth
    
    # Apply depth mask
    result = sampled_color.squeeze(0).permute(1, 2, 0) * 255.0  # [H, W, 3]
    # result[~depth_mask] = 0  # Set occluded pixels to black
    
    return result.cpu().numpy().astype(np.uint8)



def process_dataset(transforms_file, rgb_dir, gt_depth_dir, src_depth_dir, output_dir, use_gpu=True):
    """处理数据集"""
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    if use_gpu and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        use_gpu = False
    
    # Load transforms and setup
    with open(transforms_file, 'r') as f:
        pose_data = json.load(f)
    
    K = np.array([
        [pose_data['fl_x'], 0, pose_data['cx']],
        [0, pose_data['fl_y'], pose_data['cy']],
        [0, 0, 1]
    ])
    
    frames = pose_data['frames']
    center_idx = len(frames) // 2
    center_pose = np.array(frames[center_idx]['transform_matrix'])
    poses = [np.array(frame['transform_matrix']) for frame in frames]
    
    # Load center depth map from GT depth directory
    center_depth_path = os.path.join(gt_depth_dir, f"{center_idx:04d}.exr")
    center_depth = load_depth(center_depth_path)
    if center_depth is None:
        print(f"Failed to load center depth map: {center_depth_path}")
        return 0, 0
    
    shared_data, frame_transforms = precompute_transforms(poses, center_pose, K)
    os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
    
    # GPU processing - sequential
    processed_count = 0
    for i, frame_transform in enumerate(frame_transforms):
        rgb_path = os.path.join(rgb_dir, f"{i:04d}.png")
        output_path = os.path.join(output_dir, 'rgb', f"{i:04d}.png")
        
        if i == center_idx:
            # Copy center frame
            if os.path.exists(rgb_path):
                rgb_img = cv2.imread(rgb_path)
                if rgb_img is not None:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    if cv2.imwrite(output_path, rgb_img):
                        processed_count += 1
        elif os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            # Load corresponding source depth map
            src_depth_path = os.path.join(src_depth_dir, f"{i:04d}.exr")
            src_depth = load_depth(src_depth_path)
            
            if rgb_img is not None and src_depth is not None:
                refocused_rgb = refocus_image_gpu(rgb_img, src_depth, center_depth, frame_transform, shared_data, device)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if cv2.imwrite(output_path, refocused_rgb):
                    processed_count += 1
    
    return processed_count, len(frames)

def batch_process_render_data(base_dir, output_base, use_gpu=True):
    """批量处理渲染数据"""
    if use_gpu and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        use_gpu = False
    
    # Find all sequences (both OCC and GT)
    sequence_list = []
    for trajectory_type in os.listdir(base_dir):
        trajectory_dir = os.path.join(base_dir, trajectory_type)
        if not os.path.isdir(trajectory_dir):
            continue
            
        for sequence_name in os.listdir(trajectory_dir):
            # Process both OCC and GT sequences
            if not (sequence_name.endswith('_occ') or sequence_name.endswith('_GT')):
                continue
                
            sequence_dir = os.path.join(trajectory_dir, sequence_name)
            if not os.path.isdir(sequence_dir):
                continue
            
            # Determine GT depth directory (for center frame) and source depth directory
            if sequence_name.endswith('_occ'):
                # OCC sequence uses GT depth for center, OCC depth for source frames
                gt_sequence_name = sequence_name.replace('_occ', '_GT')
                gt_sequence_dir = os.path.join(trajectory_dir, gt_sequence_name)
                src_depth_dir = os.path.join(sequence_dir, "depth")  # OCC depth for source frames
            else:
                # GT sequence uses its own depth for both center and source frames
                gt_sequence_dir = sequence_dir
                src_depth_dir = os.path.join(sequence_dir, "depth")  # GT depth for source frames
            
            transforms_file = os.path.join(sequence_dir, "pose", "transforms.json")
            rgb_dir = os.path.join(sequence_dir, "rgb")
            gt_depth_dir = os.path.join(gt_sequence_dir, "depth")  # GT depth for center frame
            
            if not all(os.path.exists(p) for p in [transforms_file, rgb_dir, gt_depth_dir, src_depth_dir]):
                continue
            
            output_dir = os.path.join(output_base, trajectory_type, sequence_name)
            if os.path.exists(output_dir):
                continue
            
            sequence_list.append((trajectory_type, sequence_name, transforms_file, rgb_dir, gt_depth_dir, src_depth_dir, output_dir))
    
    if len(sequence_list) == 0:
        print("No sequences to process")
        return
    
    print(f"Found {len(sequence_list)} sequences to process")
    print(f"Using {'GPU' if use_gpu else 'CPU'} acceleration")
    print("Using INVERSE projection with GT depth map")
    
    total_processed = 0
    total_frames = 0
    
    with tqdm(sequence_list, desc="Processing sequences", unit="seq") as pbar:
        for trajectory_type, sequence_name, transforms_file, rgb_dir, gt_depth_dir, src_depth_dir, output_dir in pbar:
            pbar.set_postfix_str(f"{trajectory_type}/{sequence_name}")
            processed_count, frame_count = process_dataset(
                transforms_file, rgb_dir, gt_depth_dir, src_depth_dir, output_dir, use_gpu
            )
            total_processed += processed_count
            total_frames += frame_count
    
    print(f"✓ Processing completed!")
    print(f"  Sequences: {len(sequence_list)}")
    print(f"  Frames: {total_processed}/{total_frames}")
    print(f"  Success rate: {total_processed/total_frames*100:.1f}%")

def main():
    # ==================== 配置参数 ====================
    # 输入数据路径
    BASE_DIR = "/home_ssd/sjy/UE5_Project/PCGBiomeForestPoplar/Saved/MovieRenders/sparse_data"
    
    # 输出路径
    OUTPUT_BASE = "./refocus_sparse_data"
    
    # 是否使用GPU加速
    USE_GPU = True
    
    # ================================================
    
    batch_process_render_data(BASE_DIR, OUTPUT_BASE, USE_GPU)

if __name__ == "__main__":
    # 多进程安全保护
    mp.set_start_method('spawn', force=True)
    main()
