import torch
def generate_rays(h: int, w: int, K, c2w):
    """
    为所有视图生成光线 (Origins and Directions), 支持 GPU
    参数:
        h, w: 图像的高和宽
        K: [3, 3] (Tensor or NumPy)
        c2w: [N, 4, 4] (Tensor or NumPy)
    返回:
        origins: [N, H, W, 3]
        viewdirs: [N, H, W, 3]
    """
    if torch.is_tensor(c2w):
        device = c2w.device
        dtype = c2w.dtype
        K = K.to(device=device, dtype=dtype)
    else:
        # 如果还在 CPU 上，可以选择转 Tensor 也可以保持 NumPy，这里为了统一用 Torch 逻辑
        # 或者仅仅保持原有的 NumPy 实现？用户的需求是 "改为 GPU 处理"，所以默认转换为 GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        K = torch.from_numpy(K).float().to(device)
        c2w = torch.from_numpy(c2w).float().to(device)
        dtype = torch.float32

    pixel_center = 0.5
    i, j = torch.meshgrid(
        torch.arange(w, dtype=dtype, device=device) + pixel_center,
        torch.arange(h, dtype=dtype, device=device) + pixel_center,
        indexing="xy"
    )

    # [H, W, 3] -> (u, v, -1)
    pixels = torch.stack([i, j, -torch.ones_like(i)], dim=-1)

    # 1. 计算相机空间中的射线方向 (共享 K)
    K_inv = torch.inverse(K)
    # [3, 3] @ [H, W, 3, 1] -> [H, W, 3]
    camera_dirs = (K_inv[None, None, ...] @ pixels[..., None])[..., 0]

    # 2. 将方向变换到世界坐标系: R @ camera_dirs
    # c2w 形状为 [N, 4, 4]
    rot = c2w[..., :3, :3]   # [N, 3, 3]
    trans = c2w[..., :3, 3]  # [N, 3]

    # 利用 Broadcast 计算所有视角的方向: [N, 1, 1, 3, 3] @ [1, H, W, 3, 1]
    directions = (rot[:, None, None, ...] @ camera_dirs[None, ..., None])[..., 0]

    # 3. 计算光线起点 (Origins)
    origins = trans[:, None, None, :].expand_as(directions)
    
    # 4. 归一化方向向量 (Viewdirs)
    viewdirs = directions / (torch.norm(directions, dim=-1, keepdim=True) + 1e-10)

    return origins, viewdirs

def compute_pts3d(h, w, K, pose, depth):
    """计算世界坐标系下的 3D 点, 支持 GPU"""
    if torch.is_tensor(pose):
        device = pose.device
        dtype = pose.dtype
        K = K.to(device=device, dtype=dtype)
        depth = depth.to(device=device, dtype=dtype)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        K = torch.from_numpy(K).float().to(device)
        pose = torch.from_numpy(pose).float().to(device)
        depth = torch.from_numpy(depth).float().to(device)
        dtype = torch.float32
        
    pixel_center = 0.5
    i, j = torch.meshgrid(
        torch.arange(w, dtype=dtype, device=device) + pixel_center,
        torch.arange(h, dtype=dtype, device=device) + pixel_center,
        indexing='xy'
    )
    pixels = torch.stack([i, j, torch.ones_like(i)], dim=-1)
    K_inv = torch.inverse(K)
    
    # [H, W, 3]
    cam_dirs = (K_inv[None, None, ...] @ pixels[..., None])[..., 0]
    
    # 相机面向 -z 轴
    pts_cam = cam_dirs * (-depth[..., None])
    
    R = pose[:3, :3]
    T = pose[:3, 3]
    
    # [3, 3] @ [H, W, 3, 1] + [3]
    pts_world = (R[None, None, ...] @ pts_cam[..., None])[..., 0] + T[None, None, :]
    return pts_world

