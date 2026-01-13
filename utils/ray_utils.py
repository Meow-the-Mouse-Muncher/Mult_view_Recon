import torch

def generate_rays(h: int, w: int, K, c2w, coords=None):
    """
    生成光线 (Origins and Directions), 支持 GPU
    参数:
        h, w: 图像的高和宽
        K: [3, 3] (Tensor or NumPy)
        c2w: [N, 4, 4] 或 [4, 4]
        coords: [M, 2] (Tensor) 可选，指定像素坐标 (x, y)。如果提供，则只生成这 M 条光线。
    返回:
        如果 coords is None:
            origins: [N, H, W, 3] (如果输入 c2w 有 N 维) 或 [H, W, 3]
            viewdirs: [N, H, W, 3] 或 [H, W, 3]
        如果 coords is not None:
            origins: [M, 3] (所有光线原点相同，但在 batch 处理中通常展开)
            viewdirs: [M, 3]
    """
    if torch.is_tensor(c2w):
        device = c2w.device
        dtype = torch.float64 # 计算阶段强制使用 float64
        K = K.to(device=device, dtype=dtype)
        c2w = c2w.to(dtype=dtype)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        K = torch.from_numpy(K).double().to(device)
        c2w = torch.from_numpy(c2w).double().to(device)
        dtype = torch.float64

    # 兼容单个 c2w 的情况
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    
    # 1. 准备像素坐标 pixels: [..., 3] (u, v, -1)
    pixel_center = 0.5
    
    if coords is not None:
        # 使用指定坐标: coords [M, 2] -> (x, y)
        # 注意: coords 通常是 (u, v) 即 (x, y)
        # 我们需要构造 [u + 0.5, v + 0.5, -1]
        coords = coords.to(device=device, dtype=dtype)
        ones = -torch.ones(coords.shape[0], 1, device=device, dtype=dtype) # z = -1
        pixels = torch.cat([coords + pixel_center, ones], dim=-1) # [M, 3]
        
    else:
        # 生成全图坐标
        i, j = torch.meshgrid(
            torch.arange(w, dtype=dtype, device=device) + pixel_center,
            torch.arange(h, dtype=dtype, device=device) + pixel_center,
            indexing="xy"
        )
        # [H, W, 3] -> (u, v, -1)
        pixels = torch.stack([i, j, -torch.ones_like(i)], dim=-1)

    # 2. 计算相机空间中的射线方向
    K_inv = torch.inverse(K)
    
    # 计算 cam_dirs = K_inv * pixels
    # pixels: [..., 3]
    # K_inv: [3, 3]
    # 结果: [..., 3]
    # 使用 matmul: [..., 3] @ [3, 3].T -> [..., 3]
    camera_dirs = pixels @ K_inv.T

    # 3. 将方向变换到世界坐标系: R @ camera_dirs
    # c2w: [N, 4, 4]
    rot = c2w[:, :3, :3]   # [N, 3, 3]
    trans = c2w[:, :3, 3]  # [N, 3]

    # directions = (Rot * cam_dirs)
    if coords is not None:
        # [N, 3, 3] @ [M, 3].T -> [N, 3, M] -> [N, M, 3]
        # 如果只有一个视角 (N=1)，可以直接 squeeze
        directions = (rot @ camera_dirs.T).transpose(1, 2)
    else:
        # [N, 1, 1, 3, 3] @ [1, H, W, 3, 1] -> [N, H, W, 3]
        directions = (rot[:, None, None, ...] @ camera_dirs[None, ..., None])[..., 0]

    # 4. 计算光线起点 (Origins)
    # trans: [N, 3] -> expand like directions
    if coords is not None:
        origins = trans[:, None, :].expand_as(directions)
    else:
        origins = trans[:, None, None, :].expand_as(directions)

    # 5. 归一化方向向量 (Viewdirs)
    viewdirs = directions / (torch.norm(directions, dim=-1, keepdim=True) + 1e-10)

    # 如果输入时 c2w 维度是 2 (单个视角)，则去掉 batch 维度
    if c2w.shape[0] == 1:
        origins = origins.squeeze(0)
        viewdirs = viewdirs.squeeze(0)

    # 返回时转回 float32
    return origins.float(), viewdirs.float()

def compute_pts3d(h, w, K, pose, depth, coords=None):
    """
    计算世界坐标系下的 3D 点, 支持 GPU
    参数:
        depth: [H, W] 或 [M] (如果提供了 coords)
        coords: [M, 2] 可选
    """
    if torch.is_tensor(pose):
        device = pose.device
        dtype = torch.float64
        K = K.to(device=device, dtype=dtype)
        pose = pose.to(dtype=dtype)
        depth = depth.to(device=device, dtype=dtype)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        K = torch.from_numpy(K).double().to(device)
        pose = torch.from_numpy(pose).double().to(device)
        depth = torch.from_numpy(depth).double().to(device)
        dtype = torch.float64
        
    pixel_center = 0.5
    
    if coords is not None:
        # coords: [M, 2]
        coords = coords.to(device=device, dtype=dtype)
        # pixels: [M, 3] (u, v, 1) # 注意这里 z 是 1 因为深度是正的
        ones = torch.ones(coords.shape[0], 1, device=device, dtype=dtype)
        pixels = torch.cat([coords + pixel_center, ones], dim=-1)
        # depth: [M] -> [M, 1]
        depth_vals = depth.unsqueeze(-1)
    else:
        i, j = torch.meshgrid(
            torch.arange(w, dtype=dtype, device=device) + pixel_center,
            torch.arange(h, dtype=dtype, device=device) + pixel_center,
            indexing='xy'
        )
        pixels = torch.stack([i, j, torch.ones_like(i)], dim=-1)
        depth_vals = depth.unsqueeze(-1) # [H, W, 1]

    K_inv = torch.inverse(K)
    
    # [..., 3]
    cam_dirs = pixels @ K_inv.T
    
    # 相机坐标系下的点 (假设面向 -z 轴，深度为 z distance)
    # 注意: 通常 depth 是 z-buffer，即 z 坐标。如果是欧氏距离则 direct * depth。
    # 这里假设是标准的 z-depth，并且相机看向 -z。
    # cam_dirs 在 z=-1 平面上。
    # 真正的方向向量是 cam_dirs (z=-1)。
    # 3D点 P = (x/z, y/z, 1) * z_depth ? 
    # 标准针孔模型: pixel = K * P_cam. 
    # P_cam = K_inv * pixel * depth. (如果 depth 是 z 坐标)
    # 因为我们构造像素时用了 z=1 或 z=-1? 
    # 原代码用的是 z=-1 (camera_dirs) 但这里 compute_pts3d 原代码用的是 z=1.
    # 保持原代码逻辑: cam_dirs * (-depth)
    
    pts_cam = cam_dirs * (-depth_vals if coords is None else -depth_vals)
    # 如果原逻辑是 z=1 的 cam_dirs, 那么 z 分量是 1 * -depth = -depth. 符合预期。
    # 但原代码 generate_rays 里是 -1. 这里 compute_pts3d 是 1. 不太一致，但只要保持这里逻辑自洽即可。
    # 假设 depth 是正值。点应该在 -z 前方。
    # pixels (z=1) -> K_inv -> cam_dirs (z 通常接近 1/f) 
    # pts_cam = cam_dirs * (-depth) -> z 分量变成负值。OK.

    R = pose[:3, :3]
    T = pose[:3, 3]
    
    # [3, 3] @ [..., 3].T -> [..., 3]
    if coords is not None:
        # [M, 3] @ R.T + T
        pts_world = pts_cam @ R.T + T
    else:
        # [H, W, 3] @ R.T + T
        pts_world = pts_cam @ R.T + T
    
    return pts_world.float()


