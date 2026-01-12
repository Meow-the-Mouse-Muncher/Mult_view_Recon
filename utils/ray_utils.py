import numpy as np
def generate_rays(h: int, w: int, K: np.ndarray, c2w: np.ndarray):
    """
    为所有视图生成光线 (Origins and Directions)。
    参数:
        h, w: 图像的高和宽
        K: 共享相机内参矩阵 [3, 3]
        c2w: Camera-to-World 位姿矩阵 [N, 4, 4]
    返回:
        origins: 光线起点 [N, H, W, 3]
        viewdirs: 光线方向单位向量 [N, H, W, 3]
    """
    pixel_center = 0.5
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32) + pixel_center,
        np.arange(h, dtype=np.float32) + pixel_center,
        indexing="xy"
    )

    # 在相机坐标系下的像素坐标 (针对内参矩阵定义，通常 Z=-1 为成像平面)
    pixels = np.stack((x, y, -np.ones_like(x)), axis=-1)  # [H, W, 3]

    # 1. 计算相机空间中的射线方向 (共享 K)
    K_inv = np.linalg.inv(K)
    camera_dirs = (K_inv[None, None, :] @ pixels[..., None])[..., 0] # [H, W, 3]

    # 2. 将方向变换到世界坐标系: R @ camera_dirs
    # c2w 形状为 [N, 4, 4]
    rot = c2w[..., :3, :3]   # [N, 3, 3]
    trans = c2w[..., :3, -1] # [N, 3]

    # 利用 NumPy 广播计算所有视角的方向: [N, 1, 1, 3, 3] @ [1, H, W, 3, 1]
    directions = (rot[:, None, None, ...] @ camera_dirs[None, ..., None])[..., 0] # [N, H, W, 3]

    # 3. 计算光线起点 (Origins)
    origins = np.broadcast_to(trans[:, None, None, :], directions.shape) # [N, H, W, 3]
    
    # 4. 归一化方向向量 (Viewdirs)
    viewdirs = directions / (np.linalg.norm(directions, axis=-1, keepdims=True) + 1e-10)

    return origins, viewdirs
def compute_pts3d(h, w, K, pose, depth):
    """计算世界坐标系下的 3D 点"""
    pixel_center = 0.5
    i, j = np.meshgrid(
        np.arange(w, dtype=np.float32) + pixel_center, 
        np.arange(h, dtype=np.float32) + pixel_center, 
        indexing='xy')
    pixels = np.stack([i, j, np.ones_like(i)], axis=-1) 
    K_inv = np.linalg.inv(K)
    cam_dirs = (K_inv[None, None, :] @ pixels[..., None])[..., 0] 
    # 相机面向 -z 轴
    pts_cam = cam_dirs * (-depth[..., None])
    R = pose[:3, :3]
    T = pose[:3, 3]
    pts_world = (R[None, None, ...] @ pts_cam[..., None])[..., 0] + T[None, None, :]
    return pts_world.astype(np.float32)

