import os
import cv2
import numpy as np
import glob
import argparse

def create_epi(rgb_dir, output_path, mode='row', index=None):
    """
    从图像序列生成 EPI 图。
    
    Args:
        rgb_dir: 包含 RGB 图像的目录路径。
        output_path: 保存生成的 EPI 图像的路径。
        mode: 'row' (水平 EPI) 或 'col' (垂直 EPI)。
        index: 行或列的索引。如果为 None，则默认取中间位置。
    """
    # 获取目录下的所有 png 图像并排序
    img_list = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    
    if not img_list:
        print(f"错误: 在 {rgb_dir} 中未找到 PNG 图像。")
        return

    # 读取第一张图以获取尺寸
    first_img = cv2.imread(img_list[0])
    if first_img is None:
        print(f"错误: 无法读取图像 {img_list[0]}")
        return
    
    h, w, c = first_img.shape
    print(f"图像尺寸: {w}x{h}, 序列长度: {len(img_list)}")

    # 设置默认索引
    if index is None:
        index = h // 2 if mode == 'row' else w // 2
    
    print(f"正在基于 {'行' if mode == 'row' else '列'} 索引 {index} 生成 EPI...")

    mid_idx = len(img_list) // 2
    print(f"序列长度: {len(img_list)}，中间索引: {mid_idx}。索引 >= {mid_idx} 的图像将旋转 180 度。")

    epi_lines = []
    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # # 如果索引在中间之后，旋转 180 度
        # if i > mid_idx:
        #     img = cv2.rotate(img, cv2.ROTATE_180)
            
        if mode == 'row':
            # 提取指定行: [width, channels]
            line = img[index, :, :]
        else:
            # 提取指定列: [height, channels]
            line = img[:, index, :]
        epi_lines.append(line)

    # 堆叠生成 EPI
    if mode == 'row':
        # 结果尺寸: [num_images, width, channels]
        epi = np.stack(epi_lines, axis=0)
    else:
        # 结果尺寸: [height, num_images, channels]
        epi = np.stack(epi_lines, axis=1)

    # 保存结果
    cv2.imwrite(output_path, epi)
    print(f"成功! EPI 已保存至: {output_path}")

if __name__ == "__main__":
    # 配置路径
    base_dir = "/home_ssd/sjy/Active_cam_recon/test/rot_arc"
    rgb_path = os.path.join(base_dir, "scene_010_Target_002_height_050_ang_000_GT/rgb")
    
    # 示例 1: 生成第 512 行（图像中心附近）的水平 EPI
    bias = 50
    index = 1024//2 + bias  # 假设图像高度为 1024,偏离中心20个像素
    create_epi(rgb_path, "epi_row.png", mode='row', index=index)
    
    # 示例 2: 生成第 512 列的垂直 EPI
    create_epi(rgb_path, "epi_col.png", mode='col', index=index)