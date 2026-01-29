import os
import torch
import torch.nn as nn
import lightning as L
import argparse
import h5py
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from models.lfnr import LFNR 
from configs.config import get_config
from dataset.LF_dataset import LFDataModule

class LFModule(L.LightningModule):
    """Lightning 模型包装器，专门用于解耦推理。"""
    def __init__(self, config=None, n_rays=4096, save_dir="inference_results", use_roi=False):
        super().__init__()
        # 如果是从 checkpoint 加载，config 会由 lightning 自动恢复到 self.hparams 中
        self.save_hyperparameters(config)
        self.config = config
        self.n_rays = n_rays
        self.save_dir = save_dir
        self.use_roi = use_roi # 新增
        
        # 初始化模型
        self.model = LFNR(config=config)
        
        # 指标初始化
        metrics_kwargs = {"data_range": 1.0}
        self.psnr = PeakSignalNoiseRatio(**metrics_kwargs)
        self.ssim = StructuralSimilarityIndexMeasure(**metrics_kwargs)
        self.test_step_outputs = []

    def forward(self, batch):
        return self.model.forward(batch)

    def test_step(self, batch, batch_idx):
        pred_chunk, _ = self(batch)
        output = {
            'pred': pred_chunk.detach(),
            'gt': batch['gt_rgb'].detach(),
            'file_idx': batch['meta_file_idx'],
            'start_idx': batch['meta_start']
        }
        # 如果开启 ROI，收集 3D 点和相机中心用于计算深度
        if self.use_roi:
            output['pts_3d'] = batch['pts_3d'].detach()
            output['rays_o'] = batch['gt_rays_o'].detach()
            
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self):
        if not self.test_step_outputs: return

        local_preds = torch.cat([x['pred'] for x in self.test_step_outputs], dim=0)
        local_gts = torch.cat([x['gt'] for x in self.test_step_outputs], dim=0)
        local_fidxs = torch.cat([x['file_idx'] for x in self.test_step_outputs], dim=0)
        local_starts = torch.cat([x['start_idx'] for x in self.test_step_outputs], dim=0)

        global_preds = self.all_gather(local_preds).view(-1, local_preds.shape[1], 3).cpu()
        global_gts = self.all_gather(local_gts).view(-1, local_gts.shape[1], 3).cpu()
        global_fidxs = self.all_gather(local_fidxs).view(-1).cpu()
        global_starts = self.all_gather(local_starts).view(-1).cpu()

        # 如果使用 ROI，聚合深度信息
        global_depths = None
        if self.use_roi:
            local_pts = torch.cat([x['pts_3d'] for x in self.test_step_outputs], dim=0)
            local_os = torch.cat([x['rays_o'] for x in self.test_step_outputs], dim=0)
            local_d = torch.norm(local_pts - local_os, dim=-1)
            global_depths = self.all_gather(local_d).view(-1).cpu()

        if self.global_rank == 0:
            test_h5_files = []
            if hasattr(self.trainer.datamodule, 'test_dataset'):
                test_h5_files = self.trainer.datamodule.test_dataset.h5_files
            else:
                print("Warning: test_dataset not found.")

            unique_files = torch.unique(global_fidxs)
            os.makedirs(self.save_dir, exist_ok=True)
            log_path = os.path.join(self.save_dir, "metrics.txt")
            
            with open(log_path, "w") as f:
                f.write("Filename, PSNR, SSIM, ROI_Rect(y1,x1,y2,x2)\n")
                total_psnr, total_ssim, count = 0, 0, 0

                for fid in unique_files:
                    fid = int(fid.item())
                    if fid < 0: continue
                    
                    h5_path = test_h5_files[fid]
                    with h5py.File(h5_path, 'r') as f_h5:
                        H, W = f_h5['GT/depth'].shape
                    
                    mask = (global_fidxs == fid)
                    sort_idx = torch.argsort(global_starts[mask])
                    
                    # 确保像素总数匹配 H*W
                    img_p_flat = global_preds[mask][sort_idx].reshape(-1, 3)
                    img_g_flat = global_gts[mask][sort_idx].reshape(-1, 3)
                    if img_p_flat.shape[0] != H * W:
                        print(f"Skipping fid={fid}, shape mismatch: {img_p_flat.shape[0]} vs {H}x{W}")
                        continue
                        
                    # 2. ROI 矩形框计算逻辑
                    crop_slice = (slice(None), slice(None))
                    mask_vis = None
                    rect_info = "Full"
                    
                    if self.use_roi and global_depths is not None:
                        depth_map = global_depths[mask][sort_idx] # 1D [H*W]
                        # 剔除最远 20%
                        val_80 = torch.quantile(depth_map, 0.8)
                        valid_mask_bool = (depth_map <= val_80)
                        
                        # 生成可视化 Mask [H, W, 3]
                        mask_vis = valid_mask_bool.reshape(H, W).float().unsqueeze(-1).expand(-1, -1, 3)
                        
                        # 基于有效点坐标计算最小外接矩形 (Rectangle)
                        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).reshape(-1, 2)
                        valid_coords = coords[valid_mask_bool]
                        
                        if len(valid_coords) > 0:
                            y_min, x_min = valid_coords.min(dim=0)[0].item()
                            y_max, x_max = valid_coords.max(dim=0)[0].item()
                            crop_slice = (slice(y_min, y_max+1), slice(x_min, x_max+1))
                            rect_info = f"({y_min},{x_min},{y_max},{x_max})"
                    
                    # 3. 准备图像用于指标计算和保存
                    view_p_full = img_p_flat.view(H, W, 3).clamp(0, 1)
                    view_g_full = img_g_flat.view(H, W, 3).clamp(0, 1)
                    
                    # 裁剪出那个矩形框 (ROI) 
                    # 将张量移动到模型所在设备以便计算指标 (防止设备不匹配报错)
                    view_p_roi = view_p_full[crop_slice].permute(2, 0, 1).unsqueeze(0).to(self.device)
                    view_g_roi = view_g_full[crop_slice].permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # 计算指标 (计算后立即 reset，防止多图累加)
                    cur_psnr = self.psnr(view_p_roi, view_g_roi).item()
                    self.psnr.reset()
                    cur_ssim = self.ssim(view_p_roi, view_g_roi).item()
                    self.ssim.reset()
                    
                    total_psnr += cur_psnr
                    total_ssim += cur_ssim
                    count += 1
                    
                    # 4. 可视化拼接内容准备
                    fname = os.path.splitext(os.path.basename(h5_path))[0]
                    save_path = os.path.join(self.save_dir, f"{fname}.png")
                    
                    # 读取 Center View (带遮挡的原始输入图)
                    center_view = None
                    try:
                        with h5py.File(h5_path, 'r') as h5f:
                            if 'occ_center/rgb' in h5f:
                                # H5 中存储通常是 [H, W, 3]
                                center_view = torch.from_numpy(h5f['occ_center/rgb'][:]).float() / 255.0
                    except Exception as e:
                        print(f"Warning: Could not read occ_center for {fname}: {e}")

                    # 构造拼接列表，顺序: Mask, Center, Pred, GT
                    save_list = []
                    if mask_vis is not None:
                        save_list.append(mask_vis)
                    if center_view is not None:
                        save_list.append(center_view)
                    
                    save_list.append(view_p_full)
                    save_list.append(view_g_full)
                    
                    # 在宽度方向(dim=1)拼接所有 [H, W, 3] 的图像
                    # 然后转换成 [3, H, W*N] 以符合 save_image 格式
                    save_grid = torch.cat(save_list, dim=1).permute(2, 0, 1)
                    save_image(save_grid, save_path)
                    
                    f.write(f"{fname}, {cur_psnr:.4f}, {cur_ssim:.4f}, {rect_info}\n")
                
                if count > 0:
                    f.write(f"\nAverage: PSNR={total_psnr/count:.4f}, SSIM={total_ssim/count:.4f}\n")

        self.test_step_outputs.clear()

def main():
    parser = argparse.ArgumentParser(description="LFNR Inference Script")
    parser.add_argument("--ckpt", type=str,default="checkpoints/base_LF/rot_arc/lfnr-epoch=174.ckpt", help="Path to the checkpoint (.ckpt) file")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory of the dataset") #
    parser.add_argument("--save_dir", type=str, default="inference/base_LF", help="Directory to save the predicted images")
    parser.add_argument("--mode", type=str, default="fix_line", help="Data mode: fix_line, rot_arc, etc.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--use_roi", action="store_true", help="Calculate metrics only on non-background ROI") # 新增
    args = parser.parse_args()

    print(f"=== 开始推理 | Checkpoint: {args.ckpt} | Mode: {args.mode} ===")

    # 1. 加载配置
    config = get_config()
    
    # 2. 初始化模型
    print(f"正在从 {args.ckpt} 加载模型...")
    model = LFModule.load_from_checkpoint(checkpoint_path=args.ckpt, use_roi=args.use_roi)
    model.save_dir = os.path.join(args.save_dir, args.mode)

    # 3. 初始化数据模块
    # 简化传参，让 LFDataModule 自动根据 mode 构建路径
    dm = LFDataModule(
        data_dir=args.data_dir, # 只传根目录 "data"
        model=args.mode,        # 传入 "fix_line"
        batch_size=1,
        num_workers=4,
        n_rays=config.train.num_rays,
        val_chunk_size=config.eval.chunk
    )

    # 4. 初始化 Trainer 并执行测试
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        logger=False, # 推理时通常不需要 TensorBoard
    )

    print("=== 正在运行推理 (Test Step) ===")
    trainer.test(model, datamodule=dm)

    print(f"=== 推理完成! 结果已保存至: {args.save_dir} ===")

if __name__ == "__main__":
    main()
