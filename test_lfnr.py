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
    def __init__(self, config=None, n_rays=4096, save_dir="inference_results"):
        super().__init__()
        # 如果是从 checkpoint 加载，config 会由 lightning 自动恢复到 self.hparams 中
        self.save_hyperparameters(config)
        self.config = config
        self.n_rays = n_rays
        self.save_dir = save_dir
        
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
        self.test_step_outputs.append({
            'pred': pred_chunk.detach(),
            'gt': batch['gt_rgb'].detach(),
            'file_idx': batch['meta_file_idx'],
            'start_idx': batch['meta_start']
        })

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
                f.write("Filename, PSNR, SSIM\n")
                total_psnr, total_ssim, count = 0, 0, 0

                for fid in unique_files:
                    fid = int(fid.item())
                    if fid < 0: continue
                    mask = (global_fidxs == fid)
                    sort_idx = torch.argsort(global_starts[mask])
                    img_p = global_preds[mask][sort_idx].reshape(-1, 3)
                    img_g = global_gts[mask][sort_idx].reshape(-1, 3)
                    H = int(img_p.shape[0]**0.5)
                    if H*H != img_p.shape[0]: continue
                    
                    view_p = img_p.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1)
                    view_g = img_g.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1)
                    
                    cur_psnr = self.psnr(view_p, view_g).item()
                    cur_ssim = self.ssim(view_p, view_g).item()
                    total_psnr += cur_psnr
                    total_ssim += cur_ssim
                    count += 1
                    
                    fname = os.path.splitext(os.path.basename(test_h5_files[fid]))[0] if fid < len(test_h5_files) else f"unknown_{fid}"
                    save_path = os.path.join(self.save_dir, f"{fname}.png")
                    save_image(torch.cat([view_p, view_g], dim=3), save_path)
                    f.write(f"{fname}, {cur_psnr:.4f}, {cur_ssim:.4f}\n")
                
                if count > 0:
                    f.write(f"\nAverage: PSNR={total_psnr/count:.4f}, SSIM={total_ssim/count:.4f}\n")

        self.test_step_outputs.clear()

def main():
    parser = argparse.ArgumentParser(description="LFNR Inference Script")
    parser.add_argument("--ckpt", type=str,default="checkpoints/base_LF/rot_arc/lfnr-epoch=174.ckpt", help="Path to the checkpoint (.ckpt) file")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory of the dataset") #
    parser.add_argument("--save_dir", type=str, default="inference/base_LF", help="Directory to save the predicted images")
    parser.add_argument("--mode", type=str, default="fix_line", help="Data mode: fix_line, rot_arc, etc.")
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()

    print(f"=== 开始推理 | Checkpoint: {args.ckpt} | Mode: {args.mode} ===")

    # 1. 加载配置
    config = get_config()
    
    # 2. 初始化模型
    print(f"正在从 {args.ckpt} 加载模型...")
    model = LFModule.load_from_checkpoint(checkpoint_path=args.ckpt)
    
    # 手动赋值，绕过 OmegaConf 的只读限制
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
