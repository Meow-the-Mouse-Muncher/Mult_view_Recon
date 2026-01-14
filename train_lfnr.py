import os
import torch
import lightning as L
from torch import nn
from models.lfnr import LFNR
from dataset.LF_dataset import LFDataModule
from configs.config import get_config
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class LFModule(L.LightningModule):
    """Lightning 模型包装器，用于训练 LFNR。"""
    def __init__(self, config, n_rays=4096):
        super().__init__()
        # 直接保存整个 config 对象
        self.save_hyperparameters(config)
        self.config = config
        self.n_rays = n_rays
        
        # 初始化模型：仅传一个 config 对象
        self.model = LFNR(config=config)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        # 指标
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, batch_idx):
        """训练步骤 (Sparse Ray Training)"""
        # 1. 前向传播
        # 返回值: 预测RGB, 重叠区RGB(如有)
        pred_rgb, rgb_overlap = self(batch)
        gt_rgb = batch['gt_rgb'] # [B, n_rays, 3]
        
        # 2. 计算损耗
        # 主预测损失
        loss_pred = self.mse_loss(pred_rgb, gt_rgb)
        
        # 重叠区域/辅助损失 (如果模型支持)
        loss_overlap = self.mse_loss(rgb_overlap, gt_rgb)
        # 总损失 (L2权重衰减已在AdamW中处理)
        loss = loss_pred + loss_overlap
        
        # 3. 记录日志
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/loss_pred', loss_pred, sync_dist=True)
        self.log('train/loss_overlap', loss_overlap, sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        # """验证步骤 (通常渲染全图)"""
        # # Batch Size 在验证时应该为 1，因为全图光线 H*W 很大
        # # batch['gt_rgb']: [1, H*W, 3]
        
        # # 1. 前向传播 (Chunking 处理以防 OOM)
        # # 注意：由于 sampling_grid 已经是 [1, N, H*W, 2]，我们需要对 rays 和 grid 进行切片
        # chunk_size = 2048 # 强行改小试试，原先可能是 8192 太大了
        # view_img = batch['occ_center_rgb'] 
        # B, total_pixels, _ = batch['gt_rgb'].shape
        # assert B == 1, "验证/测试时 Batch Size 必须为 1"
        
        # all_pred_rgb = []
        
        # # 逐块进行推理
        # for i in range(0, total_pixels, chunk_size):
        #     end = min(i + chunk_size, total_pixels)
            
        #     # 构建一个 mini-batch 字典
        #     chunk_batch = {
        #         # [1, chunk, 3]
        #         'gt_rays_o': batch['gt_rays_o'][:, i:end, :],
        #         'gt_rays_d': batch['gt_rays_d'][:, i:end, :],
        #         'pts_3d':    batch['pts_3d'][:, i:end, :],
                
        #         # 参考信息部分
        #         'occ_rgb': batch['occ_rgb'], 

        #         # grid and rays need slicing
        #         'sampling_grid': batch['sampling_grid'][:, :, i:end, :],
        #         'occ_rays_d':    batch['occ_rays_d'][:, :, i:end, :],
        #         'occ_rays_o':    batch['occ_rays_o'][:, :, i:end, :]
        #     }
            
        #     # 预测
        #     with torch.no_grad():
        #         pred_chunk = self(chunk_batch)[0] 
        #         # 关键修改：立即由 GPU 转存到 CPU，腾出显存给下一块
        #         all_pred_rgb.append(pred_chunk.cpu()) 
                
        # # 拼接结果 (在 CPU 上进行)
        # pred_rgb = torch.cat(all_pred_rgb, dim=1).to(self.device) # 如果需要计算 loss 再转回去，或者直接在 CPU 算 PSNR
        
        # # 优化：为了计算 PSNR，把 gt_rgb 也转到 CPU 算，彻底省显存
        # gt_rgb_cpu = batch['gt_rgb'].cpu()
        # pred_rgb_cpu = torch.cat(all_pred_rgb, dim=1) # 已经在 CPU 上了
        
        # # 2. 计算 PSNR (推荐使用 torchmetrics 的函数式接口，或者临时新建对象，避免设备冲突)
        # # 方式 A: 直接手动计算 MSE 转 PSNR (最快，无依赖)
        # mse = torch.mean((pred_rgb_cpu.clamp(0, 1) - gt_rgb_cpu.clamp(0, 1)) ** 2)
        # psnr_val = -10.0 * torch.log10(mse)
        
        # self.log('val/psnr', psnr_val, on_epoch=True, prog_bar=True, sync_dist=True)

        # # 3. 记录第一张图像 (仅在 batch_idx == 0 时执行)
        # if batch_idx == 0:
        #     H, W = batch['H'].item(), batch['W'].item()
            
        #     # 准备图像数据: [1, 3, H, W]
        #     # 一张图显存占用很小，不会 OOM
        #     p_img_gpu = pred_rgb_cpu[0].view(H, W, 3).permute(2, 0, 1).clamp(0, 1).unsqueeze(0).to(self.device)
        #     g_img_gpu = gt_rgb_cpu[0].view(H, W, 3).permute(2, 0, 1).clamp(0, 1).unsqueeze(0).to(self.device)
        #     c_img_gpu = batch['occ_center_rgb'].to(self.device)
        #     # 确保维度一致 (有时 center 图可能是 [B, H, W, 3] 或者没有 batch 维)
        #     if c_img_gpu.ndim == 3: c_img_gpu = c_img_gpu.unsqueeze(0)
        #     if c_img_gpu.shape[-1] == 3: c_img_gpu = c_img_gpu.permute(0, 3, 1, 2) # [1, 3, H, W]
            
        #     # 使用 self.ssim (在 GPU) 计算
        #     ssim_val = self.ssim(p_img_gpu, g_img_gpu)
        #     self.log('val/ssim', ssim_val, on_epoch=True)
        #     concat_img = torch.cat([c_img_gpu, p_img_gpu, g_img_gpu], dim=3)

        #     # TensorBoard 记录 (不需要 GPU，取回 CPU)
        #     self.logger.experiment.add_image('val/View_Pred_GT', concat_img[0].cpu(), self.global_step)
            
        # return psnr_val

    def test_step(self, batch, batch_idx):
        """测试步骤（逻辑同 Val）"""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.train.lr_init,
            weight_decay=self.config.train.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.train.num_epochs,
            eta_min=self.config.train.lr_final
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # def on_after_backward(self):
    #     # 仅在训练的第一步运行一次检查
    #     if self.global_step == 0:
    #         print("\n" + "="*50)
    #         print("正在检测未使用的模型参数 (grad is None):")
    #         unused_params = []
    #         for name, param in self.named_parameters():
    #             if param.grad is None:
    #                 unused_params.append(name)
    #                 print(f"🚩 未使用的参数: {name}")
            
    #         if not unused_params:
    #             print("✅ 完美！所有参数都参与了梯度计算。")
    #         else:
    #             print(f"\n共发现 {len(unused_params)} 个未使用的参数。")
    #         print("="*50 + "\n")

if __name__ == "__main__":
    print("=== 开始训练 LFNR 模型 ===")
    
    # 加载配置
    config = get_config()
    mode = "rot_arc" # mode =[fix_line,rot_arc,rot_line]
    # 1. 初始化模型包装器
    model = LFModule(config=config, n_rays=config.train.num_rays)
    
    # 2. 创建数据模块
    dm = LFDataModule(
        data_dir="data",
        model=mode,
        batch_size=1, # 单个gpu上的batch size
        num_workers=4,
        n_rays=config.train.num_rays
    )

    # 创建 Trainer
    trainer = L.Trainer(
        max_epochs=config.train.num_epochs,
        accelerator="gpu",
        devices=2,  
        strategy="ddp",
        logger=TensorBoardLogger("logs", name=mode, version=None), 
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=os.path.join("checkpoints", mode),
                filename="lfnr-{epoch:02d}",
                monitor="epoch",  # 监控 epoch 数量
                mode="max",       # 保存 epoch 最大的（也就是最新的）
                save_top_k=4,
                every_n_epochs=5
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        log_every_n_steps=20,
        check_val_every_n_epoch=5, 
    )


    # 打印模型信息
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 增加断点重训逻辑 ---
    ckpt_dir = os.path.join("checkpoints", mode)
    last_ckpt = None
    if os.path.exists(ckpt_dir):
        # 寻找目录下所有的 .ckpt 文件
        ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if ckpts:
            # 找到最后修改的文件（通常是最近保存的）
            last_ckpt = max(ckpts, key=os.path.getmtime)
            print(f"检测到断点文件，将从此处恢复训练: {last_ckpt}")

    # 开始训练 (传入 ckpt_path 参数)
    trainer.fit(model, dm, ckpt_path=last_ckpt)
    
    # 可选：测试最佳模型
    trainer.test(model, dm, ckpt_path="best")
    
    print("=== 训练完成 ===")