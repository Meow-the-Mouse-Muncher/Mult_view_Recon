import os
import torch
import lightning as L
from torch import nn
import lpips  # 感知损失库
from models.lfnr import LFNR
from dataset.LF_dataset import LFDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class LFModule(L.LightningModule):
    """Lightning 模型包装器，用于训练 Swin-UNet。"""
    def __init__(self, n_rays=4096):
        super().__init__()
        self.n_rays = n_rays
        
        # 损失权重
        self.mse_weight = 10
        self.perceptual_weight = 1e-2
        
        # 保存超参数
        # self.save_hyperparameters()
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, occ):
        return self.model(occ)  # 输入多视角数据，输出重建的 RGB 图像

    def training_step(self, batch, ):
        """训练步骤"""
        # 从字典提取数据
        gt = batch['gt_rgb']
        gt_pose = batch['gt_pose']
        gt_depth = batch['gt_depth']
        occ = batch['ref_rgb']
        occ_poses = batch['ref_poses']
        K = batch['K']
        
        pred = self(occ)  # 前向传播: [B, 96, H, W] -> [B, 3, H, W]
        
        # 计算 MSE 损失
        mse_loss = self.mse_loss(pred, gt)
        
        # 计算感知损失 (LPIPS 期望输入范围 [-1, 1])
        pred_norm = pred * 2 - 1  # [0, 1] -> [-1, 1]
        gt_norm = gt * 2 - 1      # [0, 1] -> [-1, 1]
        perceptual_loss = self.perceptual_loss(pred_norm, gt_norm).mean()
        
        # 总损失
        total_loss = self.mse_weight * mse_loss + self.perceptual_weight * perceptual_loss
        
        # 记录损失
        self.log('train/total_loss', total_loss, on_step=True, prog_bar=True)
        self.log('train/mse_loss', mse_loss, on_step=True, prog_bar=True)
        self.log('train/perceptual_loss', perceptual_loss, on_step=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        gt = batch['gt_rgb'].squeeze(1)
        occ = batch['ref_rgb'].reshape(batch['ref_rgb'].shape[0], -1, *batch['ref_rgb'].shape[-2:])
        view = batch['center_rgb'].squeeze(1)
        
        pred = self(occ)
        
        # 计算损失
        mse_loss = self.mse_loss(pred, gt)
        pred_norm = pred * 2 - 1
        gt_norm = gt * 2 - 1
        perceptual_loss = self.perceptual_loss(pred_norm, gt_norm).mean()
        total_loss = self.mse_weight * mse_loss + self.perceptual_weight * perceptual_loss
        
        # 计算指标
        psnr_val = self.psnr(pred, gt)
        ssim_val = self.ssim(pred, gt)
        
        # 记录验证损失
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        
        # 每个验证epoch记录第一个batch的图像
        if batch_idx == 0:
            # 取第一张图像
            view_img = view[0]  # [3, H, W]
            pred_img = pred[0]  # [3, H, W]
            gt_img = gt[0]      # [3, H, W]
            
            # 拼接图像：左中右 = view, pred, GT
            combined = torch.cat([view_img, pred_img, gt_img], dim=2)  # [3, H, 3*W]
            
            # 记录到 TensorBoard
            self.logger.experiment.add_image(
                'val_images/view_pred_gt', 
                combined, 
                self.current_epoch
            )
        return total_loss

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        gt = batch['gt_rgb'].squeeze(1)
        occ = batch['ref_rgb'].reshape(batch['ref_rgb'].shape[0], -1, *batch['ref_rgb'].shape[-2:])
        view = batch['center_rgb'].squeeze(1)
        
        pred = self(occ)
        
        # 计算损失
        mse_loss = self.mse_loss(pred, gt)
        pred_norm = pred * 2 - 1
        gt_norm = gt * 2 - 1
        perceptual_loss = self.perceptual_loss(pred_norm, gt_norm).mean()
        total_loss = self.mse_weight * mse_loss + self.perceptual_weight * perceptual_loss
        
        # 计算指标
        psnr_val = self.psnr(pred, gt)
        ssim_val = self.ssim(pred, gt)
        
        # 记录测试损失
        self.log('test/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/mse_loss', mse_loss, on_step=False, on_epoch=True)
        self.log('test/perceptual_loss', perceptual_loss, on_step=False, on_epoch=True)
        self.log('test/psnr', psnr_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/ssim', ssim_val, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 使用 AdamW 优化器，对 Transformer 模型效果更好
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-4,
            weight_decay=0.01,  # 权重衰减
            betas=(0.9, 0.999)
        )
        
        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=600,  # 最大轮数
            eta_min=1e-6  # 最小学习率
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

if __name__ == "__main__":
    print("=== 开始训练 LFNR 模型 ===")
    
    # 加载配置
    config = get_config()
    mode = "rot_arc" # mode =[fix_line,rot_arc,rot_line]
    
    # 1. 初始化模型包装器
    model = LFModule(config=config, n_rays=config.train.num_rays)
    
    # 2. 创建数据模块
    dm = LFDataModule(
        data_dir="sparse_data", # 指向你生成的 h5 文件夹
        model=mode,
        batch_size=config.dataset.batch_size if hasattr(config.dataset, 'batch_size') else 2,
        num_workers=8,
        n_rays=config.train.num_rays
    )

    # 创建 Trainer
    trainer = L.Trainer(
        max_epochs=2400,
        accelerator="gpu",
        devices=[0],  
        strategy="auto",
        logger=TensorBoardLogger("logs", name=mode, version=None), 
        # 回调函数
        callbacks=[
            # 模型检查点
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=os.path.join("checkpoints", mode),
                filename="swin-unet-{epoch:02d}",
                monitor="val/total_loss",  # 推荐监控验证损失
                mode="min",
                save_top_k=4,
            ),
            # 学习率监控
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        
        # 日志设置
        log_every_n_steps=20,
        # 验证集间隔
        check_val_every_n_epoch=20,
        # 梯度裁剪（对 Transformer 有帮助）
        # gradient_clip_val=1.0,
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