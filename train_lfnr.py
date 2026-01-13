import os
import torch
import lightning as L
from torch import nn
import lpips
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
        self.lpips_loss = lpips.LPIPS(net='vgg')
        
        # 指标
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, batch_idx):
        """训练步骤 (Sparse Ray Training)"""
        # 1. 前向传播
        # 返回值: 预测RGB, 重叠区RGB(如有), 注意力权重, 学习到的相机嵌入
        pred_rgb, rgb_overlap, attn_weights, learned_embed = self(batch)
        gt_rgb = batch['gt_rgb'] # [B, n_rays, 3]
        
        # 2. 计算损耗
        # 主预测损失
        loss_pred = self.mse_loss(pred_rgb, gt_rgb)
        
        # 重叠区域/辅助损失 (如果模型支持)
        loss_overlap = self.mse_loss(rgb_overlap, gt_rgb)
        # 总损失 (L2权重衰减已在AdamW中处理)
        loss = loss_pred + loss_overlap
        
        # 3. 记录日志
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/loss_pred', loss_pred, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/loss_overlap', loss_overlap, on_step=True, on_epoch=True, sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤 (通常渲染全图)"""
        # 前向传播
        # 在验证阶段，我们可能只关心最终的 pred_rgb
        outputs = self(batch)
        if isinstance(outputs, tuple):
            pred_rgb = outputs[0]
        else:
            pred_rgb = outputs
            
        gt_rgb = batch['gt_rgb'] # [B, H*W, 3]
        
        # 计算 PSNR 基础指标
        psnr_val = self.psnr(pred_rgb.clamp(0, 1), gt_rgb.clamp(0, 1))
        self.log('val/psnr', psnr_val, on_epoch=True, prog_bar=True, sync_dist=True)

        # 记录图像可视化 (仅记录 batch 中的第一张)
        if batch_idx == 0:
            # 这里的 H, W 需要在 Dataset 提供并在 batch 中带上
            H, W = batch['H'][0].item(), batch['W'][0].item()
            
            # Reshape: [B=1, H*W, 3] -> [3, H, W]
            p_img = pred_rgb[0].view(H, W, 3).permute(2, 0, 1).clamp(0, 1)
            g_img = gt_rgb[0].view(H, W, 3).permute(2, 0, 1).clamp(0, 1)
            
            # 计算图像级指标
            ssim_val = self.ssim(p_img.unsqueeze(0), g_img.unsqueeze(0))
            # LPIPS 需要 [-1, 1] 范围输入
            lpips_val = self.lpips_loss(p_img.unsqueeze(0)*2-1, g_img.unsqueeze(0)*2-1).mean()
            
            self.log('val/ssim', ssim_val, on_epoch=True)
            self.log('val/lpips', lpips_val, on_epoch=True)

            # 写入 TensorBoard
            self.logger.experiment.add_image('val/prediction', p_img, self.global_step)
            self.logger.experiment.add_image('val/ground_truth', g_img, self.global_step)
            
        return psnr_val

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        pred_rgb = self(batch)
        gt_rgb = batch['gt_rgb']
        
        psnr_val = self.psnr(pred_rgb, gt_rgb)
        self.log('test/psnr', psnr_val, on_epoch=True)
        return psnr_val

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

if __name__ == "__main__":
    print("=== 开始训练 LFNR 模型 ===")
    
    # 加载配置
    config = get_config()
    mode = "fix_line" # mode =[fix_line,rot_arc,rot_line]
    
    # 1. 初始化模型包装器
    model = LFModule(config=config, n_rays=config.train.num_rays)
    
    # 2. 创建数据模块
    dm = LFDataModule(
        data_dir="data",
        model=mode,
        batch_size=1, # 训练时每个 batch 包含 N_rays，所以 batch_size 设为 1 即可
        num_workers=8,
        n_rays=config.train.num_rays
    )

    # 创建 Trainer
    trainer = L.Trainer(
        max_epochs=config.train.num_epochs,
        accelerator="gpu",
        devices=[0],  
        strategy="auto",
        logger=TensorBoardLogger("logs", name=mode, version=None), 
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=os.path.join("checkpoints", mode),
                filename="lfnr-{epoch:02d}",
                monitor="val/psnr",
                mode="max",
                save_top_k=4,
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        log_every_n_steps=20,
        check_val_every_n_epoch=5, # 减少验证频率
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