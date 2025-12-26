import os
import torch
import lightning as L
from torch import nn
from models.recon_model import PyramidUNet, ModelConfig, SwinUNetConfigs, create_swin_unet
from dataset.lightning_dataset import RefocusDataModule

class ReconLightningModule(L.LightningModule):
    """Lightning 模型包装器，用于训练 Swin-UNet。"""
    def __init__(self, model_config: ModelConfig = None, model_size: str = 'tiny'):
        super().__init__()
        
        # 如果没有提供配置，使用预定义配置
        if model_config is None:
            self.model = create_swin_unet(
                model_size=model_size, 
                in_chans=32*3,  # 32个视角 * 3通道
                out_chans=3,    # RGB 输出
                img_size=512
            )
        else:
            self.model = PyramidUNet(model_config)
            
        self.criterion = nn.MSELoss()  # 图像重建任务使用 MSE 损失
        
        # 保存超参数
        self.save_hyperparameters()

    def forward(self, occ):
        return self.model(occ)  # 输入多视角数据，输出重建的 RGB 图像

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        gt, occ = batch  # GT: [B, 3, H, W], occ: [B, 96, H, W]
        pred = self(occ)  # 前向传播: [B, 96, H, W] -> [B, 3, H, W]
        loss = self.criterion(pred, gt)  # 计算重建损失
        
        # 记录损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # 可选：记录学习率
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        gt, occ = batch
        pred = self(occ)
        loss = self.criterion(pred, gt)
        
        # 记录验证损失
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        gt, occ = batch
        pred = self(occ)
        loss = self.criterion(pred, gt)
        
        # 记录测试损失
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return loss

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
            T_max=200,  # 最大轮数
            eta_min=1e-6  # 最小学习率
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",  # 改为监控训练损失
                "interval": "epoch",
                "frequency": 1,
            },
        }

if __name__ == "__main__":
    print("=== 开始训练 Swin-UNet 模型 ===")
    
    # 方案1: 使用便捷函数创建模型（推荐）
    model = ReconLightningModule(model_size='tiny')  # 可选: 'tiny', 'small', 'base'
    
    # 方案2: 使用自定义配置
    # custom_config = ModelConfig(
    #     in_chans=32*3,  # 32个视角 * 3通道
    #     out_chans=3,    # RGB 输出
    #     img_size=512,
    #     embed_dim=96,
    #     depths=[2, 2, 6, 2],  # Swin-Tiny 配置
    #     num_heads=[3, 6, 12, 24],
    #     window_size=8,  # 确保能被图像尺寸整除
    #     drop_path_rate=0.1
    # )
    # model = ReconLightningModule(custom_config)

    # 创建数据模块
    dm = RefocusDataModule(
        data_dir="data", 
        batch_size=4,  # Swin-UNet 显存占用较大，减小 batch size
        num_workers=4
    )

    # 创建 Trainer
    trainer = L.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=2,  # 使用两块GPU
        strategy="ddp",  # DataDistributedParallel 策略
        
        # 回调函数
        callbacks=[
            # 模型检查点
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="swin-unet-{epoch:02d}-{train_loss:.4f}",
                monitor="train_loss",  # 改为监控训练损失
                mode="min",
                save_top_k=3,
            ),
            # 学习率监控
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        
        # 日志设置
        log_every_n_steps=10,
        
        # 梯度裁剪（对 Transformer 有帮助）
        gradient_clip_val=1.0,
        
        # 累积梯度（如果显存不够）
        # accumulate_grad_batches=2,
    )

    # 打印模型信息
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 开始训练
    trainer.fit(model, dm)
    
    # 可选：测试最佳模型
    # trainer.test(model, dm, ckpt_path="best")
    
    print("=== 训练完成 ===")