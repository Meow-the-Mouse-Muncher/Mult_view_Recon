import os
import torch
import lightning as L
from torch import nn
from models.recon_model import PyramidUNet, ModelConfig  # 更新为新的模型类
from dataset.lightning_dataset import RefocusDataModule

class ReconLightningModule(L.LightningModule):
    """Lightning 模型包装器，用于训练 PyramidUNet。"""
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model = PyramidUNet(model_config)  # 使用新的模型
        self.criterion = nn.MSELoss()  # 假设损失函数为 MSE

    def forward(self, occ):
        return self.model(occ)  # 输入 occ，输出重建的 GT

    def training_step(self, batch, batch_idx):# 训练循环
        gt, occ = batch  # 从 DataLoader 获取 GT 和 occ
        pred = self(occ)  # 前向传播
        loss = self.criterion(pred, gt)  # 计算损失
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        gt, occ = batch
        pred = self(occ)
        loss = self.criterion(pred, gt)
        self.log('test_loss', loss)

    def validation_step(self, batch, batch_idx):
        gt, occ = batch
        pred = self(occ)
        loss = self.criterion(pred, gt)
        self.log('val_loss', loss)

    def configure_optimizers(self): #定义优化器和学习率调度器
        return torch.optim.Adam(self.parameters(), lr=1e-4)  # 优化器

if __name__ == "__main__":
    # 配置模型
    model_config = ModelConfig(
        in_chans=32*3,  # 根据你的数据调整
        out_chans=3,  # RGB 输出
        encoder_name='resnet34',  # 可选: resnet18, resnet50, efficientnet-b0 等
        encoder_weights=None  # 随机初始化
    )

    # 创建模型
    model = ReconLightningModule(model_config)

    # 创建数据模块
    dm = RefocusDataModule(data_dir="data", batch_size=4, num_workers=4)

    # 创建 Trainer（启用分布式训练，两块GPU）
    trainer = L.Trainer(
        max_epochs=200,  # 训练轮数
        accelerator="gpu",
        devices=2,  # 使用两块GPU
        strategy="ddp",  # DataDistributedParallel 策略
        log_every_n_steps=10
    )

    # 开始训练（自动调用 dm.setup 和数据加载）
    trainer.fit(model, dm)