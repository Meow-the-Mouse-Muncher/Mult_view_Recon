import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    简化的模型配置类
    """
    in_chans: int = 32*3  # 输入通道数
    out_chans: int = 3  # 输出通道数 (RGB)
    encoder_name: str = 'resnet34'  # 编码器类型
    encoder_weights: str = None  # 预训练权重，None表示随机初始化

class PyramidUNet(nn.Module):
    """
    基于 segmentation_models_pytorch 的金字塔 U-Net 模型
    使用现成的库，简单可靠
    """
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        
        # 使用 SMP 的 U-Net，支持自定义输入通道数
        self.model = smp.Unet(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_chans,
            classes=config.out_chans,
            activation=None,  # 不使用激活函数，输出原始值
        )
        
        # 如果输入通道数不是3，需要修改第一层
        if config.in_chans != 3:
            self._modify_first_conv()
    
    def _modify_first_conv(self):
        """修改第一层卷积以适应自定义输入通道数"""
        # 获取原始第一层卷积
        if hasattr(self.model.encoder, 'conv1'):
            old_conv = self.model.encoder.conv1
            # 创建新的卷积层
            new_conv = nn.Conv2d(
                in_channels=self.config.in_chans,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # 替换第一层
            self.model.encoder.conv1 = new_conv
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [N, C, H, W]
        Returns:
            输出张量 [N, 3, H, W]
        """
        return self.model(x)

# --- 测试与演示代码 ---
if __name__ == '__main__':
    # 测试新的 PyramidUNet
    
    # 1. 定义配置
    config = ModelConfig(
        in_chans=32,
        out_chans=3,
        encoder_name='resnet34',
        encoder_weights=None
    )
    
    # 2. 实例化模型
    print(f"正在构建 PyramidUNet (encoder: {config.encoder_name})...")
    model = PyramidUNet(config=config)
    
    # 3. 验证
    H, W = 512, 512
    dummy_input = torch.randn(2, config.in_chans, H, W)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"--- PyramidUNet 测试 ---")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"编码器: {config.encoder_name}")
    
    assert output.shape == (2, config.out_chans, H, W)
    print("测试通过！")