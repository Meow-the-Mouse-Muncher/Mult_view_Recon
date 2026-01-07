import torch
import torch.nn as nn
from dataclasses import dataclass
from models.vision_transformer import SwinUnet

@dataclass
class ModelConfig:
    """
    Swin-UNet 模型配置
    """
    # 32个视角 * 3通道 = 96
    in_chans: int = 32 * 3  
    out_chans: int = 3
    img_size: int = 512
    patch_size: int = 4
    embed_dim: int = 96
    depths: list = None
    depths_decoder: list = None
    num_heads: list = None
    window_size: int = 8  # 改为 8，512 可以被 8 整除
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    
    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 2, 6, 2]  # Swin-T 配置
        if self.depths_decoder is None:
            self.depths_decoder = [2, 6, 2, 2]
        if self.num_heads is None:
            self.num_heads = [3, 6, 12, 24]  # Swin-T 配置

class SimpleConfig:
    """简化的配置类，兼容 SwinUnet"""
    def __init__(self, config: ModelConfig):
        # DATA 配置
        self.DATA = type('DATA', (), {})()
        self.DATA.IMG_SIZE = config.img_size
        
        # MODEL 配置
        self.MODEL = type('MODEL', (), {})()
        self.MODEL.DROP_RATE = config.drop_rate
        self.MODEL.DROP_PATH_RATE = config.drop_path_rate
        self.MODEL.PRETRAIN_CKPT = None
        
        # SWIN 配置
        self.MODEL.SWIN = type('SWIN', (), {})()
        self.MODEL.SWIN.PATCH_SIZE = config.patch_size
        self.MODEL.SWIN.IN_CHANS = config.in_chans
        self.MODEL.SWIN.EMBED_DIM = config.embed_dim
        self.MODEL.SWIN.DEPTHS = config.depths
        self.MODEL.SWIN.NUM_HEADS = config.num_heads
        self.MODEL.SWIN.WINDOW_SIZE = config.window_size
        self.MODEL.SWIN.MLP_RATIO = config.mlp_ratio
        self.MODEL.SWIN.QKV_BIAS = config.qkv_bias
        self.MODEL.SWIN.QK_SCALE = config.qk_scale
        self.MODEL.SWIN.APE = config.ape
        self.MODEL.SWIN.PATCH_NORM = config.patch_norm
        
        # TRAIN 配置
        self.TRAIN = type('TRAIN', (), {})()
        self.TRAIN.USE_CHECKPOINT = config.use_checkpoint

class PyramidUNet(nn.Module):
    """
    基于现有 SwinUnet 的包装器
    """
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        
        print(f"构建 Swin-UNet 模型:")
        print(f"  输入通道: {config.in_chans}")
        print(f"  输出通道: {config.out_chans}")
        print(f"  图像尺寸: {config.img_size}")
        print(f"  嵌入维度: {config.embed_dim}")
        print(f"  深度: {config.depths}")
        print(f"  注意力头数: {config.num_heads}")
        
        # 创建兼容的配置
        simple_config = SimpleConfig(config)
        
        # 使用现有的 SwinUnet
        self.swin_unet = SwinUnet(
            config=simple_config,
            img_size=config.img_size,
            num_classes=config.out_chans,
            zero_head=False,
            vis=False
        )
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: [N, 96, H, W] - 32个视角的多视图输入
        Returns:
            [N, 3, H, W] - 重建的RGB图像
        """
        # 确保输入维度正确
        if x.dim() != 4:
            raise ValueError(f"期望4D输入 [N, C, H, W], 得到 {x.dim()}D")
        
        if x.size(1) != self.config.in_chans:
            raise ValueError(f"期望输入通道数 {self.config.in_chans}, 得到 {x.size(1)}")
            
        # 通过 SwinUnet
        output = self.swin_unet(x)
        
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设 float32
            'config': self.config
        }

# 预定义的模型配置
class SwinUNetConfigs:
    """预定义的 Swin-UNet 配置"""
    
    @staticmethod
    def swin_tiny_512():
        """Swin-Tiny 配置，适用于 512x512 输入"""
        return ModelConfig(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            depths_decoder=[2, 6, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8  # 512 可以被 8 整除
        )
    
    @staticmethod
    def swin_small_512():
        """Swin-Small 配置，适用于 512x512 输入"""
        return ModelConfig(
            embed_dim=96,
            depths=[2, 2, 18, 2],
            depths_decoder=[2, 18, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8  # 512 可以被 8 整除
        )
    
    @staticmethod
    def swin_base_512():
        """Swin-Base 配置，适用于 512x512 输入"""
        return ModelConfig(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            depths_decoder=[2, 18, 2, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8  # 512 可以被 8 整除
        )

# 便捷的模型创建函数
def create_swin_unet(model_size='small', in_chans=96, out_chans=3, img_size=512):
    """
    创建 Swin-UNet 模型
    
    Args:
        model_size: 'tiny', 'small', 'base'
        in_chans: 输入通道数
        out_chans: 输出通道数  
        img_size: 输入图像尺寸
    """
    if model_size == 'tiny':
        config = SwinUNetConfigs.swin_tiny_512()
    elif model_size == 'small':
        config = SwinUNetConfigs.swin_small_512()
    elif model_size == 'base':
        config = SwinUNetConfigs.swin_base_512()
    else:
        raise ValueError(f"不支持的模型尺寸: {model_size}")
    
    # 更新配置
    config.in_chans = in_chans
    config.out_chans = out_chans
    config.img_size = img_size
    
    return PyramidUNet(config)

# 测试代码
if __name__ == "__main__":
    print("=== 使用现有的 SwinUnet 实现 ===")
    
    # 测试默认配置
    try:
        config = ModelConfig()
        model = PyramidUNet(config)
        
        # 打印模型信息
        info = model.get_model_info()
        print(f"参数总数: {info['total_params']:,}")
        print(f"可训练参数: {info['trainable_params']:,}")
        print(f"模型大小: {info['model_size_mb']:.2f} MB")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 96, 512, 512)
        print(f"输入形状: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"输出形状: {output.shape}")
            
        print("✅ 模型测试成功!")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    # 测试便捷创建函数
    print(f"\n=== 测试便捷创建函数 ===")
    try:
        model = create_swin_unet('tiny', in_chans=96, out_chans=3, img_size=512)
        dummy_input = torch.randn(2, 96, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"便捷函数创建成功: 输入 {dummy_input.shape} -> 输出 {output.shape}")
        print("✅ 便捷函数测试成功!")
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        import traceback
        traceback.print_exc()