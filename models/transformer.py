import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """Transformer MLP 块。"""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfAttentionTransformerLayer(nn.Module):
    """单个 Transformer 层。"""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        # PyTorch 的 MultiheadAttention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.mlp = Mlp(
            in_features=embed_dim, 
            hidden_features=mlp_dim, 
            out_features=embed_dim, 
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 遵循 Flax 版的 Post-Norm 结构: Norm(x + Attn(x))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


class SelfAttentionTransformer(nn.Module):
    """PyTorch 实现的 Self Attention Transformer。"""
    def __init__(self, config):
        """
        Args:
            config: 包含 layers, heads, embed_dim, mlp_dim 等字段的配置对象
        """
        super().__init__()
        # 从 OmegaConf 对象提取参数
        num_layers = config.layers
        embed_dim = config.embed_dim
        num_heads = config.heads
        mlp_dim = config.mlp_dim
        dropout = getattr(config, 'dropout_rate', 0.0)

        # 构建层堆栈
        self.layers = nn.ModuleList([
            SelfAttentionTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, N, C] 输入张量
        Returns:
            x: 处理后的张量
        """
        for layer in self.layers:
            x = layer(x)
        return x
