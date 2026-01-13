# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Light Field Neural Rendering model."""

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from . import transformer
from utils import lf_utils

class LFNR(nn.Module):
    """Light Field Neural Rendering model in PyTorch."""
    def __init__(self, config):
        super().__init__()
    
        # 将各子配置直接挂载，简化引用
        self.config = config
        self.m_config = config.model
        self.cnn_config = config.model.cnn
        self.lf_config = config.lightfield
        
        # 获取返回权限
        self.return_attn = getattr(self.m_config, 'return_attn', False)
        
        # 工具和模块初始化：直接传对应的子配置段
        self.lightfield = lf_utils.get_lightfield_obj(self.lf_config)
        
        # Transformer 定义
        self.view_transformer = transformer.SelfAttentionTransformer(self.m_config.view_transformer)
        
        # 后续层使用点号访问对应配置
        self.view_correspondence = nn.Linear(self.m_config.view_transformer.embed_dim*2, 1)
        self.rgb_dense = nn.Linear(self.m_config.view_transformer.embed_dim, self.m_config.num_rgb_channels)
        
        # Transforms for Query/Key mapping
        self.key_transform = nn.Linear(self.m_config.view_transformer.key_dim, self.m_config.view_transformer.embed_dim)   
        self.query_transform = nn.Linear(self.m_config.view_transformer.query_dim, self.m_config.view_transformer.embed_dim)
        
        if self.cnn_config.use_learned_embedding:
             self.camera_embedding = nn.Embedding(
                 num_embeddings=self.cnn_config.num_train_views,
                 embedding_dim=self.cnn_config.embedding_dim 
             )

        if self.cnn_config.use_conv_features:
            self.conv_layer = nn.Conv2d(
                in_channels=3, 
                out_channels=self.cnn_config.conv_feature_dim,
                kernel_size=self.cnn_config.ksize,
                padding=self.cnn_config.ksize // 2
            )
            self.feature_activation = nn.ELU()

    def _get_query(self, rays_o, rays_d):
        """将目标光线编码为光场特征。"""
        _, q_enc, _ = self.lightfield.get_lf_encoding(rays_o, rays_d)
        return q_enc
    def _get_key(self, projected_rgb_and_feat, ref_rays_o, ref_rays_d, pts_3d):
        """参考视角下的 光场编码特征||世界坐标编码||rgb和cnn特征||相机嵌入特征 """
        B, N, n_rays, _ = projected_rgb_and_feat.shape

        # 1. 光场编码 (针对参考光线): [B, N, n_rays, D_lf] D_lf=36
        _, k_enc, _ = self.lightfield.get_lf_encoding(ref_rays_o, ref_rays_d)

        # 2. 3D 点地理位置编码: [B, n_rays, D_w] D_w=27
        wcoords_enc_raw = lf_utils.posenc(
            pts_3d,
            self.lf_config.min_deg_point,
            self.lf_config.max_deg_point,
        )
        # 扩展维度以匹配参考视角 N: [B, N, n_rays, D_w]
        wcoords_enc = wcoords_enc_raw.unsqueeze(1).expand(-1, N, -1, -1)

        # 3. 学习到的相机嵌入 (Camera Embedding): [N, D_cam]
        # 注意：这里使用 arange(N)，确保每个视角有独立的 ID
        indices = torch.arange(N, device=projected_rgb_and_feat.device)
        learned_embed_raw = self.camera_embedding(indices) # [N, D_cam]
        
        # 扩展到 Batch 和 Rays 维度: [B, N, n_rays, D_cam]
        learned_embed = learned_embed_raw[None, :, None, :].expand(B, -1, n_rays, -1)

        # 4. 拼接所有特征 [B, N, n_rays, D_total]
        # k_enc: [B, N, n_rays, D_lf]
        # wcoords_enc: [B, N, n_rays, D_w]
        # projected_rgb_and_feat: [B, N, n_rays, C+3]
        # learned_embed: [B, N, n_rays, D_cam]
        input_k = torch.cat([
            k_enc, 
            wcoords_enc, 
            projected_rgb_and_feat, 
            learned_embed
        ], dim=-1)

        return input_k,learned_embed

    def _get_pixel_projection(self, sampling_grid, ref_images):
        """
        从参考图像中采样 RGB 和 CNN 特征。
        """
        B, N, C, H, W = ref_images.shape
        ref_images_flat = ref_images.reshape(B * N, C, H, W)
        grid_flat = sampling_grid.reshape(B * N, -1, 1, 2)

        # 1. 采样原始 RGB
        projected_rgb = F.grid_sample(ref_images_flat, grid_flat, align_corners=True, mode='bilinear')
        projected_rgb = projected_rgb.squeeze(-1).permute(0, 2, 1) # [B*N, n_rays, 3]

        # 2. 提取并采样 CNN 特征
        if self.cnn_config.use_conv_features:
            ref_features = self.feature_activation(self.conv_layer(ref_images_flat)) #[B*N, feat_dim, H, W]
            projected_features = F.grid_sample(ref_features, grid_flat, align_corners=True, mode='bilinear')
            projected_features = projected_features.squeeze(-1).permute(0, 2, 1) # [B*N, n_rays, feat_dim]
            out = torch.cat([projected_features, projected_rgb], dim=-1)
        else:
            out = projected_rgb
            
        return out.reshape(B, N, -1, out.shape[-1])
    def _predict_color(self, input_q, input_k):
        """
        基于 Transformer 的注意力机制预测颜色。
        Args:
            input_q: [B, n_rays, q_dim]
            input_k: [B, N, n_rays, k_dim]
        Returns:
            rgb: [B, n_rays, 3]
            attn_weights: [B, n_rays, N, 1]
        """
        B, N, n_rays, _ = input_k.shape
        
        # 1. 投影到统一的嵌入空间
        # q: [B, n_rays, q_dim] -> [B, n_rays, 1, embed_dim] -> [(B*n_rays), 1, embed_dim]
        q = self.query_transform(input_q).unsqueeze(2).reshape(B * n_rays, 1, -1)
        
        # k: [B, N, n_rays, k_dim] -> [B, n_rays, N, k_dim] -> [(B*n_rays), N, embed_dim]
        # 注意这里要先 permute 把视角维度 N 换到最后，然后再投影和 reshape
        k = input_k.permute(0, 2, 1, 3).reshape(B * n_rays, N, -1)
        k = self.key_transform(k) # [(B*n_rays), N, embed_dim]
        
        # 2. 现在 q 和 k 都是 3D 张量且维度匹配了
        # q_k_combined: [(B*n_rays), (1 + N), embed_dim]
        q_k_combined = torch.cat([q, k], dim=1)

        # 3. 通过 View Transformer
        out = self.view_transformer(q_k_combined) # [(B*n_rays), 1+N, embed_dim]

        # 4. 分离更新后的 Query 和 Keys
        refined_query = out[:, 0:1, :]           # [(B*n_rays), 1, embed_dim]
        refined_key = out[:, 1:, :]             # [(B*n_rays), N, embed_dim]

        # 5. 计算注意力权重
        # 扩展 Query 以对齐每个 Key: [(B*n_rays), N, embed_dim]
        refined_query_expanded = refined_query.expand(-1, N, -1)
        concat_feat = torch.cat([refined_query_expanded, refined_key], dim=-1)
        
        # attn_weights: [(B*n_rays), N, 1]
        attn_weights = self.view_correspondence(concat_feat)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 6. 聚合特征并生成颜色
        # combined_feature: [(B*n_rays), embed_dim]
        combined_feature = (refined_key * attn_weights).sum(dim=1)
        
        # 恢复到原始 batch 形状并激活
        raw_rgb = self.rgb_dense(combined_feature) # [(B*n_rays), 3]
        rgb = torch.sigmoid(raw_rgb).reshape(B, n_rays, 3) 

        # 恢复权重形状以供后续使用
        attn_weights = attn_weights.reshape(B, n_rays, N, 1).permute(0, 2, 1, 3) # [B, N, n_rays, 1]

        return rgb, attn_weights

    def _overlap_color(self, projected_rgb_and_feat, n_attn):
        """基于注意力权重计算重叠颜色。"""
        projected_rgb = projected_rgb_and_feat[..., -3:] # [B, N, n_rays, 3]
        rgb_overlap = (projected_rgb * n_attn).sum(dim=1) # [B, n_rays, 3]
        return rgb_overlap


    def forward(self, batch):
        """前向传播逻辑。"""
        # 1. 解包数据
        target_rays_o = batch['gt_rays_o']  # [B, n_rays, 3]
        target_rays_d = batch['gt_rays_d']  # [B, n_rays, 3]
        ref_images = batch['occ_rgb']       # [B, N, 3, H, W]
        sampling_grid = batch['sampling_grid'] # [B, N, n_rays, 2]
        ref_rays_o = batch['occ_rays_o']    # [B, N, n_rays, 3]
        ref_rays_d = batch['occ_rays_d']    # [B, N, n_rays, 3]
        pts_3d = batch['pts_3d']        # [B, n_rays, 3]
        K = batch['K']                    # [B, 3, 3]

        # 2. Query 编码 (目标光线) q_dim = 36
        input_q = self._get_query(target_rays_o, target_rays_d) # [B, n_rays, q_dim]

        # 3. Key 编码 (参考图像采样 + 参考光线方向)
        # 采样颜色和cnn特征 # [B, N, n_rays, feat_dim] feat_dim=35
        projected_rgb_and_feat = self._get_pixel_projection(sampling_grid,
                                                    ref_images)
        input_k, learned_embed = self._get_key(projected_rgb_and_feat,ref_rays_o, ref_rays_d, pts_3d) # [B, N, n_rays, k_dim] k_dim=130

        rgb,n_attn= self._predict_color(input_q, input_k)

        rgb_overlap = self._overlap_color(projected_rgb_and_feat, n_attn)
        return rgb, rgb_overlap, n_attn, learned_embed

