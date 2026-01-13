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

"""File containing light field utils."""
import torch
import torch.nn as nn

def posenc(x, min_deg, max_deg):
    """
    位置编码 PyTorch 实现。
    Args:
        x: [..., D] 输入张量
        min_deg, max_deg: 频率指数范围
    Returns:
        [..., D + D * (max_deg - min_deg) * 2] 编码后的张量
    """
    if max_deg <= min_deg:
        return x
    
    device = x.device
    dtype = x.dtype
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], device=device, dtype=dtype)
    # x: [..., D] -> [..., D, 1] * [freq] -> [..., D, freq]
    xb = x[..., None] * scales
    # [..., D, freq * 2]
    four_feat = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
    # [..., D + D * freq * 2]
    return torch.cat([x, four_feat.flatten(-2)], dim=-1)

class LightSlab:
  """光线与双平面 (ST-UV) 的交点计算。"""
  def __init__(self, config):
    self.config = config

  def ray_plane_intersection(self, zconst, origins, directions):
    """计算光线与 Z=zconst 平面的交点。"""
    # t = (z_const - O_z) / D_z
    t = (zconst - origins[..., -1]) / (directions[..., -1] + 1e-10)
    # P_xy = O_xy + t * D_xy
    xy = origins[..., :2] + t[..., None] * directions[..., :2]
    return xy

  def ray2lightfield(self, rays_o, rays_d):
    """计算 ST 和 UV 平面的交点，拼接成 4 维向量。"""
    st = self.ray_plane_intersection(self.config.st_plane, rays_o, rays_d)
    uv = self.ray_plane_intersection(self.config.uv_plane, rays_o, rays_d)
    return torch.cat([st, uv], dim=-1) # [..., 4]

  def encode(self, lf_samples):
    """特征编码入口。"""
    if self.config.encoding_name == "positional_encoding":
      return posenc(lf_samples, self.config.min_deg_point, self.config.max_deg_point)
    return lf_samples

  def get_lf_encoding(self, rays_o, rays_d):
    """对光场进行编码。"""
    lf_samples = self.ray2lightfield(rays_o, rays_d)
    lf_samples_enc = self.encode(lf_samples)
    return lf_samples, lf_samples_enc, None # Mask 暂时不用，设为 None

def get_lightfield_obj(lf_config):
  """工厂函数，返回 LightSlab 实例。"""
  if lf_config.name == "lightslab":
    return LightSlab(lf_config)
  else:
    raise ValueError(f"Parametrization: {lf_config.name} not supported.")
