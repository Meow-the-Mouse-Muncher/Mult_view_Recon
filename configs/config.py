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

"""Default config for training implicit models using OmegaConf."""

from omegaconf import OmegaConf


def get_config():
    """使用 OmegaConf 构建配置对象"""
    conf_dict = {
        "dataset": get_dataset_config(),
        "model": get_model_config(),
        "lightfield": get_lf_config(),
        "train": get_train_config(),
        "eval": get_eval_config(),
        "seed": 42,
        "dev_run": False,
        "trial": 0
    }

    config = OmegaConf.create(conf_dict)
    # 设置为只读，保证实验可追溯性
    OmegaConf.set_readonly(config, True)
    return config


def get_dataset_config():
    """数据配置"""
    return {
        "name": "ff_epipolar",
        "data_dir": "",
        "base_dir": "",
        "scene": "",
        "batch_size": 4,
        "batching": "single_image",
        "use_pixel_centers": False,
        "image_height": -1,
        "image_width": -1,
        "num_interpolation_views": 10
    }


def get_model_config():
    """模型架构配置"""
    return {
        "name": "lfnr",
        "num_rgb_channels": 3,
        "return_attn": False,
        "view_transformer": {
            "layers": 8,
            "heads": 4,
            "embed_dim": 256,
            "key_dim": 130,
            "query_dim": 36,
            "mlp_dim": 256
        },
        "cnn": {
            "use_conv_features": True,
            "conv_feature_dim": 32,
            "ksize": 3,
            "use_learned_embedding": True,
            "embedding_dim": 32,
            "num_train_views": 32
        }
    }


def get_lf_config():
    """光场解析配置"""
    return {
        "name": "lightslab",
        "st_plane": 0.5,
        "uv_plane": 1.0,
        "encoding_name": "positional_encoding",
        "min_deg_point": 0,
        "max_deg_point": 4
    }


def get_train_config():
    """训练超参数配置"""
    return {
        "lr_init": 1.5e-3,
        "weight_decay": 0.0,
        "warmup_steps": 2.5e3,
        "max_steps": 2.5e5,
        "lr_final": 1.0e-5,
        "grad_max_norm": 0,
        "grad_max_val": 0,
        "warmup_epochs": 10,
        "num_epochs": 300,
        "num_rays": 4096,
        "checkpoint_every_steps": 1000,
        "log_loss_every_steps": 500,
        "render_every_steps": 5000,
        "gc_every_steps": 10000
    }


def get_eval_config():
    """评估配置"""
    return {
        "eval_once": False,
        "save_output": True,
        "chunk": 4096,
        "inference": False
    }


