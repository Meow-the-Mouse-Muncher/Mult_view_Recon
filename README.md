configs
职责：保存训练/推理/数据预处理配置（yaml/json）。
建议：保持单一来源配置（config.yaml），并添加 defaults/ 或 experiments/ 子文件夹管理多试验配置。
models
职责：定义模型类、网络构件、权重加载/保存接口。
建议：将模型拆成 backbones/、heads/、modules/（可选）；recon_model.py 保持为组合入口并导出标准接口（如 build_model(cfg)）。
dataset
职责：数据集类、数据预处理脚本、数据生成/切分工具。
建议：在该目录下添加 datasets.py（包含 Dataset 子类）、transforms.py、loader.py（构建 DataLoader），并明确数据格式（.npy/.pt/image）。将原始样本放到 data，把代码和清单放在 dataset。
data
职责：存放原始与中间数据（不放代码）。
建议：按照 train/val/test 子目录组织，或 raw/ 和 processed/ 两级目录；在 README 或 dataset/README.md 说明格式与命名约定。
utils
职责：通用工具（日志、可视化、metric、loss、checkpoint helper）。
建议：拆分为 io.py、visualization.py、metrics.py、losses.py、misc.py，并提供统一的 set_seed()、save_checkpoint() 接口。
train.py
职责：训练流程编排（解析 configs、构建 dataset/model/optimizer/scheduler、训练循环或 Lightning Trainer）。
建议：简化入口，调用 scripts/train.py 或 train/train_loop.py 以便更好组织；支持命令行覆盖配置（如 --cfg、--resume）。
checkpoints 与 logs
职责：运行时产物存放。
建议：在 .gitignore 中忽略模型和大日志；在训练脚本中按 experiments/{exp_name}/{checkpoints,logs} 组织输出。