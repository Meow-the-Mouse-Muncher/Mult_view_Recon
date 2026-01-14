import os
import torch
import lightning as L
from torch import nn
from models.lfnr import LFNR
from dataset.LF_dataset import LFDataModule
from configs.config import get_config
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import h5py # 确保顶部导入
torch.set_float32_matmul_precision('high')

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
        
        # 指标初始化
        metrics_kwargs = {"data_range": 1.0}
        self.psnr = PeakSignalNoiseRatio(**metrics_kwargs)
        self.ssim = StructuralSimilarityIndexMeasure(**metrics_kwargs)
        
        # [新增] 验证集结果缓存
        # 结构: { file_idx: { 'preds': [], 'gts': [], 'center': ... } }
        self.val_outputs = {}

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, batch_idx):
        """训练步骤 (Sparse Ray Training)"""
        # 1. 前向传播
        # 返回值: 预测RGB, 重叠区RGB(如有)
        pred_rgb, rgb_overlap = self(batch)
        gt_rgb = batch['gt_rgb'] # [B, n_rays, 3]
        
        # 2. 计算损耗
        # 主预测损失
        loss_pred = self.mse_loss(pred_rgb, gt_rgb)
        
        # 重叠区域/辅助损失 (如果模型支持)
        loss_overlap = self.mse_loss(rgb_overlap, gt_rgb)
        # 总损失 (L2权重衰减已在AdamW中处理)
        loss = loss_pred + loss_overlap
        
        # 3. 记录日志
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/loss_pred', loss_pred, sync_dist=True)
        self.log('train/loss_overlap', loss_overlap, sync_dist=True)
            
        return loss

    # === DDP 兼容的验证逻辑 ===
    
    def on_validation_epoch_start(self):
        # 改用列表存储，方便 stack
        self.val_step_outputs = [] 

    def validation_step(self, batch, batch_idx):
        pred_chunk, _ = self(batch)
        # 仅收集预测、GT和索引，不收集 Center 图
        self.val_step_outputs.append({
            'pred': pred_chunk.detach(),
            'gt': batch['gt_rgb'].detach(),
            'file_idx': batch['meta_file_idx'],
            'start_idx': batch['meta_start']
        })

    def on_validation_epoch_end(self):
        """
        全卡收集 -> CPU 拼图 -> 计算指标 -> 可视化
        """
        
        # [修复] 如果是在进行 Sanity Check (只跑几个 batch), 数据肯定不全
        # 此时强制跳过繁重的拼图和可视化逻辑
        if self.trainer.sanity_checking:
            self.val_step_outputs.clear()
            return
            
        # 1. 在单卡内部 Stack 起来
        if not self.val_step_outputs: return
        
        # 1. DDP 收集
        local_preds = torch.cat([x['pred'] for x in self.val_step_outputs], dim=0)
        local_gts = torch.cat([x['gt'] for x in self.val_step_outputs], dim=0)
        local_fidxs = torch.cat([x['file_idx'] for x in self.val_step_outputs], dim=0)
        local_starts = torch.cat([x['start_idx'] for x in self.val_step_outputs], dim=0)
        
        global_preds = self.all_gather(local_preds).view(-1, local_preds.shape[1], 3).cpu()
        global_gts = self.all_gather(local_gts).view(-1, local_gts.shape[1], 3).cpu()
        global_fidxs = self.all_gather(local_fidxs).view(-1).cpu()
        global_starts = self.all_gather(local_starts).view(-1).cpu()

        # 2. 仅在 Rank 0 拼图、计算指标和可视化
        if self.global_rank == 0:
            val_h5_files = self.trainer.datamodule.val_dataset.h5_files
            unique_files = torch.unique(global_fidxs)
            
            # [新增] 累加器
            total_psnr = 0.0
            total_ssim = 0.0
            num_images = 0

            for fid in unique_files:
                if fid < 0: continue
                mask = (global_fidxs == fid)
                
                # 拼图逻辑
                sort_idx = torch.argsort(global_starts[mask])
                img_p = global_preds[mask][sort_idx].reshape(-1, 3)
                img_g = global_gts[mask][sort_idx].reshape(-1, 3)
                
                H = int(img_p.shape[0]**0.5)
                if H*H != img_p.shape[0]: continue
                
                # 转换为图像维度 [1, 3, H, W]
                view_p = img_p.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1).to(self.device)
                view_g = img_g.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1).to(self.device)
                
                # === [新增] 计算单张图的指标 ===
                cur_psnr = self.psnr(view_p, view_g)
                cur_ssim = self.ssim(view_p, view_g)
                
                total_psnr += cur_psnr
                total_ssim += cur_ssim
                num_images += 1
                
                # 可视化第一张图
                if num_images == 1:
                    h5_path = val_h5_files[fid]
                    with h5py.File(h5_path, 'r') as f:
                        center_img = torch.from_numpy(f['occ_center/rgb'][:]).float() / 255.0
                        center_view = center_img.permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    if center_view.shape[2:] != view_p.shape[2:]:
                        print(f"[Warning] Skip Vis: Shape mismatch Ref{center_view.shape} vs Pred{view_p.shape}")
                    else:
                        grid = torch.cat([center_view, view_p, view_g], dim=3)
                        self.logger.experiment.add_image('val/Comparison', grid[0], self.global_step)
            
            # === [新增] 记录平均指标 ===
            if num_images > 0:
                avg_psnr = total_psnr / num_images
                avg_ssim = total_ssim / num_images
                # rank_zero_only=True 避免多卡重复记录
                self.log('val/psnr', avg_psnr, rank_zero_only=True)
                self.log('val/ssim', avg_ssim, rank_zero_only=True)

        # 清空
        self.val_step_outputs.clear()

    # === 验证逻辑结束 ===
    
    def test_step(self, batch, batch_idx):
        """测试步骤（逻辑同 Val）"""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """配置带线性预热和余弦退火的学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.train.lr_init,
            weight_decay=self.config.train.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        warmup_steps = self.config.train.warmup_steps
        max_steps = self.config.train.max_steps
        
        # 1. 线性预热调度器: 在 warmup_steps 内从 lr_init * 0.01 增加到 lr_init
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            total_iters=warmup_steps
        )
        
        # 2. 余弦退火调度器: 从 warmup_steps 开始，在剩余步数内降至 lr_final
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_steps - warmup_steps,
            eta_min=self.config.train.lr_final
        )
        
        # 3. 顺序组合调度器
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # 关键：设置为按 step 更新
            },
        }

    # def on_after_backward(self):
    #     # 仅在训练的第一步运行一次检查
    #     if self.global_step == 0:
    #         print("\n" + "="*50)
    #         print("正在检测未使用的模型参数 (grad is None):")
    #         unused_params = []
    #         for name, param in self.named_parameters():
    #             if param.grad is None:
    #                 unused_params.append(name)
    #                 print(f"🚩 未使用的参数: {name}")
            
    #         if not unused_params:
    #             print("✅ 完美！所有参数都参与了梯度计算。")
    #         else:
    #             print(f"\n共发现 {len(unused_params)} 个未使用的参数。")
    #         print("="*50 + "\n")

if __name__ == "__main__":
    print("=== 开始训练 LFNR 模型 ===")
    
    # 加载配置
    config = get_config()
    mode = "rot_arc" # mode =[fix_line,rot_arc,rot_line]
    # 1. 初始化模型包装器
    model = LFModule(config=config, n_rays=config.train.num_rays)
    
    # 2. 创建数据模块
    dm = LFDataModule(
        data_dir="data",
        model=mode,
        batch_size=1,
        num_workers=4,
        n_rays=config.train.num_rays,
        val_chunk_size=config.eval.chunk 
    )

    # 创建 Trainer
    trainer = L.Trainer(
        max_epochs=config.train.num_epochs,
        accelerator="gpu",
        devices=2,  
        strategy="ddp",
        logger=TensorBoardLogger("logs", name=mode, version=None), 
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=os.path.join("checkpoints", mode),
                filename="lfnr-{epoch:02d}",
                monitor="epoch",  # 监控 epoch 数量
                mode="max",       # 保存 epoch 最大的（也就是最新的）
                save_top_k=4,
                every_n_epochs=5
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        log_every_n_steps=20,
        check_val_every_n_epoch=1, 
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
    
    # # 可选：测试最佳模型
    # trainer.test(model, dm, ckpt_path="best")
    
    print("=== 训练完成 ===")