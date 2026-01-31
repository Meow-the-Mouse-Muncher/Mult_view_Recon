import os
import time
import torch
import lightning as L
from torch import nn
from models.lfnr import LFNR
from dataset.LF_dataset import LFDataModule
from configs.config import get_config
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import h5py 
from torchvision.utils import save_image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_float32_matmul_precision('high')

class LFModule(L.LightningModule):
    """Lightning 模型包装器，用于训练 LFNR。"""
    def __init__(self, config, n_rays=4096, save_dir="pred_data"):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.n_rays = n_rays
        self.save_dir = save_dir
        self.learning_rate = config.train.lr_init
        
        # 初始化模型
        self.model = LFNR(config=config)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        # 指标初始化
        metrics_kwargs = {"data_range": 1.0}
        self.psnr = PeakSignalNoiseRatio(**metrics_kwargs)
        self.ssim = StructuralSimilarityIndexMeasure(**metrics_kwargs)
        
        # 缓存
        self.val_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, batch_idx):
        pred_rgb, rgb_overlap = self(batch)
        gt_rgb = batch['gt_rgb'] 
        
        loss_pred = self.mse_loss(pred_rgb, gt_rgb)
        loss_overlap = self.mse_loss(rgb_overlap, gt_rgb)
        loss = loss_pred + loss_overlap
        
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/loss_pred', loss_pred, sync_dist=True)
        self.log('train/loss_overlap', loss_overlap, sync_dist=True)
            
        return loss

    # === DDP 兼容的验证逻辑 ===
    
    def on_validation_epoch_start(self):
        self.val_step_outputs = [] 

    def validation_step(self, batch, batch_idx):
        pred_chunk, _ = self(batch)
        self.val_step_outputs.append({
            'pred': pred_chunk.detach(),
            'gt': batch['gt_rgb'].detach(),
            'file_idx': batch['meta_file_idx'],
            'start_idx': batch['meta_start']
        })

    def on_validation_epoch_end(self):
        """
        全卡收集 -> CPU 拼图 -> 计算指标 -> 可视化 (Center | Pred | GT)
        """
        # 如果是在进行 Sanity Check (只跑几个 batch), 数据肯定不全
        # 此时强制跳过繁重的拼图和可视化逻辑
        if self.trainer.sanity_checking:
            self.val_step_outputs.clear()
            return

        if not self.val_step_outputs:
            return
            
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
            # 验证阶段 val_dataset 肯定存在
            val_h5_files = self.trainer.datamodule.val_dataset.h5_files
            unique_files = torch.unique(global_fidxs)
            
            # [恢复] 累加器
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
                
                # 转换为图像维度 [1, 3, H, W]，移回 GPU 计算指标
                view_p = img_p.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1).to(self.device)
                view_g = img_g.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1).to(self.device)
                
                # === [恢复] 计算单张图的指标 ===
                cur_psnr = self.psnr(view_p, view_g)
                cur_ssim = self.ssim(view_p, view_g)
                
                total_psnr += cur_psnr
                total_ssim += cur_ssim
                num_images += 1
                
                # [恢复] 可视化第一张图 (Center | Pred | GT)
                if num_images == 1:
                    h5_path = val_h5_files[fid]
                    # 尝试读取 Center View
                    center_view = None
                    try:
                        with h5py.File(h5_path, 'r') as f:
                            if 'occ_center/rgb' in f:
                                center_img = torch.from_numpy(f['occ_center/rgb'][:]).float() / 255.0
                                center_view = center_img.permute(2, 0, 1).unsqueeze(0).to(self.device)
                    except Exception as e:
                        print(f"[Vis Warning] Could not read center view: {e}")

                    # 检查形状匹配并拼图
                    if center_view is not None and center_view.shape[2:] == view_p.shape[2:]:
                         # 三联图
                         grid = torch.cat([center_view, view_p, view_g], dim=3)
                    else:
                         # 只有 Pred 和 GT
                         grid = torch.cat([view_p, view_g], dim=3)

                    self.logger.experiment.add_image('val/Comparison', grid[0], self.global_step)
            
            # === [恢复] 记录平均指标 ===
            if num_images > 0:
                avg_psnr = total_psnr / num_images
                avg_ssim = total_ssim / num_images
                # rank_zero_only=True 避免多卡重复记录
                self.log('val/psnr', avg_psnr, rank_zero_only=True)
                self.log('val/ssim', avg_ssim, rank_zero_only=True)

        # 清空
        self.val_step_outputs.clear()

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        # 必须独立实现，因为我们要保存用于拼图的所有 chunks
        pred_chunk, _ = self(batch)
        self.test_step_outputs.append({
            'pred': pred_chunk.detach(),
            'gt': batch['gt_rgb'].detach(),
            'file_idx': batch['meta_file_idx'], # 哪张图
            'start_idx': batch['meta_start']    # 哪个位置
        })

    def on_test_epoch_end(self):
        if not self.test_step_outputs: return

        # 1. 聚合所有卡上的 Chunks
        local_preds = torch.cat([x['pred'] for x in self.test_step_outputs], dim=0)
        local_gts = torch.cat([x['gt'] for x in self.test_step_outputs], dim=0)
        local_fidxs = torch.cat([x['file_idx'] for x in self.test_step_outputs], dim=0)
        local_starts = torch.cat([x['start_idx'] for x in self.test_step_outputs], dim=0)

        # 移动到 CPU 并聚合 (防止 OOM)
        global_preds = self.all_gather(local_preds).view(-1, local_preds.shape[1], 3).cpu()
        global_gts = self.all_gather(local_gts).view(-1, local_gts.shape[1], 3).cpu()
        global_fidxs = self.all_gather(local_fidxs).view(-1).cpu()
        global_starts = self.all_gather(local_starts).view(-1).cpu()

        # 2. 仅在 Rank 0 处理拼图和保存图片
        if self.global_rank == 0:
            # [修正点]：安全获取 test filenames
            test_h5_files = []
            if hasattr(self.trainer.datamodule, 'test_dataset'):
                test_h5_files = self.trainer.datamodule.test_dataset.h5_files
            else:
                print("Warning: test_dataset not found in DataModule. Filenames will be unavailable.")

            unique_files = torch.unique(global_fidxs) # 使用全局索引
            print(f"正在处理 {len(unique_files)} 张测试图像...")
            
            # 使用 log 文件记录指标
            os.makedirs(self.save_dir, exist_ok=True)
            log_path = os.path.join(self.save_dir, "metrics.txt")
            
            with open(log_path, "w") as f:
                f.write("Filename, PSNR, SSIM\n")
                
                total_psnr = 0
                total_ssim = 0
                count = 0

                for fid in unique_files:
                    fid = int(fid.item())
                    if fid < 0: continue
                    
                    mask = (global_fidxs == fid)
                    
                    current_starts = global_starts[mask]
                    sort_idx = torch.argsort(current_starts)
                    
                    img_p = global_preds[mask][sort_idx].reshape(-1, 3)
                    img_g = global_gts[mask][sort_idx].reshape(-1, 3)
                    
                    H = int(img_p.shape[0]**0.5)
                    if H*H != img_p.shape[0]: 
                        print(f"Skipping fid={fid}, shape mismatch: {img_p.shape}")
                        continue
                    
                    # 转换为图像维度 (此时在 CPU 上)
                    view_p = img_p.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1)
                    view_g = img_g.view(1, H, H, 3).permute(0, 3, 1, 2).clamp(0, 1)
                    
                    cur_psnr = self.psnr(view_p, view_g).item()
                    cur_ssim = self.ssim(view_p, view_g).item()
                    
                    total_psnr += cur_psnr
                    total_ssim += cur_ssim
                    count += 1
                    
                    # 获取文件名
                    if fid < len(test_h5_files):
                        h5_path = test_h5_files[fid]
                        fname = os.path.splitext(os.path.basename(h5_path))[0]
                        
                        # [新增] 尝试读取 Center View 用于同步三拼图可视化
                        center_view = None
                        try:
                            with h5py.File(h5_path, 'r') as h5f:
                                if 'occ_center/rgb' in h5f:
                                    center_img = torch.from_numpy(h5f['occ_center/rgb'][:]).float() / 255.0
                                    center_view = center_img.permute(2, 0, 1).unsqueeze(0)
                        except Exception as e:
                            print(f"[Test Vis Warning] Could not read center view for {fname}: {e}")
                    else:
                        fname = f"unknown_{fid}"
                        center_view = None

                    # 保存图片 (根据是否有 center_view 决定是三拼还是双拼)
                    save_path = os.path.join(self.save_dir, f"{fname}.png")
                    if center_view is not None and center_view.shape[2:] == view_p.shape[2:]:
                        grid = torch.cat([center_view, view_p, view_g], dim=3)
                    else:
                        grid = torch.cat([view_p, view_g], dim=3)
                    
                    save_image(grid, save_path)
                    
                    log_str = f"{fname}, {cur_psnr:.4f}, {cur_ssim:.4f}"
                    f.write(log_str + "\n")
                
                if count > 0:
                    avg_psnr = total_psnr / count
                    avg_ssim = total_ssim / count
                    final_log = f"\nAverage: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}\n"
                    print(final_log)
                    f.write(final_log)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """配置带线性预热和余弦退火的学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
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
                "interval": "step",
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
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="LF", help="消融实验名称")
    parser.add_argument("--mode", type=str, default="rot_arc", help="数据模式: fix_line, rot_arc, rot_line, mix")
    args, _ = parser.parse_known_args()

    print(f"=== 开始训练 LFNR 模型 | 实验: {args.exp_name} | 模式: {args.mode} ===")
    
    # 1. 加载参数
    config = get_config()
    exp_name = args.exp_name
    mode = args.mode
    
    # 2. 构造路径: 实验名/mode
    result_save_dir = os.path.join("pred_data", exp_name, mode)
    checkpoint_dir = os.path.join("checkpoints", exp_name, mode)
    os.makedirs(result_save_dir, exist_ok=True)

    # 初始化模型包装器
    model = LFModule(
        config=config, 
        n_rays=config.train.num_rays,
        save_dir=result_save_dir
    )
    
    # 创建数据模块
    dm = LFDataModule(
        data_dir="data",
        train_data_dir="data/train_data",
        test_data_dir="data/test_data",
        model=mode,
        batch_size=1,
        num_workers=4,
        n_rays=config.train.num_rays,
        val_chunk_size=config.eval.chunk 
    )
    last_ckpt = None
    if os.path.exists(checkpoint_dir):
        ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if ckpts:
            # 找到最新的 checkpoint
            last_ckpt = max(ckpts, key=os.path.getmtime)
            print(f"🔄 检测到断点文件: {last_ckpt}")
    if last_ckpt is None:
        # ==========================================
        # [新增] 阶段 1: 学习率搜索 (LR Finder)
        # ==========================================
        print(">>> 正在进行学习率搜索 (LR Finder)...")
        
        # 1. 创建一个临时的 Trainer (必须是单卡，不能用 DDP)
        # 如果你想指定只用第一张卡跑搜索，设 devices=[0]
        tuner_trainer = L.Trainer(
            accelerator="gpu",
            devices=[0],      # 搜索只需一张卡
            strategy="auto",  # 必须是 auto，不能是 ddp
            enable_checkpointing=False,
            logger=False,     # 关掉日志，保持清爽
            max_epochs=1      # 只是为了初始化，实际 Tuner 会控制步数
        )

        # 2. 初始化 Tuner
        from lightning.pytorch.tuner import Tuner
        tuner = Tuner(tuner_trainer)

        # 3. 开始搜索 (会自动更新 model.learning_rate)
        # min_lr 和 max_lr 可以根据你的经验调整范围
        lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-6, max_lr=1e-3, num_training=100)

        # 4. 获取结果并可视化
        new_lr = lr_finder.suggestion()
        print(f"✅ 建议的最佳学习率: {new_lr}")
        
        fig = lr_finder.plot(suggest=True)
        fig.savefig(os.path.join(result_save_dir, "lr_finder_plot.png"))
        print(f"📊 学习率曲线已保存至: {os.path.join(result_save_dir, 'lr_finder_plot.png')}")
    # 6. 清理显存 
        # 6. 清理显存 
        del tuner_trainer, tuner
        torch.cuda.empty_cache()
    else :
        print(">>> 跳过学习率搜索，直接进入正式训练...")
    # ==========================================
    # 阶段 2: 正式训练 (DDP)
    # ==========================================
    # 3. 创建正式的 Trainer 
    trainer = L.Trainer(
        max_steps=config.train.max_steps,
        accelerator="gpu",
        devices=1,  
        strategy="ddp",
        logger=TensorBoardLogger("logs", name=exp_name, version=mode), 
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="lfnr-{epoch:02d}",
                monitor="epoch",
                mode="max",
                save_last=True,
                save_top_k=4,
                every_n_epochs=1,
                save_on_train_epoch_end=True
            ),
            # 这里会记录实际使用的 LR，你可以看到它是否变成了新值
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        log_every_n_steps=50,
        check_val_every_n_epoch=10, 
    )
    # 开始训练
    trainer.fit(model, dm, ckpt_path=last_ckpt)
    
    # 测试
    print("=== 开始测试 ===")
    trainer.test(model, dm, ckpt_path=last_ckpt)
    
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"=== 完成 ===")
    print(f"🚀 运行总时长: {hours}小时 {minutes}分 {seconds}秒")