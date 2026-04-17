import torch
import torch.optim as optim
import os
import time
import datetime
from typing import Tuple, Dict, Any
from tqdm import tqdm
import numpy as np

from config import config
from nets.resunet_pro import create_model
# from nets.unet import create_model

from data_process import create_data_loaders
from utils import Metrics, Visualizer, EarlyStopping, create_loss_function, AdaptiveWeightedLoss
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


class Trainer:
    """训练器类 - 彻底优化版本，解决NCCL超时问题"""

    def __init__(self, local_rank=-1, world_size=1):
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        # 设置设备
        if self.is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = config.DEVICE

        # 设置分布式训练环境变量 - 关键优化
        if self.is_distributed:
            os.environ['NCCL_BLOCKING_WAIT'] = '0'
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
            os.environ['NCCL_TIMEOUT'] = '1800'  # 增加超时时间到30分钟

        # 创建模型
        self.model = create_model("resnet50_unet_seg")
        # self.model = create_model("unet", num_channels=1, num_classes=1)
        self.model = self.model.to(self.device)

        # 多卡并行训练 - 优化配置
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,  # 提高性能
                gradient_as_bucket_view=True  # 减少内存使用
            )
            print(f"初始化分布式训练，rank: {local_rank}, world_size: {world_size}")

        # 初始化训练状态
        self.start_epoch = 0
        self.best_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []

        # 新增：损失权重历史记录
        self.dice_weights = []
        self.bce_weights = []

        # 分别优化编码器和解码器
        encoder_params = []
        decoder_params = []

        # 处理多卡并行时的参数名称
        model_for_params = self.model.module if self.is_distributed else self.model

        for name, param in model_for_params.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        self.optimizer = self._create_optimizer()

        self.scheduler = self._create_scheduler()
      
        # 使用新的损失函数创建方式
        # 在训练脚本中这样使用
        if config.LOSS_FUNCTION == "focal_tversky":
             self.criterion = create_loss_function(
                loss_type="focal_tversky",
                smooth=config.FOCAL_TVERSKY_SMOOTH,
                focal_tversky_alpha=config.FOCAL_TVERSKY_ALPHA,
                focal_tversky_beta=config.FOCAL_TVERSKY_BETA,
                focal_tversky_gamma=config.FOCAL_TVERSKY_GAMMA
            )
        else:
             self.criterion = create_loss_function(
              loss_type=config.LOSS_FUNCTION,
              lesion_weight=config.WCE_LESION_WEIGHT,
              background_weight=config.WCE_BACKGROUND_WEIGHT,
              adaptive=config.ADAPTIVE_WEIGHT,
              pos_weight=config.BCE_POS_WEIGHT,
              smooth=config.DICE_SMOOTH
        ).to(self.device)

        # 梯度累积
        self.accumulation_steps = config.ACCUMULATION_STEPS if config.USE_GRADIENT_ACCUMULATION else 1

        self.early_stopping = EarlyStopping(patience=config.PATIENCE)

        # 创建数据加载器
        try:
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders()
        except Exception as e:
            print(f"数据加载器创建失败: {e}")
            if self.is_distributed:
                cleanup_distributed()
            raise

        self.visualizer = Visualizer()

        # 创建模型保存目录和日志目录（只在主进程创建）
        if local_rank <= 0:
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            os.makedirs(config.LOG_DIR, exist_ok=True)

            # 创建日志文件
            self.log_file = os.path.join(config.LOG_DIR,
                                         f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            self._write_log("训练开始", include_time=True)
            self._write_log(f"设备: {self.device}")
            self._write_log(f"世界大小: {world_size}")
            self._write_log(f"是否分布式训练: {self.is_distributed}")
            self._write_log(f"损失函数: {config.LOSS_FUNCTION}")

        # 断点续训
        if config.RESUME_TRAINING:
            self._resume_from_checkpoint()

    def calculate_lesion_specific_dice(self, pred, target):
        """计算病灶特定的Dice系数 - 使用新的Metrics类"""
        return Metrics.calculate_dice(pred, target)

    def calculate_lesion_specific_iou(self, pred, target):
        """计算病灶特定的IoU系数 - 使用新的Metrics类"""
        return Metrics.calculate_iou(pred, target)

    def calculate_all_metrics(self, pred, target):
        """计算所有指标"""
        return Metrics.calculate_all_metrics(pred, target)

    def analyze_class_imbalance(self, masks):
        """分析类别不平衡情况"""
        masks_np = masks.cpu().numpy()
        total_pixels = masks_np.size
        lesion_pixels = np.sum(masks_np > 0.5)
        background_pixels = total_pixels - lesion_pixels

        lesion_ratio = lesion_pixels / total_pixels if total_pixels > 0 else 0
        background_ratio = background_pixels / total_pixels if total_pixels > 0 else 0

        return lesion_ratio, background_ratio

    def _log_train_sample_info(self, image: torch.Tensor, pred: torch.Tensor,
                               target: torch.Tensor, epoch: int, batch_idx: int, sample_idx: int):
        """记录训练样本的详细信息"""
        # 计算所有指标
        metrics = self.calculate_all_metrics(pred, target)

        # 统计信息
        image_stats = f"图像范围: [{image.min():.3f}, {image.max():.3f}]"
        target_stats = f"真实掩码: {(target > 0.5).sum().item():.0f} 像素"
        pred_stats = f"预测掩码: {(torch.sigmoid(pred) > 0.5).sum().item():.0f} 像素"

        log_message = (
            f"训练样本可视化 - Epoch {epoch}, Batch {batch_idx}, Sample {sample_idx}:\n"
            f"  {image_stats}\n"
            f"  {target_stats}\n"
            f"  {pred_stats}\n"
            f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}\n"
            f"  精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, 准确率: {metrics['accuracy']:.4f}"
        )

        self._write_log(log_message)

    def _create_optimizer(self):
        """创建优化器"""
        model_for_params = self.model.module if self.is_distributed else self.model

        if config.OPTIMIZER_TYPE == "AdamW":
            return optim.AdamW(
                model_for_params.parameters(),
                lr=config.INITIAL_LR,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER_TYPE == "Adam":
            return optim.Adam(
                model_for_params.parameters(),
                lr=config.INITIAL_LR,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER_TYPE == "SGD":
            return optim.SGD(
                model_for_params.parameters(),
                lr=config.INITIAL_LR,
                momentum=0.9,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"不支持的优化器类型: {config.OPTIMIZER_TYPE}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        if config.LR_SCHEDULER == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=config.EPOCHS - config.LR_WARMUP_EPOCHS,
                eta_min=config.MIN_LR
            )
        elif config.LR_SCHEDULER == "step":
            return StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        elif config.LR_SCHEDULER == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=config.MIN_LR
            )
        else:
            # 默认使用余弦退火
            return CosineAnnealingLR(
                self.optimizer,
                T_max=config.EPOCHS,
                eta_min=config.MIN_LR
            )

    def _warmup_lr(self, epoch, batch_idx, num_batches):
        """简单的学习率预热"""
        if epoch < config.LR_WARMUP_EPOCHS:
            # 线性预热
            warmup_ratio = (batch_idx + epoch * num_batches) / (config.LR_WARMUP_EPOCHS * num_batches)
            warmup_lr = config.INITIAL_LR * warmup_ratio

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr

    def _write_log(self, message, include_time=False):
        """写入日志到文件和控制台"""
        if self.local_rank <= 0:
            if include_time:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_message = f"[{timestamp}] {message}"
            else:
                log_message = message

            # 写入文件
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')

            # 输出到控制台
            print(log_message)

    def _resume_from_checkpoint(self):
        """从检查点恢复训练状态"""
        try:
            if self.local_rank <= 0:
                self._write_log(f"正在从检查点恢复训练: {config.RESUME_CHECKPOINT_PATH}")

            checkpoint = torch.load(config.RESUME_CHECKPOINT_PATH, map_location=self.device)

            # 加载模型状态（处理多卡并行）
            if self.is_distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载学习率调度器状态
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # 加载早停状态
            # if 'early_stopping_state' in checkpoint:
            # self.early_stopping.load_state_dict(checkpoint['early_stopping_state'])

            # 恢复训练状态
            self.start_epoch = checkpoint['epoch']
            self.best_dice = checkpoint.get('best_dice', 0.0)

            # 恢复训练历史
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_dices = checkpoint.get('train_dices', [])
            self.val_dices = checkpoint.get('val_dices', [])

            # 恢复损失权重历史
            self.dice_weights = checkpoint.get('dice_weights', [])
            self.bce_weights = checkpoint.get('bce_weights', [])

            if self.local_rank <= 0:
                self._write_log(f"成功恢复训练状态，从第 {self.start_epoch} 轮继续训练")
                self._write_log(f"当前最佳 Dice: {self.best_dice:.4f}")

        except Exception as e:
            if self.local_rank <= 0:
                self._write_log(f"恢复训练失败: {e}")
            raise

    def train_epoch(self, epoch: int) -> Tuple[float, float, Dict[str, Any]]:
        """训练epoch - 优化版本"""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_lesion_dice = 0.0
        num_batches = len(self.train_loader)

        # 损失权重统计
        epoch_dice_weight = 0.0
        epoch_bce_weight = 0.0

        # 在主进程显示进度条
        if self.local_rank <= 0:
            pbar = tqdm(total=num_batches, desc=f'训练 Epoch {epoch}', unit='batch')
        else:
            pbar = None

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            try:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # 分析类别不平衡
                lesion_ratio, bg_ratio = self.analyze_class_imbalance(masks)

                # 前向传播
                self.optimizer.zero_grad(set_to_none=True)
                seg_output = self.model(images)

                # 使用损失函数
                if isinstance(self.criterion, AdaptiveWeightedLoss):
                    loss, loss_info = self.criterion(seg_output, masks)
                    # 记录权重
                    epoch_dice_weight += loss_info['dice_weight']
                    epoch_bce_weight += loss_info['bce_weight']
                else:
                    loss = self.criterion(seg_output, masks)
                    loss_info = {'total_loss': loss.item()}

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 计算指标
                dice = self.calculate_lesion_specific_dice(seg_output, masks)
                lesion_dice = dice  # 现在使用相同的Dice计算

                total_loss += loss_info['total_loss']
                total_dice += dice
                total_lesion_dice += lesion_dice

                if self.local_rank <= 0 and pbar is not None:
                    pbar.update(1)
                    postfix_dict = {
                        'Loss': f'{loss_info["total_loss"]:.4f}',
                        'Dice': f'{dice:.4f}',
                        'LesionRatio': f'{lesion_ratio:.4f}'
                    }

                    # 如果是自适应权重损失，显示权重信息
                    if isinstance(self.criterion, AdaptiveWeightedLoss):
                        postfix_dict['DiceW'] = f'{loss_info["dice_weight"]:.3f}'
                        postfix_dict['BCEW'] = f'{loss_info["bce_weight"]:.3f}'

                    pbar.set_postfix(postfix_dict)

                # 定期清理GPU缓存
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "NCCL" in str(e):
                    self._write_log(f"Rank {self.local_rank}: NCCL错误，跳过该批次: {e}")
                    continue
                else:
                    raise

        if self.local_rank <= 0 and pbar is not None:
            pbar.close()

        # 分布式训练时聚合所有进程的指标
        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_dice_tensor = torch.tensor(total_dice, device=self.device)
            num_batches_tensor = torch.tensor(num_batches, device=self.device)

            total_loss = self._reduce_tensor(total_loss_tensor).item()
            total_dice = self._reduce_tensor(total_dice_tensor).item()
            num_batches = self._reduce_tensor(num_batches_tensor).item()

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        # 计算平均权重
        weight_info = {}
        if isinstance(self.criterion, AdaptiveWeightedLoss):
            avg_dice_weight = epoch_dice_weight / num_batches
            avg_bce_weight = epoch_bce_weight / num_batches
            weight_info = {
                'dice_weight': avg_dice_weight,
                'bce_weight': avg_bce_weight
            }

        return avg_loss, avg_dice, weight_info

    def _reduce_tensor(self, tensor):
        """聚合所有进程的tensor - 优化版本"""
        if self.is_distributed:
            try:
                reduced_tensor = tensor.clone()
                dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
                return reduced_tensor / self.world_size
            except Exception as e:
                self._write_log(f"分布式聚合失败: {e}")
                return tensor
        return tensor

    def calculate_batch_dice(self, pred, target):
        """改进的Dice计算，处理全零预测 - 使用新的Metrics类"""
        return Metrics.calculate_dice(pred, target)

    def validate_epoch(self, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """验证一个epoch - 优化版本"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)

        # 存储所有指标
        all_metrics = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0
        }

        # 在主进程显示进度条
        if self.local_rank <= 0:
            pbar = tqdm(total=num_batches, desc=f'验证 Epoch {epoch}', unit='batch')
        else:
            pbar = None

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                try:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)

                    if batch_idx == 0 and self.local_rank <= 0:
                        self._analyze_validation_data(images, masks, epoch)

                    # 前向传播
                    seg_output = self.model(images)

                    # 计算损失
                    if isinstance(self.criterion, AdaptiveWeightedLoss):
                        loss, _ = self.criterion(seg_output, masks)
                    else:
                        loss = self.criterion(seg_output, masks)

                    # 计算所有指标
                    batch_metrics = self.calculate_all_metrics(seg_output, masks)

                    total_loss += loss.item()
                    total_dice += batch_metrics['dice']

                    # 累加所有指标
                    for key in all_metrics.keys():
                        all_metrics[key] += batch_metrics[key]

                    # 在主进程更新进度条
                    if self.local_rank <= 0 and pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Dice': f'{batch_metrics["dice"]:.4f}',
                            'IoU': f'{batch_metrics["iou"]:.4f}'
                        })

                except RuntimeError as e:
                    if "NCCL" in str(e):
                        self._write_log(f"Rank {self.local_rank}: NCCL错误，跳过验证批次: {e}")
                        continue
                    else:
                        raise

        if self.local_rank <= 0 and pbar is not None:
            pbar.close()

        # 分布式训练时聚合所有进程的指标
        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_dice_tensor = torch.tensor(total_dice, device=self.device)
            num_batches_tensor = torch.tensor(num_batches, device=self.device)

            # 聚合所有指标
            metrics_tensors = {}
            for key in all_metrics.keys():
                metrics_tensors[key] = torch.tensor(all_metrics[key], device=self.device)

            total_loss = self._reduce_tensor(total_loss_tensor).item()
            total_dice = self._reduce_tensor(total_dice_tensor).item()
            num_batches = self._reduce_tensor(num_batches_tensor).item()

            for key in all_metrics.keys():
                all_metrics[key] = self._reduce_tensor(metrics_tensors[key]).item()

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        # 计算平均指标
        for key in all_metrics.keys():
            all_metrics[key] /= num_batches

        return avg_loss, avg_dice, all_metrics

    def _analyze_validation_data(self, images, masks, epoch):
        """分析验证集数据"""
        masks_np = masks.cpu().numpy()

        # 统计肿瘤像素比例
        total_pixels = masks_np.size
        tumor_pixels = np.sum(masks_np > 0)
        tumor_ratio = tumor_pixels / total_pixels

        # 统计每个batch中有肿瘤的样本数量
        batch_size = masks_np.shape[0]
        has_tumor_count = np.sum([np.any(mask > 0) for mask in masks_np])

        print(f"验证集分析 - Epoch {epoch}:")
        print(f"  批次大小: {batch_size}")
        print(f"  有肿瘤的样本: {has_tumor_count}/{batch_size}")
        print(f"  肿瘤像素比例: {tumor_ratio:.6f}")
        print(f"  掩码数值范围: [{np.min(masks_np)}, {np.max(masks_np)}]")

    def save_model(self, epoch: int, dice: float, is_best: bool = False, weight_info: Dict = None):
        """保存模型，包含完整的训练状态"""
        # 只在主进程保存模型
        if self.local_rank > 0:
            return

        # 获取模型状态字典（处理多卡并行）
        if self.is_distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'early_stopping_state': self.early_stopping.state_dict(),
            'dice': dice,
            'best_dice': self.best_dice,
            'config': config.__dict__,
            # 保存训练历史
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            # 保存损失权重历史
            'dice_weights': self.dice_weights,
            'bce_weights': self.bce_weights,
        }

        # 保存常规检查点
        checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self._write_log(f"保存最佳模型，Dice: {dice:.4f}")

    def train(self):
        """训练主循环 - 优化版本"""
        if self.local_rank <= 0:
            self._write_log("开始训练...")
            config.print_config()

        # 验证断点续训配置
        if config.RESUME_TRAINING and self.local_rank <= 0:
            config.validate_resume_config()

        # 总体训练进度条
        if self.local_rank <= 0:
            epoch_pbar = tqdm(total=config.EPOCHS, desc="总体训练进度", unit="epoch")
        else:
            epoch_pbar = None

        try:
            for epoch in range(self.start_epoch, config.EPOCHS):
                if self.local_rank <= 0:
                    self._write_log(f"\n{'=' * 50}")
                    self._write_log(f"Epoch {epoch + 1}/{config.EPOCHS}")
                    self._write_log(f"{'=' * 50}")

                # 训练
                train_loss, train_dice, weight_info = self.train_epoch(epoch + 1)
                self.train_losses.append(train_loss)
                self.train_dices.append(train_dice)

                # 记录损失权重
                if weight_info:
                    self.dice_weights.append(weight_info.get('dice_weight', 0.0))
                    self.bce_weights.append(weight_info.get('bce_weight', 0.0))

                # 验证
                val_loss, val_dice, val_metrics = self.validate_epoch(epoch + 1)
                self.val_losses.append(val_loss)
                self.val_dices.append(val_dice)

                # 学习率调整
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                # 早停检查
                self.early_stopping(val_loss)

                # 保存最佳模型
                is_best = val_dice > self.best_dice
                if is_best:
                    self.best_dice = val_dice

                # 保存检查点（只在主进程）
                if (epoch + 1) % config.SAVE_INTERVAL == 0 or is_best:
                    self.save_model(epoch + 1, val_dice, is_best, weight_info)

                # 绘制训练曲线（只在主进程）
                if self.local_rank <= 0:
                    self.visualizer.plot_training_curve(
                        self.train_losses, self.val_losses,
                        self.train_dices, self.val_dices,
                        epoch + 1
                    )

                    # 更新总体进度条
                    if epoch_pbar is not None:
                        epoch_pbar.update(1)
                        postfix_dict = {
                            'Train_Loss': f'{train_loss:.4f}',
                            'Val_Dice': f'{val_dice:.4f}',
                            'Best_Dice': f'{self.best_dice:.4f}'
                        }

                        # 显示损失权重信息
                        if weight_info:
                            postfix_dict['DiceW'] = f'{weight_info.get("dice_weight", 0.0):.3f}'
                            postfix_dict['BCEW'] = f'{weight_info.get("bce_weight", 0.0):.3f}'

                        epoch_pbar.set_postfix(postfix_dict)

                    # 打印epoch总结并保存到日志
                    epoch_summary = f"\nEpoch {epoch + 1} 总结:\n"
                    epoch_summary += f"  训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}\n"
                    epoch_summary += f"  训练Dice: {train_dice:.4f}, 验证Dice: {val_dice:.4f}\n"
                    epoch_summary += f"  验证IoU: {val_metrics['iou']:.4f}, 精确率: {val_metrics['precision']:.4f}\n"
                    epoch_summary += f"  验证召回率: {val_metrics['recall']:.4f}, 准确率: {val_metrics['accuracy']:.4f}\n"
                    epoch_summary += f"  最佳Dice: {self.best_dice:.4f}\n"
                    epoch_summary += f"  学习率: {self.optimizer.param_groups[0]['lr']:.2e}"

                    # 添加损失权重信息
                    if weight_info:
                        epoch_summary += f"\n  损失权重 - Dice: {weight_info.get('dice_weight', 0.0):.3f}, BCE: {weight_info.get('bce_weight', 0.0):.3f}"

                    self._write_log(epoch_summary)

                # 早停检查
                if self.early_stopping.early_stop:
                    if self.local_rank <= 0:
                        self._write_log("早停触发，停止训练")
                    break

                # 定期清理内存
                if epoch % 5 == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            error_msg = f"训练过程中发生错误: {e}"
            self._write_log(error_msg)
            if self.is_distributed:
                cleanup_distributed()
            raise

        finally:
            if self.local_rank <= 0 and epoch_pbar is not None:
                epoch_pbar.close()

            if self.local_rank <= 0:
                self._write_log("训练完成!")
                self._write_log(f"最佳验证Dice: {self.best_dice:.4f}")


def setup_distributed(rank, world_size):
    """初始化分布式训练环境 - 优化版本"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_BLOCKING_WAIT'] = '0'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时

    # 初始化进程组 - 使用datetime.timedelta而不是torch.timedelta
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30)  # 修正这里
    )

    # 设置设备
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_distributed(rank, world_size):
    """分布式训练函数 - 优化版本"""
    try:
        setup_distributed(rank, world_size)

        # 创建训练器
        trainer = Trainer(local_rank=rank, world_size=world_size)

        # 开始训练
        trainer.train()

    except Exception as e:
        print(f"Rank {rank} 训练失败: {e}")
        raise
    finally:
        cleanup_distributed()


def main():
    """训练主函数"""
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，使用CPU训练")
        trainer = Trainer()
        trainer.train()
        return

    # 获取GPU数量
    world_size = torch.cuda.device_count()

    if world_size > 1:
        print(f"检测到 {world_size} 个GPU，启动分布式训练")

        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)

        # 启动多进程训练
        mp.spawn(
            train_distributed,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        print("使用单GPU训练")
        trainer = Trainer()
        trainer.train()


if __name__ == "__main__":
    main()
