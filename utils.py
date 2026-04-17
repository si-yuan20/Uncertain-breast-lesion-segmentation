import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

# 必须在导入pyplot之前设置后端
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import sklearn.metrics as metrics


class Metrics:
    """评估指标计算器 - 优化版本，只关注前景（肿块区域）"""

    @staticmethod
    def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """计算Dice系数 - 只关注前景（肿块区域）"""
        # 确保张量维度正确
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        if pred.dim() != 4:
            raise ValueError(f"预测张量应该是3D或4D的，但得到的是{pred.dim()}D")
        if target.dim() != 4:
            raise ValueError(f"目标张量应该是3D或4D的，但得到的是{target.dim()}D")

        # 调整预测张量尺寸以匹配目标张量
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = (target > 0.5).float()

        # 只在前景区域计算Dice
        batch_dice = 0.0
        batch_size = pred_binary.size(0)
        valid_samples = 0

        for i in range(batch_size):
            pred_i = pred_binary[i]
            target_i = target_binary[i]

            # 只计算有病灶的样本
            if target_i.sum() > 0:
                intersection = (pred_i * target_i).sum()
                union = pred_i.sum() + target_i.sum()

                if union > 0:
                    dice = (2. * intersection) / (union + 1e-6)
                    if torch.is_tensor(dice):
                        dice = dice.item()
                    batch_dice += dice
                    valid_samples += 1

        # 如果没有有效样本，返回0
        return batch_dice / valid_samples if valid_samples > 0 else 0.0

    @staticmethod
    def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """计算IoU - 只关注前景（肿块区域）"""
        # 确保张量维度正确
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        if pred.dim() != 4:
            raise ValueError(f"预测张量应该是3D或4D的，但得到的是{pred.dim()}D")
        if target.dim() != 4:
            raise ValueError(f"目标张量应该是3D或4D的，但得到的是{target.dim()}D")

        # 调整预测张量尺寸以匹配目标张量
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = (target > 0.5).float()

        # 只在前景区域计算IoU
        batch_iou = 0.0
        batch_size = pred_binary.size(0)
        valid_samples = 0

        for i in range(batch_size):
            pred_i = pred_binary[i]
            target_i = target_binary[i]

            # 只计算有病灶的样本
            if target_i.sum() > 0:
                intersection = (pred_i * target_i).sum()
                union = pred_i.sum() + target_i.sum() - intersection

                if union > 0:
                    iou = intersection / (union + 1e-6)
                    if torch.is_tensor(iou):
                        iou = iou.item()
                    batch_iou += iou
                    valid_samples += 1

        # 如果没有有效样本，返回0
        return batch_iou / valid_samples if valid_samples > 0 else 0.0

    @staticmethod
    def calculate_precision_recall_accuracy(pred: torch.Tensor, target: torch.Tensor,
                                            threshold: float = 0.5) -> Tuple[float, float, float]:
        """计算精确率、召回率和准确率 - 只关注前景（肿块区域）"""
        # 确保张量维度正确
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        if pred.dim() != 4:
            raise ValueError(f"预测张量应该是3D或4D的，但得到的是{pred.dim()}D")
        if target.dim() != 4:
            raise ValueError(f"目标张量应该是3D或4D的，但得到的是{target.dim()}D")

        # 调整预测张量尺寸以匹配目标张量
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

        pred_binary = (torch.sigmoid(pred) > threshold).cpu().numpy()
        target_binary = (target > 0.5).cpu().numpy()

        # 只在前景区域计算指标
        all_precision = []
        all_recall = []
        all_accuracy = []
        all_specificity = []

        for i in range(pred_binary.shape[0]):
            pred_flat = pred_binary[i].reshape(-1)
            target_flat = target_binary[i].reshape(-1)

            # 只计算有病灶的样本
            if np.sum(target_flat) > 0:
                # 精确率、召回率、准确率
                precision = metrics.precision_score(target_flat, pred_flat, zero_division=0)
                recall = metrics.recall_score(target_flat, pred_flat, zero_division=0)
                accuracy = metrics.accuracy_score(target_flat, pred_flat)

                specificity = metrics.recall_score(
                    1 - target_flat, 1 - pred_flat, zero_division=0
                )

                all_precision.append(precision)
                all_recall.append(recall)
                all_accuracy.append(accuracy)
                all_specificity.append(specificity)

        # 计算平均值
        avg_precision = np.mean(all_precision) if all_precision else 0.0
        avg_recall = np.mean(all_recall) if all_recall else 0.0
        avg_accuracy = np.mean(all_accuracy) if all_accuracy else 0.0
        avg_specificity = np.mean(all_specificity) if all_specificity else 0.0

        return avg_precision, avg_recall, avg_accuracy, avg_specificity

    @staticmethod
    def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor,
                              threshold: float = 0.5) -> Dict[str, float]:
        """计算所有指标"""
        dice = Metrics.calculate_dice(pred, target, threshold)
        iou = Metrics.calculate_iou(pred, target, threshold)
        #
        precision, recall, accuracy, specificity = Metrics.calculate_precision_recall_accuracy(pred, target, threshold)

        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'specificity': specificity,  #
            'f1_score': 2 * precision * recall / (precision + recall + 1e-6)  #
        }


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,1,H,W) ; target: (B,1,H,W) or (B,H,W)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        prob = torch.sigmoid(logits)
        prob = prob.view(prob.size(0), -1)
        tgt = target.view(target.size(0), -1)

        intersection = (prob * tgt).sum(dim=1)
        union = prob.sum(dim=1) + tgt.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELossV2(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6, pos_weight: float | None = None):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.dw = float(dice_weight)
        self.bw = float(bce_weight)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        dice_loss = self.dice(logits, target)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=self.pos_weight if self.pos_weight is not None else None
        )
        return self.dw * dice_loss + self.bw * bce_loss




class AdaptiveWeightedLoss(nn.Module):
    """
    AdaptiveWeightedLoss：自适应平衡 DiceLoss 与 BCEWithLogitsLoss 的权重。

    设计目标（实用版，稳定优先）：
    - 同时计算 dice_loss 与 bce_loss
    - 维护二者的 EMA（指数滑动平均）用于平滑
    - 按“当前更难(更大loss)分量给予更大权重”的策略自适应分配权重
    - 权重会被 clamp 到 [min_w, max_w]，避免某一项被压到接近0导致训练不稳

    forward 返回：
      total_loss, info_dict
    其中 info_dict 包含 dice_loss/bce_loss 以及 dice_weight/bce_weight，便于在训练脚本里记录曲线
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        pos_weight: float | None = None,
        ema_decay: float = 0.98,
        min_w: float = 0.05,
        max_w: float = 0.95,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.ema_decay = float(ema_decay)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.eps = float(eps)

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

        # EMA buffers（用 buffer 便于 DDP/ckpt 一致性；不参与梯度）
        self.register_buffer("ema_dice", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("ema_bce", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _update_ema(self, dice_loss_val: float, bce_loss_val: float):
        d = self.ema_decay
        if int(self.step.item()) == 0:
            # 第一次用当前值初始化，避免起步偏置
            self.ema_dice.fill_(float(dice_loss_val))
            self.ema_bce.fill_(float(bce_loss_val))
        else:
            self.ema_dice.mul_(d).add_((1.0 - d) * float(dice_loss_val))
            self.ema_bce.mul_(d).add_((1.0 - d) * float(bce_loss_val))
        self.step.add_(1)

    @staticmethod
    def _clamp_weight(w: float, min_w: float, max_w: float) -> float:
        if w < min_w:
            return min_w
        if w > max_w:
            return max_w
        return w

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        dice_loss = self.dice(logits, target)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=self.pos_weight if self.pos_weight is not None else None
        )

        # 更新EMA（不回传梯度）
        self._update_ema(float(dice_loss.detach().cpu()), float(bce_loss.detach().cpu()))

        # 自适应权重：谁更难（EMA更大）谁权重更大
        ema_d = float(self.ema_dice.detach().cpu())
        ema_b = float(self.ema_bce.detach().cpu())
        denom = (ema_d + ema_b + self.eps)
        dice_w = ema_d / denom
        bce_w = ema_b / denom

        dice_w = self._clamp_weight(dice_w, self.min_w, self.max_w)
        bce_w = self._clamp_weight(bce_w, self.min_w, self.max_w)

        # 归一化（clamp后再归一化一次，保证和为1）
        s = dice_w + bce_w + self.eps
        dice_w = dice_w / s
        bce_w = bce_w / s

        total = dice_w * dice_loss + bce_w * bce_loss

        info = {
            "dice_loss": float(dice_loss.detach().cpu()),
            "bce_loss": float(bce_loss.detach().cpu()),
            "dice_weight": float(dice_w),
            "bce_weight": float(bce_w),
            "ema_dice": float(ema_d),
            "ema_bce": float(ema_b),
            "total_loss": float(total.detach().cpu()),
        }
        return total, info


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (binary)
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        prob = torch.sigmoid(logits)
        prob = prob.view(prob.size(0), -1)
        tgt = target.view(target.size(0), -1)

        tp = (prob * tgt).sum(dim=1)
        fp = (prob * (1 - tgt)).sum(dim=1)
        fn = ((1 - prob) * tgt).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = torch.pow((1.0 - tversky), self.gamma)
        return loss.mean()


class BoundaryLossSobel(nn.Module):
    """
    边界损失：用 Sobel 梯度近似边缘（无需距离变换，易复现）
    思路：对 target 和 sigmoid(pred) 做梯度幅值，计算 L1
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        prob = torch.sigmoid(logits)
        g_pred = self._grad_mag(prob)
        g_tgt = self._grad_mag(target)

        return F.l1_loss(g_pred, g_tgt)


class CombinedSegLoss(nn.Module):
    """
    将 Dice-CE 作为主干，再按需叠加 focal / boundary / focal_tversky
    返回：loss, loss_dict（用于日志）
    """
    def __init__(
        self,
        base_dice_ce: DiceBCELossV2,
        add_focal: bool = False,
        add_boundary: bool = False,
        add_focal_tversky: bool = False,
        focal_alpha: float = 0.8,
        focal_gamma: float = 2.0,
        ft_alpha: float = 0.7,
        ft_beta: float = 0.3,
        ft_gamma: float = 0.75,
        w_base: float = 1.0,
        w_focal: float = 1.0,
        w_boundary: float = 1.0,
        w_focal_tversky: float = 1.0,
    ):
        super().__init__()
        self.base = base_dice_ce
        self.add_focal = add_focal
        self.add_boundary = add_boundary
        self.add_focal_tversky = add_focal_tversky

        self.focal = FocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma) if add_focal else None
        self.boundary = BoundaryLossSobel() if add_boundary else None
        self.ft = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma) if add_focal_tversky else None

        self.w_base = float(w_base)
        self.w_focal = float(w_focal)
        self.w_boundary = float(w_boundary)
        self.w_ft = float(w_focal_tversky)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        loss_dict = {}

        base = self.base(logits, target)
        total = self.w_base * base
        loss_dict["base_dice_ce"] = float(base.detach().cpu())

        if self.add_focal:
            lf = self.focal(logits, target)
            total = total + self.w_focal * lf
            loss_dict["focal"] = float(lf.detach().cpu())

        if self.add_boundary:
            lb = self.boundary(logits, target)
            total = total + self.w_boundary * lb
            loss_dict["boundary"] = float(lb.detach().cpu())

        if self.add_focal_tversky:
            lft = self.ft(logits, target)
            total = total + self.w_ft * lft
            loss_dict["focal_tversky"] = float(lft.detach().cpu())

        loss_dict["total_loss"] = float(total.detach().cpu())
        return total, loss_dict


def create_loss_function(
    loss_ablation: str = "dice_ce",
    smooth: float = 1e-6,
    pos_weight: float | None = None,
    # base weights
    dice_weight: float = 0.7,
    bce_weight: float = 0.3,
    # focal params
    focal_alpha: float = 0.8,
    focal_gamma: float = 2.0,
    # focal tversky params
    ft_alpha: float = 0.7,
    ft_beta: float = 0.3,
    ft_gamma: float = 0.75,
    # component weights (可做附加消融)
    w_base: float = 1.0,
    w_focal: float = 1.0,
    w_boundary: float = 1.0,
    w_focal_tversky: float = 1.0,
):
    """
    统一loss工厂：直接对应你论文(1)消融组别
    返回一个 nn.Module，其 forward:
      - 输出 (loss_tensor, loss_dict)
    """
    base = DiceBCELossV2(
        dice_weight=dice_weight,
        bce_weight=bce_weight,
        smooth=smooth,
        pos_weight=pos_weight
    )

    loss_ablation = (loss_ablation or "dice_ce").lower()


    # ✅ Adaptive weighted (Dice + BCE) —— 动态调整权重
    if loss_ablation in {"adaptive_weighted", "dice_ce_adaptive", "dice_ce_awl"}:
        return AdaptiveWeightedLoss(
            smooth=smooth,
            pos_weight=pos_weight,
            ema_decay=0.98,
            min_w=0.05,
            max_w=0.95,
        )
    if loss_ablation == "dice_ce":
        return CombinedSegLoss(base, add_focal=False, add_boundary=False, add_focal_tversky=False,
                               focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                               ft_alpha=ft_alpha, ft_beta=ft_beta, ft_gamma=ft_gamma,
                               w_base=w_base, w_focal=w_focal, w_boundary=w_boundary, w_focal_tversky=w_focal_tversky)

    if loss_ablation == "dice_ce_focal":
        return CombinedSegLoss(base, add_focal=True, add_boundary=False, add_focal_tversky=False,
                               focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                               ft_alpha=ft_alpha, ft_beta=ft_beta, ft_gamma=ft_gamma,
                               w_base=w_base, w_focal=w_focal, w_boundary=w_boundary, w_focal_tversky=w_focal_tversky)

    if loss_ablation == "dice_ce_boundary":
        return CombinedSegLoss(base, add_focal=False, add_boundary=True, add_focal_tversky=False,
                               focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                               ft_alpha=ft_alpha, ft_beta=ft_beta, ft_gamma=ft_gamma,
                               w_base=w_base, w_focal=w_focal, w_boundary=w_boundary, w_focal_tversky=w_focal_tversky)

    if loss_ablation == "dice_ce_focal_tversky":
        return CombinedSegLoss(base, add_focal=False, add_boundary=False, add_focal_tversky=True,
                               focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                               ft_alpha=ft_alpha, ft_beta=ft_beta, ft_gamma=ft_gamma,
                               w_base=w_base, w_focal=w_focal, w_boundary=w_boundary, w_focal_tversky=w_focal_tversky)

    if loss_ablation == "dice_ce_focal_tversky_align":
        # align 在训练循环里加（因为要从 model.get_align_losses() 取）
        return CombinedSegLoss(base, add_focal=False, add_boundary=False, add_focal_tversky=True,
                               focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                               ft_alpha=ft_alpha, ft_beta=ft_beta, ft_gamma=ft_gamma,
                               w_base=w_base, w_focal=w_focal, w_boundary=w_boundary, w_focal_tversky=w_focal_tversky)

    raise ValueError(f"Unsupported loss_ablation={loss_ablation}")

class Visualizer:
    """可视化工具类 - 修复版本"""

    def __init__(self, save_dir: str = None):
        """
        初始化可视化工具

        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir or "./results"
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_training_curve(self,
                            train_losses: List[float],
                            val_losses: List[float],
                            train_dices: List[float],
                            val_dices: List[float],
                            current_epoch: int,
                            model_name: str = "Model",
                            show: bool = False):
        """
        绘制训练曲线并保存

        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_dices: 训练Dice分数列表
            val_dices: 验证Dice分数列表
            current_epoch: 当前epoch
            model_name: 模型名称
            show: 是否显示图表
        """
        epochs = list(range(1, len(train_losses) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'{model_name} - Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Dice分数曲线
        axes[0, 1].plot(epochs, train_dices, 'b-', label='Train Dice', linewidth=2)
        axes[0, 1].plot(epochs, val_dices, 'r-', label='Val Dice', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title(f'{model_name} - Dice Score Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # 损失差值
        if len(train_losses) > 1:
            train_loss_diff = np.abs(np.diff(train_losses))
            val_loss_diff = np.abs(np.diff(val_losses))
            axes[1, 0].plot(range(2, len(train_losses) + 1), train_loss_diff,
                            'b-', label='Train Loss Diff', linewidth=2)
            axes[1, 0].plot(range(2, len(train_losses) + 1), val_loss_diff,
                            'r-', label='Val Loss Diff', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Difference')
            axes[1, 0].set_title('Loss Convergence')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Dice分数差值
        if len(train_dices) > 1:
            train_dice_diff = np.abs(np.diff(train_dices))
            val_dice_diff = np.abs(np.diff(val_dices))
            axes[1, 1].plot(range(2, len(train_dices) + 1), train_dice_diff,
                            'b-', label='Train Dice Diff', linewidth=2)
            axes[1, 1].plot(range(2, len(train_dices) + 1), val_dice_diff,
                            'r-', label='Val Dice Diff', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Dice Difference')
            axes[1, 1].set_title('Dice Convergence')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.save_dir, f'training_curve_epoch{current_epoch}_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def save_training_curves(self,
                             train_losses: List[float],
                             val_losses: List[float],
                             train_dices: List[float],
                             val_dices: List[float],
                             train_ious: Optional[List[float]] = None,
                             val_ious: Optional[List[float]] = None,
                             save_dir: Optional[str] = None,
                             model_name: str = "Model"):
        """
        保存完整的训练曲线（新增方法）

        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_dices: 训练Dice分数列表
            val_dices: 验证Dice分数列表
            train_ious: 训练IoU分数列表（可选）
            val_ious: 验证IoU分数列表（可选）
            save_dir: 保存目录
            model_name: 模型名称
        """
        if save_dir:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

        epochs = list(range(1, len(train_losses) + 1))

        # 确定子图数量
        n_plots = 3 if train_ious is not None and val_ious is not None else 2
        fig_height = 5 * n_plots

        fig, axes = plt.subplots(n_plots, 2, figsize=(15, fig_height))

        # 如果只有一行，确保axes是2D数组
        if n_plots == 1:
            axes = axes.reshape(1, -1)

        row = 0

        # 1. 损失曲线
        axes[row, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        axes[row, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        axes[row, 0].set_xlabel('Epoch', fontsize=12)
        axes[row, 0].set_ylabel('Loss', fontsize=12)
        axes[row, 0].set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
        axes[row, 0].legend(fontsize=10)
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].tick_params(axis='both', labelsize=10)

        # 损失平滑曲线（移动平均）
        if len(train_losses) > 5:
            window_size = min(5, len(train_losses) // 4)
            train_smooth = self._moving_average(train_losses, window_size)
            val_smooth = self._moving_average(val_losses, window_size)
            axes[row, 1].plot(epochs[:len(train_smooth)], train_smooth, 'b--',
                              label=f'Train Loss (MA{window_size})', linewidth=2)
            axes[row, 1].plot(epochs[:len(val_smooth)], val_smooth, 'r--',
                              label=f'Val Loss (MA{window_size})', linewidth=2)
        axes[row, 1].set_xlabel('Epoch', fontsize=12)
        axes[row, 1].set_ylabel('Smoothed Loss', fontsize=12)
        axes[row, 1].set_title('Smoothed Loss Curve', fontsize=14, fontweight='bold')
        axes[row, 1].legend(fontsize=10)
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].tick_params(axis='both', labelsize=10)

        row += 1

        # 2. Dice分数曲线
        axes[row, 0].plot(epochs, train_dices, 'g-', label='Train Dice', linewidth=2, marker='o', markersize=4)
        axes[row, 0].plot(epochs, val_dices, 'm-', label='Val Dice', linewidth=2, marker='s', markersize=4)
        axes[row, 0].set_xlabel('Epoch', fontsize=12)
        axes[row, 0].set_ylabel('Dice Score', fontsize=12)
        axes[row, 0].set_title(f'{model_name} - Dice Score', fontsize=14, fontweight='bold')
        axes[row, 0].legend(fontsize=10)
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_ylim([0, 1])
        axes[row, 0].tick_params(axis='both', labelsize=10)

        # 最佳Dice标记
        best_dice_epoch = epochs[val_dices.index(max(val_dices))] if val_dices else 0
        best_dice_value = max(val_dices) if val_dices else 0
        axes[row, 0].axvline(x=best_dice_epoch, color='r', linestyle='--', alpha=0.5,
                             label=f'Best Dice: {best_dice_value:.4f} at Epoch {best_dice_epoch}')
        axes[row, 0].legend(fontsize=10)

        # Dice收敛分析
        if len(val_dices) > 1:
            dice_convergence = np.abs(np.diff(val_dices))
            axes[row, 1].plot(epochs[1:], dice_convergence, 'c-',
                              label='Dice Change', linewidth=2, marker='^', markersize=4)
            axes[row, 1].axhline(y=0.001, color='r', linestyle='--', alpha=0.5,
                                 label='Convergence Threshold')
            axes[row, 1].set_xlabel('Epoch', fontsize=12)
            axes[row, 1].set_ylabel('Dice Change', fontsize=12)
            axes[row, 1].set_title('Dice Convergence Analysis', fontsize=14, fontweight='bold')
            axes[row, 1].legend(fontsize=10)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].tick_params(axis='both', labelsize=10)

        row += 1

        # 3. IoU分数曲线（如果提供）
        if train_ious is not None and val_ious is not None and len(train_ious) > 0:
            axes[row, 0].plot(epochs[:len(train_ious)], train_ious, 'c-',
                              label='Train IoU', linewidth=2, marker='o', markersize=4)
            axes[row, 0].plot(epochs[:len(val_ious)], val_ious, 'y-',
                              label='Val IoU', linewidth=2, marker='s', markersize=4)
            axes[row, 0].set_xlabel('Epoch', fontsize=12)
            axes[row, 0].set_ylabel('IoU Score', fontsize=12)
            axes[row, 0].set_title(f'{model_name} - IoU Score', fontsize=14, fontweight='bold')
            axes[row, 0].legend(fontsize=10)
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 0].set_ylim([0, 1])
            axes[row, 0].tick_params(axis='both', labelsize=10)

            # 最佳IoU标记
            best_iou_epoch = epochs[val_ious.index(max(val_ious))] if val_ious else 0
            best_iou_value = max(val_ious) if val_ious else 0
            axes[row, 0].axvline(x=best_iou_epoch, color='r', linestyle='--', alpha=0.5,
                                 label=f'Best IoU: {best_iou_value:.4f} at Epoch {best_iou_epoch}')
            axes[row, 0].legend(fontsize=10)

            # IoU与Dice相关性
            if len(val_dices) == len(val_ious):
                axes[row, 1].scatter(val_dices, val_ious, c=epochs, cmap='viridis',
                                     s=50, alpha=0.7, edgecolors='k')
                axes[row, 1].set_xlabel('Dice Score', fontsize=12)
                axes[row, 1].set_ylabel('IoU Score', fontsize=12)
                axes[row, 1].set_title('Dice-IoU Correlation', fontsize=14, fontweight='bold')
                axes[row, 1].grid(True, alpha=0.3)
                axes[row, 1].tick_params(axis='both', labelsize=10)

                # 添加趋势线
                z = np.polyfit(val_dices, val_ious, 1)
                p = np.poly1d(z)
                axes[row, 1].plot(sorted(val_dices), p(sorted(val_dices)),
                                  "r--", alpha=0.8, label=f'Correlation: {np.corrcoef(val_dices, val_ious)[0, 1]:.3f}')
                axes[row, 1].legend(fontsize=10)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.save_dir, f'training_summary_{model_name}_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存数据为CSV文件
        self._save_training_data_to_csv(
            epochs, train_losses, val_losses, train_dices, val_dices,
            train_ious, val_ious, save_dir, model_name
        )

        return save_path

    def plot_loss_weights(self,
                          dice_weights: List[float],
                          bce_weights: List[float],
                          save_dir: Optional[str] = None,
                          model_name: str = "Model"):
        """
        绘制损失权重曲线

        Args:
            dice_weights: Dice损失权重列表
            bce_weights: BCE损失权重列表
            save_dir: 保存目录
            model_name: 模型名称
        """
        if save_dir:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)

        if not dice_weights or not bce_weights:
            return None

        epochs = list(range(1, len(dice_weights) + 1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 权重曲线
        axes[0].plot(epochs, dice_weights, 'b-', label='Dice Weight', linewidth=2, marker='o')
        axes[0].plot(epochs, bce_weights, 'r-', label='BCE Weight', linewidth=2, marker='s')
        axes[0].plot(epochs, np.array(dice_weights) + np.array(bce_weights),
                     'g--', label='Total Weight', linewidth=1, alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Weight')
        axes[0].set_title(f'{model_name} - Loss Weights')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 权重比例
        weight_ratio = np.array(dice_weights) / (np.array(bce_weights) + 1e-8)
        axes[1].plot(epochs, weight_ratio, 'm-', label='Dice/BCE Ratio', linewidth=2, marker='^')
        axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Weight')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Weight Ratio')
        axes[1].set_title('Loss Weight Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.save_dir, f'loss_weights_{model_name}_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def _moving_average(self, data: List[float], window_size: int) -> np.ndarray:
        """
        计算移动平均

        Args:
            data: 输入数据
            window_size: 窗口大小

        Returns:
            移动平均后的数据
        """
        if len(data) < window_size:
            return np.array(data)

        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')

    def _save_training_data_to_csv(self,
                                   epochs: List[int],
                                   train_losses: List[float],
                                   val_losses: List[float],
                                   train_dices: List[float],
                                   val_dices: List[float],
                                   train_ious: Optional[List[float]],
                                   val_ious: Optional[List[float]],
                                   save_dir: str,
                                   model_name: str):
        """
        保存训练数据为CSV文件

        Args:
            epochs: epoch列表
            train_losses: 训练损失
            val_losses: 验证损失
            train_dices: 训练Dice
            val_dices: 验证Dice
            train_ious: 训练IoU
            val_ious: 验证IoU
            save_dir: 保存目录
            model_name: 模型名称
        """
        try:
            import csv

            csv_path = os.path.join(save_dir, f'training_data_{model_name}.csv')

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 写入表头
                headers = ['Epoch', 'Train_Loss', 'Val_Loss', 'Train_Dice', 'Val_Dice']
                if train_ious is not None:
                    headers.extend(['Train_IoU', 'Val_IoU'])
                headers.extend(['Learning_Rate', 'Best_Dice_Epoch', 'Best_IoU_Epoch'])

                writer.writerow(headers)

                # 写入数据
                for i, epoch in enumerate(epochs):
                    row = [epoch]

                    # 添加损失和Dice
                    if i < len(train_losses):
                        row.append(f"{train_losses[i]:.6f}")
                    else:
                        row.append("")

                    if i < len(val_losses):
                        row.append(f"{val_losses[i]:.6f}")
                    else:
                        row.append("")

                    if i < len(train_dices):
                        row.append(f"{train_dices[i]:.6f}")
                    else:
                        row.append("")

                    if i < len(val_dices):
                        row.append(f"{val_dices[i]:.6f}")
                    else:
                        row.append("")

                    # 添加IoU（如果存在）
                    if train_ious is not None:
                        if i < len(train_ious):
                            row.append(f"{train_ious[i]:.6f}")
                        else:
                            row.append("")

                        if i < len(val_ious):
                            row.append(f"{val_ious[i]:.6f}")
                        else:
                            row.append("")

                    # 添加占位符
                    row.extend(["", "", ""])

                    writer.writerow(row)

                # 添加总结行
                writer.writerow([])
                summary_row = ['Summary',
                               f"Min Train Loss: {min(train_losses):.6f}" if train_losses else "",
                               f"Min Val Loss: {min(val_losses):.6f}" if val_losses else "",
                               f"Max Train Dice: {max(train_dices):.6f}" if train_dices else "",
                               f"Max Val Dice: {max(val_dices):.6f}" if val_dices else ""]

                if train_ious is not None:
                    summary_row.extend([
                        f"Max Train IoU: {max(train_ious):.6f}" if train_ious else "",
                        f"Max Val IoU: {max(val_ious):.6f}" if val_ious else ""
                    ])

                writer.writerow(summary_row)

            print(f"训练数据已保存到: {csv_path}")

        except Exception as e:
            print(f"保存训练数据到CSV失败: {e}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def state_dict(self):
        """返回早停状态"""
        return {
            'patience': self.patience,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min,
            'delta': self.delta
        }

    def load_state_dict(self, state_dict):
        """加载早停状态"""
        self.patience = state_dict['patience']
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.val_loss_min = state_dict['val_loss_min']
        self.delta = state_dict['delta']