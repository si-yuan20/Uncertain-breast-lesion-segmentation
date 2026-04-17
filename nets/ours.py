# -*- coding: gbk -*-
from __future__ import annotations
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from mamba_ssm import Mamba
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.convnext import LayerNorm2d, CNBlockConfig

from .attention_zoo import build_attention


class ResConvBlock(nn.Module):
    """轻量残差卷积块：用于替代浅层Mamba，避免全分辨率序列开销"""
    def __init__(self, spatial_dims: int, in_channels: int, norm: tuple | str,
                 kernel_size: int = 3, act: tuple | str = ("RELU", {"inplace": True})):
        super().__init__()
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)

        # 用DWConv+PWConv保持轻量
        self.conv = get_dwconv_layer(spatial_dims, in_channels, in_channels, kernel_size=kernel_size)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x)
        x = x + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x


class UDAFFusion(nn.Module):
    """
    UDAF: Uncertainty-guided Deformable Alignment Fusion
    - Deformable alignment: predict 2D flow and warp CNN features to Mamba space
    - Uncertainty gating: predict pixel-wise uncertainties for two streams and softmax them to weights
    - Bi-directional interaction: lightweight gated residual injection
    Returns:
        fused: (B, out_channels, H, W)
        align_loss: scalar tensor (for training monitor / auxiliary loss)
    """
    def __init__(
        self,
        spatial_dims: int,
        conv_channels: int,
        mamba_channels: int,
        out_channels: int,
        latent_dim: int | None = None,
        max_offset: float = 4.0,       # max pixel displacement for warp
        flow_smooth_weight: float = 0.05,
        align_l1_weight: float = 1.0,
        eps: float = 1e-6
    ):
        super().__init__()

        if spatial_dims != 2:
            raise ValueError("UDAFFusion当前实现仅支持2D（spatial_dims=2）。")

        if latent_dim is None:
            latent_dim = mamba_channels

        self.spatial_dims = spatial_dims
        self.max_offset = float(max_offset)
        self.flow_smooth_weight = float(flow_smooth_weight)
        self.align_l1_weight = float(align_l1_weight)
        self.eps = float(eps)

        # ? 缓存 base grid（避免每个 forward 重建大 grid）
        self.register_buffer("_base_grid", torch.empty(0), persistent=False)
        self._base_grid_hw = (-1, -1)
        self._base_grid_dtype = None
        self._base_grid_device = None

        # 1) Channel align: ConvNeXt -> mamba_channels (保持你现有融合策略一致)
        self.conv_proj = get_conv_layer(
            spatial_dims, conv_channels, mamba_channels, kernel_size=1
        )

        # 2) Project both to latent_dim (让对齐&交互发生在同一空间)
        self.cnn_latent = get_conv_layer(
            spatial_dims, mamba_channels, latent_dim, kernel_size=1
        )
        self.mamba_latent = get_conv_layer(
            spatial_dims, mamba_channels, latent_dim, kernel_size=1
        )

        # 3) Flow predictor (deform-align): input [cnn_latent, mamba_latent] -> flow(2,H,W)
        #    用轻量 DWConv + 1x1 组合，稳定&省算
        self.flow_net = nn.Sequential(
            # DWConv
            nn.Conv2d(latent_dim * 2, latent_dim * 2, kernel_size=3, padding=1, groups=latent_dim * 2, bias=False),
            nn.BatchNorm2d(latent_dim * 2),
            nn.ReLU(inplace=True),
            # PWConv
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            # predict flow
            nn.Conv2d(latent_dim, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),  # constrain to [-1,1], then scale by max_offset
        )

        # 4) Uncertainty estimators: predict u in [0,1]
        self.unc_cnn = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2 if latent_dim >= 2 else 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim // 2 if latent_dim >= 2 else 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2 if latent_dim >= 2 else 1, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.unc_mamba = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2 if latent_dim >= 2 else 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim // 2 if latent_dim >= 2 else 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2 if latent_dim >= 2 else 1, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 5) Bi-directional interaction gate
        self.gate = nn.Sequential(
            nn.Conv2d(latent_dim * 3, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.inject_cnn = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.inject_mamba = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # 6) Output projection to out_channels (保持输出通道= mamba_channels 的约定)
        self.output_proj = get_conv_layer(
            spatial_dims, latent_dim, out_channels, kernel_size=1
        )

    @staticmethod
    def _make_base_grid(B: int, H: int, W: int, device, dtype):
        # grid_sample expects normalized grid in [-1,1]
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2) with (x,y)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)   # (B,H,W,2)
        return grid

    def _warp_with_flow(self, x: torch.Tensor, flow_px: torch.Tensor):
        """
        x: (B,C,H,W)
        flow_px: (B,2,H,W) in pixels, order (dx, dy)
        """
        B, C, H, W = x.shape

        # pixel -> normalized
        denom_w = max(W - 1, 1)
        denom_h = max(H - 1, 1)
        dx = flow_px[:, 0:1] * (2.0 / float(denom_w))
        dy = flow_px[:, 1:2] * (2.0 / float(denom_h))
        flow_norm = torch.cat([dx, dy], dim=1)  # (B,2,H,W)

        # ? 使用缓存 base grid，只 expand 不 repeat（避免分配）
        base_grid = self._get_or_build_base_grid(H, W, x.device, x.dtype)  # (1,H,W,2)
        grid = base_grid.expand(B, -1, -1, -1) + flow_norm.permute(0, 2, 3, 1)  # (B,H,W,2)

        x_warp = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return x_warp

    def _get_or_build_base_grid(self, H: int, W: int, device, dtype):
        """
        返回 shape=(1,H,W,2) 的 base grid（normalized, align_corners=False 对应）
        仅当 (H,W,device,dtype) 变化时才重建。
        """
        need_rebuild = (
                self._base_grid.numel() == 0
                or self._base_grid_hw != (H, W)
                or self._base_grid_device != device
                or self._base_grid_dtype != dtype
        )
        if need_rebuild:
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)
            grid = grid.unsqueeze(0)  # (1,H,W,2)

            self._base_grid = grid
            self._base_grid_hw = (H, W)
            self._base_grid_device = device
            self._base_grid_dtype = dtype

        return self._base_grid

    @staticmethod
    def _tv_loss(flow: torch.Tensor):
        # flow: (B,2,H,W)
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        return (dx.mean() + dy.mean())

    def forward(self, conv_feat: torch.Tensor, mamba_feat: torch.Tensor):
        """
        conv_feat:  (B, conv_channels, H, W)
        mamba_feat: (B, mamba_channels, H, W)
        """
        # 1) channel align
        conv_aligned = self.conv_proj(conv_feat)  # (B, mamba_channels, H, W)

        # 2) latent projection
        f_c = self.cnn_latent(conv_aligned)       # (B, latent_dim, H, W)
        f_m = self.mamba_latent(mamba_feat)       # (B, latent_dim, H, W)

        # 3) deformable alignment (warp CNN -> Mamba)
        flow_unit = self.flow_net(torch.cat([f_c, f_m], dim=1))  # (B,2,H,W) in [-1,1]
        flow_px = flow_unit * self.max_offset                   # pixels
        f_c_warp = self._warp_with_flow(f_c, flow_px)

        # 4) uncertainty gating
        u_c = self.unc_cnn(f_c_warp)   # (B,1,H,W) in [0,1]
        u_m = self.unc_mamba(f_m)      # (B,1,H,W) in [0,1]

        # weights = softmax(-u) over 2 branches
        logits = torch.cat([-u_c, -u_m], dim=1)  # (B,2,H,W)
        w = F.softmax(logits, dim=1)
        w_c, w_m = w[:, 0:1], w[:, 1:2]

        fused0 = w_c * f_c_warp + w_m * f_m  # (B,latent_dim,H,W)

        # 5) bi-directional interaction (gated residual injection)
        g = self.gate(torch.cat([f_c_warp, f_m, fused0], dim=1))  # (B,1,H,W)
        fused = fused0 + g * self.inject_mamba(f_m) + (1.0 - g) * self.inject_cnn(f_c_warp)

        # 6) auxiliary alignment loss (uncertainty-weighted L1 + flow smoothness)
        # encourage alignment mainly on "confident" regions
        conf = (1.0 - 0.5 * (u_c + u_m)).clamp(min=0.0, max=1.0)  # (B,1,H,W)
        l_align = (conf * torch.abs(f_c_warp - f_m)).mean()
        l_smooth = self._tv_loss(flow_px)
        align_loss = self.align_l1_weight * l_align + self.flow_smooth_weight * l_smooth

        # 7) output projection
        fused_out = self.output_proj(fused)  # (B,out_channels,H,W)

        return fused_out, align_loss


def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
    ):
    """深度可分离卷积层"""

    depth_conv = Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        strides=stride,
        kernel_size=kernel_size,
        bias=bias,
        conv_only=True,
        groups=in_channels
    )
    point_conv = Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=stride,
        kernel_size=1,
        bias=bias,
        conv_only=True,
        groups=1
    )
    return nn.Sequential(depth_conv, point_conv)


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt编码器 - 视觉特征提取流"""

    def __init__(
            self,
            spatial_dims: int = 2,
            in_channels: int = 1,
            pretrained: bool = False,
            freeze_backbone: bool = False,
            use_weights: bool = True
    ):
        """
        Args:
            spatial_dims: 空间维度，仅支持2D
            in_channels: 输入通道数
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone权重
            use_weights: 是否使用ImageNet预训练权重
        """
        super().__init__()

        if spatial_dims != 2:
            raise ValueError("ConvNeXt仅支持2D图像处理")

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.pretrained = bool(pretrained)
        self.use_weights = bool(use_weights)

        # -----------------------------
        # 关键修复：
        # 只有 pretrained=True 且 use_weights=True 时，才允许加载官方权重
        # 否则一律 weights=None，避免 torchvision 联网下载
        # -----------------------------
        weights = None
        if self.pretrained and self.use_weights:
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1

        self.convnext = convnext_base(weights=weights)

        # 修改第一层以接受任意输入通道数
        original_first_conv = self.convnext.features[0][0]
        self.convnext.features[0][0] = nn.Conv2d(
            in_channels,
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False
        )

        # 如果输入通道不同，重新初始化第一层权重
        if in_channels != 3:
            nn.init.kaiming_normal_(
                self.convnext.features[0][0].weight,
                mode='fan_out',
                nonlinearity='relu'
            )

        # 冻结backbone权重（可选）
        if freeze_backbone:
            for param in self.convnext.parameters():
                param.requires_grad = False

        # 提取各阶段特征输出维度
        self.feature_channels = []
        current_channels = in_channels

        for stage in self.convnext.features:
            if isinstance(stage, nn.Sequential):
                for block in stage:
                    if hasattr(block, 'block') and hasattr(block.block, 'layers'):
                        for layer in block.block.layers:
                            if isinstance(layer, nn.Conv2d):
                                current_channels = layer.out_channels
            elif isinstance(stage, nn.Conv2d):
                current_channels = stage.out_channels

            if current_channels not in self.feature_channels:
                self.feature_channels.append(current_channels)

        if len(self.feature_channels) < 4:
            self.feature_channels = self.feature_channels[:4]
        else:
            self.feature_channels = [self.feature_channels[0]] + self.feature_channels[1:5]

        # 与现有网络其余部分保持一致
        self.feature_channels = [128, 256, 512, 1024]

    def forward(self, x: torch.Tensor):
        """前向传播，返回各阶段特征图"""
        features = []

        for i, stage in enumerate(self.convnext.features):
            x = stage(x)
            if i in [1, 3, 5, 7]:
                features.append(x)

        return features


class MambaLayer(nn.Module):
    """Mamba层 - 序列建模"""

    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 层归一化
        self.norm = nn.LayerNorm(input_dim)

        # Mamba SSM层
        self.mamba = Mamba(
            d_model=input_dim,  # 模型维度
            d_state=d_state,  # SSM状态扩展因子
            d_conv=d_conv,  # 局部卷积宽度
            expand=expand,  # 块扩展因子
        )

        # 投影层
        self.proj = nn.Linear(input_dim, output_dim)

        # 可学习的跳跃连接缩放因子
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 确保输入类型
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]
        assert C == self.input_dim, f"输入通道数{C}不等于预期{self.input_dim}"

        # 获取图像维度
        img_dims = x.shape[2:]
        n_tokens = x.shape[2:].numel()

        # 重塑为序列格式: (B, n_tokens, C)
        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)

        # Mamba处理
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat

        # 投影和重塑回图像格式
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(1, 2).reshape(B, self.output_dim, *img_dims)

        return out


def get_mamba_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1
):
    """获取Mamba层，可选下采样"""
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)

    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(
                mamba_layer,
                nn.MaxPool2d(kernel_size=stride, stride=stride)
            )
        elif spatial_dims == 3:
            return nn.Sequential(
                mamba_layer,
                nn.MaxPool3d(kernel_size=stride, stride=stride)
            )

    return mamba_layer


class ResMambaBlock(nn.Module):
    """残差Mamba块 - 用于Mamba流"""

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size应为奇数")

        # 归一化层
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)

        # 激活函数
        self.act = get_act_layer(act)

        # Mamba层
        self.conv1 = get_mamba_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)
        self.conv2 = get_mamba_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):
        identity = x

        # 第一个Mamba块
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        # 第二个Mamba块
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        # 残差连接
        x += identity

        return x


class ResUpBlock(nn.Module):
    """上采样残差块"""

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size应为奇数")

        # 归一化层
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)

        # 激活函数
        self.act = get_act_layer(act)

        # 深度可分离卷积
        self.conv = get_dwconv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size
        )

        # 可学习的跳跃连接缩放因子
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)

        return x


class CrossModalFusion(nn.Module):
    """原始跨模态融合模块（保持兼容性）"""
    def __init__(self, spatial_dims, conv_channels, mamba_channels, out_channels, fusion_type="concat"):
        super().__init__()

        self.fusion_type = fusion_type

        # ConvNeXt → Mamba 通道对齐（关键）
        self.conv_proj = get_conv_layer(
            spatial_dims,
            conv_channels,
            mamba_channels,
            kernel_size=1
        )

        if fusion_type == "concat":
            self.fusion_conv = get_conv_layer(
                spatial_dims,
                mamba_channels * 2,
                out_channels,
                kernel_size=1
            )
        elif fusion_type == "add":
            self.fusion_conv = get_conv_layer(
                spatial_dims,
                mamba_channels,
                out_channels,
                kernel_size=1
            )
        elif fusion_type == "attention":
            self.attn = nn.Sequential(
                get_conv_layer(spatial_dims, mamba_channels * 2, mamba_channels, 1),
                nn.Sigmoid()
            )
            self.fusion_conv = get_conv_layer(
                spatial_dims,
                mamba_channels,
                out_channels,
                kernel_size=1
            )
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")

    def forward(self, conv_feat, mamba_feat):
        conv_feat = self.conv_proj(conv_feat)  # 永远对齐到 mamba_channels

        if self.fusion_type == "concat":
            fused = torch.cat([conv_feat, mamba_feat], dim=1)
        elif self.fusion_type == "add":
            fused = conv_feat + mamba_feat
        else:  # attention
            attn = self.attn(torch.cat([conv_feat, mamba_feat], dim=1))
            fused = conv_feat * attn + mamba_feat * (1 - attn)

        return self.fusion_conv(fused)


class DualStreamLightMUNet(nn.Module):
    """双流LightMUNet：ConvNeXt + ResMamba"""

    def __init__(
            self,
            spatial_dims: int = 2,
            init_filters: int = 32,
            in_channels: int = 1,
            out_channels: int = 2,
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
            convnext_pretrained: bool = False,
            convnext_freeze: bool = False,
            use_weights: bool = False,
            fusion_type: str = "concat",
            hsap_latent_dim: int = None,
            mamba_start_stage: int = 2,
            post_attn_type: str = "none",
            post_attn_stages: str = "all",
            post_attn_kwargs: dict | None = None,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims`只能是2或3")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)

        # HSAP相关配置
        self.fusion_type = fusion_type
        self.hsap_latent_dim = hsap_latent_dim
        self.use_alignment_fusion = fusion_type in ["single_udaf", "udaf"]

        # 归一化配置
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"废弃的选项'norm_name={norm_name}'，请使用'norm'替代")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm

        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        # ========== 初始化流 ==========
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)

        # ========== ConvNeXt流 ==========
        self.convnext_encoder = ConvNeXtEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            pretrained=convnext_pretrained,
            freeze_backbone=convnext_freeze,
            use_weights=use_weights
        )

        # ========== Mamba流 ==========
        self.mamba_start_stage = int(mamba_start_stage)

        self.mamba_down_layers = self._make_mamba_down_layers()

        # ========== 跨模态融合层 ==========
        # 根据fusion_type选择不同的融合策略
        self.fusion_layers = self._make_fusion_layers()

        # ========= Post-Fusion Attention (ablation) =========
        self.post_attn_type = post_attn_type
        self.post_attn_stages = post_attn_stages
        self.post_attn_kwargs = post_attn_kwargs or {}

        mamba_channels = [self.init_filters * 2 ** i for i in range(len(self.blocks_down))]

        # 解析 stages
        if isinstance(self.post_attn_stages, str) and self.post_attn_stages.lower().strip() == "all":
            enabled = set(range(len(mamba_channels)))
        else:
            enabled = set(int(s) for s in str(self.post_attn_stages).split(",") if s.strip() != "")

        self.post_attn_layers = nn.ModuleList()
        for i, ch in enumerate(mamba_channels):
            if i in enabled:
                # 针对 DANet 可选降低显存：pam_downsample_scale=2/4
                self.post_attn_layers.append(build_attention(self.post_attn_type, ch, **self.post_attn_kwargs))
            else:
                self.post_attn_layers.append(nn.Identity())


        # 用于存储对齐损失（训练时可用）
        self.align_losses = []

        # ========== 解码器 ==========
        self.up_layers, self.up_samples = self._make_up_layers()

        # ========== 最终卷积层 ==========
        self.conv_final = self._make_final_conv(out_channels)

        # Dropout层
        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_mamba_down_layers(self):
        """构建Mamba流的下采样层（优化：浅层不用Mamba，避免全分辨率序列开销）"""
        mamba_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (
            self.blocks_down, self.spatial_dims, self.init_filters, self.norm
        )

        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i

            # 下采样（第0层不下采样）
            downsample = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, kernel_size=3, stride=2)
                if i > 0 else nn.Identity()
            )

            # ===== 关键：浅层用ResConvBlock，中高层用ResMambaBlock =====
            if i < self.mamba_start_stage:
                block_cls = ResConvBlock
            else:
                block_cls = ResMambaBlock

            stage_blocks = nn.Sequential(
                downsample,
                *[
                    block_cls(spatial_dims, layer_in_channels, norm=norm, act=self.act)
                    for _ in range(item)
                ]
            )
            mamba_layers.append(stage_blocks)

        return mamba_layers

    def _make_fusion_layers(self):
        """根据fusion_type构建融合层"""
        fusion_layers = nn.ModuleList()
        
        convnext_channels = [128, 256, 512, 1024]  # ConvNeXt编码器各阶段输出通道
        mamba_channels = [self.init_filters * 2 ** i for i in range(len(self.blocks_down))]
        
        for i, (conv_ch, mamba_ch) in enumerate(zip(convnext_channels, mamba_channels)):
            # 根据fusion_type选择融合模块
            if self.fusion_type in ["single_udaf", "udaf"]:
                if self.fusion_type == "single_udaf" and i == 1:
                    fusion_layers.append(
                        UDAFFusion(
                            spatial_dims=self.spatial_dims,
                            conv_channels=conv_ch,
                            mamba_channels=mamba_ch,
                            out_channels=mamba_ch,
                            latent_dim=self.hsap_latent_dim  # 复用同一个参数，保持接口简单
                        )
                    )
                elif self.fusion_type == "udaf":
                    fusion_layers.append(
                        UDAFFusion(
                            spatial_dims=self.spatial_dims,
                            conv_channels=conv_ch,
                            mamba_channels=mamba_ch,
                            out_channels=mamba_ch,
                            latent_dim=self.hsap_latent_dim
                        )
                    )
                else:
                    fusion_layers.append(
                        CrossModalFusion(
                            spatial_dims=self.spatial_dims,
                            conv_channels=conv_ch,
                            mamba_channels=mamba_ch,
                            out_channels=mamba_ch,
                            fusion_type="add"
                        )
                    )
            else:
                # 原始融合模块
                fusion_layers.append(
                    CrossModalFusion(
                        spatial_dims=self.spatial_dims,
                        conv_channels=conv_ch,
                        mamba_channels=mamba_ch,
                        out_channels=mamba_ch,
                        fusion_type=self.fusion_type
                    )
                )
        
        return fusion_layers

    def _make_up_layers(self):
        """构建上采样层"""
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )

        n_up = len(blocks_up)

        for i in range(n_up):
            encoder_out_channels = self.init_filters * 2 ** (len(self.blocks_down) - 1)
            sample_in_channels = encoder_out_channels // (2 ** i)

            # 上采样块
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )

            # 上采样层
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )

        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        """构建最终卷积层"""
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor):
        """编码器前向传播"""
        self.align_losses = []

        convnext_features = self.convnext_encoder(x)

        mamba_x = self.convInit(x)
        if self.dropout_prob is not None:
            mamba_x = self.dropout(mamba_x)

        mamba_features = []
        for down in self.mamba_down_layers:
            mamba_x = down(mamba_x)
            mamba_features.append(mamba_x)

        fused_features = []
        for i, (conv_feat, mamba_feat) in enumerate(zip(convnext_features, mamba_features)):

            # 1) 先对齐空间尺寸（必要时）
            if conv_feat.shape[2:] != mamba_feat.shape[2:]:
                conv_feat = F.interpolate(
                    conv_feat,
                    size=mamba_feat.shape[2:],
                    mode='bilinear' if self.spatial_dims == 2 else 'trilinear',
                    align_corners=False
                )

            # 2) 再做融合（UDAF需要在这里执行）
            if self.fusion_type in ["single_udaf", "udaf"] and isinstance(self.fusion_layers[i], UDAFFusion):
                fused_feat, align_loss = self.fusion_layers[i](conv_feat, mamba_feat)
                self.align_losses.append(align_loss)
            else:
                fused_feat = self.fusion_layers[i](conv_feat, mamba_feat)

            assert fused_feat.shape[1] == mamba_feat.shape[1], \
                f"融合层{i}输出通道不匹配: {fused_feat.shape[1]} != {mamba_feat.shape[1]}"
            assert fused_feat.shape[2:] == mamba_feat.shape[2:], \
                f"融合层{i}输出空间维度不匹配"

            # 3) post attention refine (ablation)
            fused_feat = self.post_attn_layers[i](fused_feat)
            fused_features.append(fused_feat)

        encoded = fused_features[-1]
        return encoded, fused_features

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        """解码器前向传播"""

        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            # 上采样并与对应编码器特征融合
            x = up(x)

            # 调整特征图大小（如果需要）
            if x.shape[2:] != down_x[i + 1].shape[2:]:
                x = F.interpolate(
                    x,
                    size=down_x[i + 1].shape[2:],
                    mode='bilinear' if self.spatial_dims == 2 else 'trilinear',
                    align_corners=False
                )

            # 跳跃连接
            x = x + down_x[i + 1]
            x = upl(x)

        # 最终卷积
        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码
        x, down_x = self.encode(x)
        down_x.reverse()

        # 解码
        x = self.decode(x, down_x)

        return x

    def get_align_losses(self):
        """获取所有层的对齐损失（用于训练监控）"""
        return self.align_losses


def create_model(
        num_classes=1,
        num_channels=1,
        fusion_type="add",
        hsap_latent_dim=None,
        post_attn_type="none",
        post_attn_stages="all",
        post_attn_kwargs=None,
        pretrained=False,
        convnext_pretrained=None,
        convnext_freeze=False,
        use_weights=None,
        **kwargs
):
    """
    统一创建接口

    关键修复：
    1. 支持外部传入 pretrained=False
    2. 不再把 convnext_pretrained 写死为 True
    3. 可视化/推理阶段默认完全离线，不联网下载 torchvision 权重
    """

    if convnext_pretrained is None:
        convnext_pretrained = bool(pretrained)

    if use_weights is None:
        use_weights = bool(convnext_pretrained)

    return DualStreamLightMUNet(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=num_classes,
        convnext_pretrained=bool(convnext_pretrained),
        convnext_freeze=bool(convnext_freeze),
        fusion_type=fusion_type,
        hsap_latent_dim=hsap_latent_dim,
        post_attn_type=post_attn_type,
        post_attn_stages=post_attn_stages,
        post_attn_kwargs=post_attn_kwargs,
        use_weights=bool(use_weights),
        **kwargs
    )



# 测试模型
if __name__ == "__main__":
    import torch

    # 测试双流模型
    print("测试双流LightMUNet模型）...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试不同的融合类型
    fusion_types = ["udaf","single_udaf"]
    
    for fusion_type in fusion_types:
        print(f"\n{'='*50}")
        print(f"测试融合类型: {fusion_type}")
        print('='*50)
        
        # 创建模型
        model = create_model(
            num_classes=1,
            num_channels=1,
            fusion_type=fusion_type,
            hsap_latent_dim=64
        ).to(device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 测试前向传播
        input_tensor = torch.randn(2, 1, 256, 256).to(device)  # (batch_size, channels, height, width)

        # 确保模型在评估模式
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            # if fusion_type in ["single_hsap", "hsap"]:
            #     align_losses = model.get_align_losses()
            #     print(f"HSAP对齐损失数量: {len(align_losses)}")

        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")


    print("\n所有融合类型测试完成!")