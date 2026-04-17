import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, List
from config import config


class AttentionBlock(nn.Module):
    """注意力模块，让模型聚焦于肿瘤区域"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# 在DecoderBlock中添加注意力机制
class AttentionDecoderBlock(nn.Module):
    """带注意力机制的解码器块"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # 上采样
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # 注意力机制
        if self.use_attention:
            self.attention = AttentionBlock(F_g=out_channels, F_l=skip_channels, F_int=out_channels // 2)

        # 特征融合后的卷积
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 上采样
        x = self.up(x)

        # 调整skip连接的尺寸
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 应用注意力机制
        if self.use_attention:
            skip = self.attention(x, skip)

        # 拼接特征
        x = torch.cat([x, skip], dim=1)

        # 卷积处理
        x = self.conv(x)
        return x


class ResNet50UNetSeg(nn.Module):
    """基于ResNet50的UNet分割网络 - 修复输出尺寸版本"""

    def __init__(self, num_classes: int = 1, num_channels: int = 1, pretrained: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels

        # 加载预训练的ResNet50
        resnet = models.resnet50(pretrained=pretrained)

        # 修改第一层卷积以适应指定通道数的医学图像
        self._adapt_first_conv(resnet, num_channels)

        # 编码器部分
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # 添加maxpool保持尺寸对齐
        )  # [B, 64, 64, 64]

        self.encoder2 = resnet.layer1  # [B, 256, 64, 64]
        self.encoder3 = resnet.layer2  # [B, 512, 32, 32]
        self.encoder4 = resnet.layer3  # [B, 1024, 16, 16]
        self.encoder5 = resnet.layer4  # [B, 2048, 8, 8]

        # 解码器部分 - 修正上采样次数
        self.decoder1 = AttentionDecoderBlock(2048, 1024, 512)  # [B, 512, 16, 16]
        self.decoder2 = AttentionDecoderBlock(512, 512, 256)  # [B, 256, 32, 32]
        self.decoder3 = AttentionDecoderBlock(256, 256, 128)  # [B, 128, 64, 64]
        self.decoder4 = AttentionDecoderBlock(128, 64, 64)  # [B, 64, 128, 128]

        # 最终上采样到原始尺寸 - 修复这里
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # [B, 32, 256, 256]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 分割输出层
        self.seg_output = nn.Conv2d(32, num_classes, kernel_size=1)  # [B, 1, 256, 256]

        # 初始化权重
        self._initialize_weights()

    def _adapt_first_conv(self, resnet, num_channels):
        """修改第一层卷积以适应指定通道数的输入"""
        original_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        with torch.no_grad():
            if original_conv.weight.shape[1] == 3 and num_channels == 1:
                # 如果原始是3通道输入，现在是1通道，取平均值
                new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            elif original_conv.weight.shape[1] == 3 and num_channels > 1:
                # 如果原始是3通道输入，现在是多通道，重复复制
                new_conv.weight.data = original_conv.weight.data.repeat(1, num_channels, 1, 1) / num_channels
            else:
                # 其他情况，使用kaiming初始化
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            if new_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

        resnet.conv1 = new_conv

    def _initialize_weights(self):
        """初始化解码器和输出层的权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 确保输入输出尺寸一致
        """
        # 编码器前向传播
        e1 = self.encoder1(x)  # [B, 64, 64, 64]
        e2 = self.encoder2(e1)  # [B, 256, 64, 64]
        e3 = self.encoder3(e2)  # [B, 512, 32, 32]
        e4 = self.encoder4(e3)  # [B, 1024, 16, 16]
        e5 = self.encoder5(e4)  # [B, 2048, 8, 8]

        # 解码器前向传播
        d1 = self.decoder1(e5, e4)  # [B, 512, 16, 16]
        d2 = self.decoder2(d1, e3)  # [B, 256, 32, 32]
        d3 = self.decoder3(d2, e2)  # [B, 128, 64, 64]
        d4 = self.decoder4(d3, e1)  # [B, 64, 128, 128]

        # 最终上采样
        d5 = self.final_upsample(d4)  # [B, 32, 256, 256]

        # 分割输出
        seg_output = self.seg_output(d5)  # [B, 1, 256, 256]

        return seg_output


class DecoderBlock(nn.Module):
    """解码器块，包含上采样、特征融合和卷积"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        # 上采样
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # 特征融合后的卷积
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 上采样
        x = self.up(x)

        # 调整skip连接的尺寸（如果需要）
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 拼接特征
        x = torch.cat([x, skip], dim=1)

        # 卷积处理
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """双卷积块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# resunet.py 修改后的完整create_model函数
def create_model(model_type: str = "resnet50_unet_seg", num_classes: int = 1, num_channels: int = 1,
                 pretrained: bool = True, **kwargs) -> nn.Module:
    """
    创建模型实例 - 修复参数签名以匹配统一接口

    Args:
        model_type: 模型类型，目前仅支持'resnet50_unet_seg'
        num_classes: 输出类别数
        num_channels: 输入通道数
        pretrained: 是否使用预训练权重
        **kwargs: 其他参数（保持向后兼容）

    Returns:
        模型实例
    """
    # 保持向后兼容：如果model_type是位置参数而不是关键字参数
    if isinstance(model_type, int) or isinstance(model_type, str) and model_type.isdigit():
        # 可能是旧的调用方式，将第一个参数视为num_classes
        num_classes = int(model_type) if isinstance(model_type, str) else model_type
        model_type = "resnet50_unet_seg"

    if model_type == "resnet50_unet_seg":
        model = ResNet50UNetSeg(
            num_classes=num_classes,
            num_channels=num_channels,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model.to(config.DEVICE)


if __name__ == "__main__":
    # 测试模型
    model = create_model("resnet50_unet_seg")
    print("模型创建成功!")

    # 打印模型结构
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 测试前向传播
    dummy_input = torch.randn(2, 1, 256, 256).to(config.DEVICE)

    try:
        seg_output = model(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"分割输出形状: {seg_output.shape}")
        print("前向传播测试成功!")

        # 检查参数梯度
        encoder_frozen = True
        for name, param in model.named_parameters():
            if "encoder" in name and param.requires_grad:
                encoder_frozen = False
                break

        if encoder_frozen:
            print("编码器参数已成功冻结")
        else:
            print("警告: 部分编码器参数未被冻结")

    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback

        traceback.print_exc()