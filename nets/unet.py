import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models
from config import config
from typing import Optional, Tuple, List


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        # 编码器
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # 解码器
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        return c10


nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=1, pretrained=True):
        super(ResNet34UNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        # 修改第一层卷积以适应单通道医学图像
        self._adapt_first_conv(resnet, num_channels)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def _adapt_first_conv(self, resnet, num_channels):
        """修改第一层卷积以适应单通道输入"""
        if num_channels == 1:
            original_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

            with torch.no_grad():
                if original_conv.weight.shape[1] == 3:
                    new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

                if new_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data

            resnet.conv1 = new_conv

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # return nn.Sigmoid()(out)
        return out

def create_model(model_type: str = "unet", num_classes: int = 1, num_channels: int = 1,
                 pretrained: bool = True) -> nn.Module:
    """
    创建模型实例

    Args:
        model_type: 模型类型 ("unet" 或 "resnet34_unet")
        num_classes: 输出类别数
        num_channels: 输入通道数
        pretrained: 是否使用预训练权重

    Returns:
        模型实例
    """
    if model_type == "unet":
        model = Unet(
            in_ch=num_channels,
            out_ch=num_classes
        )
    elif model_type == "resnet34_unet":
        model = ResNet34UNet(
            num_classes=num_classes,
            num_channels=num_channels,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model.to(config.DEVICE)


def freeze_encoder(model: nn.Module, freeze: bool = True):
    """
    冻结/解冻编码器 (仅适用于ResNet34UNet)

    Args:
        model: 模型实例
        freeze: 是否冻结
    """
    if isinstance(model, ResNet34UNet):
        # 冻结编码器部分
        for param in model.firstconv.parameters():
            param.requires_grad = not freeze
        for param in model.firstbn.parameters():
            param.requires_grad = not freeze
        for param in model.encoder1.parameters():
            param.requires_grad = not freeze
        for param in model.encoder2.parameters():
            param.requires_grad = not freeze
        for param in model.encoder3.parameters():
            param.requires_grad = not freeze
        for param in model.encoder4.parameters():
            param.requires_grad = not freeze

        status = "冻结" if freeze else "解冻"
        print(f"ResNet34UNet编码器已{status}")
    else:
        print("警告: 不支持的模型类型或该模型没有可冻结的编码器")


if __name__ == "__main__":
    # 测试UNet模型
    print("测试UNet模型:")
    model_unet = create_model("unet", num_channels=1, num_classes=1)
    print("UNet模型创建成功!")

    dummy_input = torch.randn(2, 1, 256, 256).to(config.DEVICE)
    try:
        output_unet = model_unet(dummy_input)
        print(f"UNet输入形状: {dummy_input.shape}")
        print(f"UNet输出形状: {output_unet.shape}")
        print("UNet前向传播测试成功!")
    except Exception as e:
        print(f"UNet前向传播失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50 + "\n")

    # 测试ResNet34UNet模型
    print("测试ResNet34UNet模型:")
    model_resnet34 = create_model("resnet34_unet", num_channels=1, num_classes=1, pretrained=True)
    print("ResNet34UNet模型创建成功!")

    try:
        output_resnet34 = model_resnet34(dummy_input)
        print(f"ResNet34UNet输入形状: {dummy_input.shape}")
        print(f"ResNet34UNet输出形状: {output_resnet34.shape}")
        print("ResNet34UNet前向传播测试成功!")

        # 测试冻结编码器
        freeze_encoder(model_resnet34, freeze=True)

        # 检查参数梯度
        encoder_frozen = True
        for name, param in model_resnet34.named_parameters():
            if any(keyword in name for keyword in ["firstconv", "firstbn", "encoder"]):
                if param.requires_grad:
                    encoder_frozen = False
                    break

        if encoder_frozen:
            print("ResNet34UNet编码器参数已成功冻结")
        else:
            print("警告: 部分ResNet34UNet编码器参数未被冻结")

    except Exception as e:
        print(f"ResNet34UNet前向传播失败: {e}")
        import traceback

        traceback.print_exc()

    # 打印模型参数数量
    print(f"\nUNet参数数量: {sum(p.numel() for p in model_unet.parameters())}")
    print(f"ResNet34UNet参数数量: {sum(p.numel() for p in model_resnet34.parameters())}")