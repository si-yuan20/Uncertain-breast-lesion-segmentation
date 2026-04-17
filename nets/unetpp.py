from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from config import config
from typing import Optional, Tuple, List, Union
import warnings


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


class NestedUNet(nn.Module):
    def __init__(self, in_channel: int = 1, out_channel: int = 1,
                 deep_supervision: bool = False, filters: List[int] = None):
        """
        UNet++ 模型

        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            deep_supervision: 是否使用深度监督
            filters: 各层过滤器数量，默认为 [32, 64, 128, 256, 512]
        """
        super().__init__()

        self.deep_supervision = deep_supervision

        if filters is None:
            filters = [32, 64, 128, 256, 512]

        nb_filter = filters

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 第0列
        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # 第1列
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        # 第2列
        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        # 第3列
        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        # 第4列
        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        # 输出层
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # if self.deep_supervision:
        #    output1 = torch.sigmoid(self.final1(x0_1))
        #    output2 = torch.sigmoid(self.final2(x0_2))
        #    output3 = torch.sigmoid(self.final3(x0_3))
        #    output4 = torch.sigmoid(self.final4(x0_4))
        #    return [output1, output2, output3, output4]
        #else:
        #    output = torch.sigmoid(self.final(x0_4))
        #    return output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output
        # return x0_4

def create_model(num_classes: int = 1,
                 num_channels: int = 1,
                 deep_supervision: bool = False,
                 pretrained: bool = False, **kwargs) -> nn.Module:
    """
    创建模型实例

    Args:
        num_classes: 输出类别数
        num_channels: 输入通道数
        deep_supervision: 是否使用深度监督（仅UNet++有效）
        pretrained: 是否使用预训练权重（仅ResNet34UNet有效）
        **kwargs: 其他参数

    Returns:
        模型实例
    """
    model_type = kwargs.get('model_type', 'unetpp')
    if model_type.lower() == "unetpp":
        model = NestedUNet(
            in_channel=num_channels,
            out_channel=num_classes,
            deep_supervision=deep_supervision
        )
    elif model_type.lower() == "unet":
        # 从unet.py导入的Unet
        try:
            from unet import Unet
            model = Unet(
                in_ch=num_channels,
                out_ch=num_classes
            )
        except ImportError:
            raise ImportError("请确保unet.py在同一目录下")
    elif model_type.lower() == "resnet34_unet":
        try:
            from unet import ResNet34UNet
            model = ResNet34UNet(
                num_classes=num_classes,
                num_channels=num_channels,
                pretrained=pretrained
            )
        except ImportError:
            raise ImportError("请确保unet.py在同一目录下")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model.to(config.DEVICE)



if __name__ == "__main__":
    # 测试UNet++模型
    print("测试UNet++模型:")

    # 测试不使用深度监督
    print("\n1. 测试不使用深度监督:")
    model_unetpp = create_model(
        model_type="unetpp",
        num_channels=1,
        num_classes=1,
        deep_supervision=False
    )
    print("UNet++模型（无深度监督）创建成功!")

    dummy_input = torch.randn(2, 1, 256, 256).to(config.DEVICE)
    try:
        output_unetpp = model_unetpp(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output_unetpp.shape}")
        print("前向传播测试成功!")
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试使用深度监督
    print("\n2. 测试使用深度监督:")
    model_unetpp_ds = create_model(
        model_type="unetpp",
        num_channels=1,
        num_classes=1,
        deep_supervision=True
    )
    print("UNet++模型（深度监督）创建成功!")

    try:
        outputs_ds = model_unetpp_ds(dummy_input)
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出数量: {len(outputs_ds)}")
        for i, output in enumerate(outputs_ds):
            print(f"输出{i + 1}形状: {output.shape}")
        print("深度监督前向传播测试成功!")
    except Exception as e:
        print(f"深度监督前向传播失败: {e}")
        import traceback

        traceback.print_exc()

    # # 测试冻结编码器函数（应显示警告）
    # print("\n3. 测试冻结编码器函数:")
    # # freeze_encoder(model_unetpp, freeze=True)
    #
    # # 参数量统计
    # print("\n4. 参数量统计:")
    # total_params = sum(p.numel() for p in model_unetpp.parameters())
    # trainable_params = sum(p.numel() for p in model_unetpp.parameters() if p.requires_grad)
    # print(f"总参数量: {total_params:,}")
    # print(f"可训练参数量: {trainable_params:,}")
    #
    # # 测试不同的输入输出通道
    # print("\n5. 测试多通道输入输出:")
    # model_multi = create_model(
    #     model_type="unetpp",
    #     num_channels=3,
    #     num_classes=4,
    #     deep_supervision=False
    # )
    # print("多通道UNet++模型创建成功!")
    #
    # dummy_input_multi = torch.randn(2, 3, 256, 256).to(config.DEVICE)
    # try:
    #     output_multi = model_multi(dummy_input_multi)
    #     print(f"输入形状: {dummy_input_multi.shape}")
    #     print(f"输出形状: {output_multi.shape}")
    #     print("多通道前向传播测试成功!")
    # except Exception as e:
    #     print(f"多通道前向传播失败: {e}")