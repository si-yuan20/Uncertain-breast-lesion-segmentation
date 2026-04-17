import torch
import torch.nn as nn
from config import config

# 导入所有模型
from .unet import create_model as create_unet
from .unetpp import create_model as create_unetpp
from .AttUnet import create_model as create_attunet
from .resunet import create_model as create_resunet
from .ours import create_model as create_ours
from .ACC_UNet import create_model as create_accunet
from .SwinUnet import create_model as create_swinunet
from .TransUnet import create_model as create_transunet
from .VMUnet.vmunet import create_model as create_vmunet
from .LightMUNet import create_model as create_lightmunet
from .DUALNet.Conv_Mamba import create_model as create_convnext_or_mamba
from .DUALNet.ConvNextMamba import create_model as create_convnextmamba
from .DUALNet.TestConvNextMamba import create_model as create_test_convnextmamba
############
from .DUALNet.temp.DualMambaSwinNet import create_model as create_dualmambaswinnet
from .DUALNet.temp.ConvNextMambaNet import create_model as create_dualmambaconvnextnet
from .DUALNet.temp.LightConvMamba import create_model as create_resmambaconvnext
from .TransFuse.TransFuse import create_model as create_transfuse
###########
import inspect


def create_model(model_name, num_classes=1, num_channels=1, pretrained=False, **kwargs):
    """
    统一的模型创建接口，支持所有实现的模型

    Args:
        model_name: 模型名称
        num_classes: 输出类别数
        num_channels: 输入通道数
        pretrained: 是否使用预训练权重
        **kwargs: 模型特定参数

    Returns:
        模型实例
    """
    model_name = model_name.lower()

    try:
        # 处理不同模型的创建逻辑
        if model_name == 'unet':
            return create_unet(model_type='unet', num_classes=num_classes, num_channels=num_channels,
                               pretrained=pretrained)


        elif model_name == 'unetpp':
            deep_supervision = kwargs.get('deep_supervision', False)
            return create_unetpp(num_classes=num_classes,
                                 num_channels=num_channels, deep_supervision=deep_supervision,
                                 pretrained=pretrained)

        elif model_name == 'attunet':
            return create_attunet(num_classes=num_classes, num_channels=num_channels)

        elif model_name == 'resunet':
            return create_resunet(num_classes=num_classes, num_channels=num_channels, pretrained=pretrained)

        elif model_name == 'acc_unet':
            n_filts = kwargs.get('n_filts', 32)
            return create_accunet(num_classes=num_classes, num_channels=num_channels, n_filts=n_filts)

        elif model_name == 'vmunet':
            return create_vmunet(num_classes=num_classes, num_channels=num_channels)

        elif model_name == 'lightmunet':
            return create_lightmunet(num_classes=num_classes, num_channels=num_channels)

        elif model_name == 'swinunet':
            img_size = kwargs.get('img_size', 256)
            return create_swinunet(num_classes=num_classes, num_channels=num_channels, img_size=img_size)
        
        elif model_name == 'transfuse':
            return create_transfuse(num_classes=num_classes, num_channels=num_channels)
            
        elif model_name == 'transunet':
            vit_name = kwargs.get('vit_name', 'ViT-B_16')
            img_size = kwargs.get('img_size', 256)
            pretrained_path = kwargs.get('pretrained_path', None)
            return create_transunet(num_classes=num_classes, num_channels=num_channels,
                                    vit_name=vit_name, img_size=img_size,
                                    pretrained_path=pretrained_path)

        elif model_name == "convnext_unet":
            return create_convnext_or_mamba(
                model_type="convnext",
                num_classes=num_classes,
                num_channels=num_channels,
                pretrained=False,  # 🔥关键
                convnext_pretrained=False  # 🔥关键
            )
        
        elif model_name =="mamba_unet":
            return create_convnext_or_mamba(model_type="mamba",num_classes=num_classes, num_channels=num_channels)
        
        elif model_name =="convnext_resmamba_add":
            return create_convnextmamba(fusion_type="add",num_classes=num_classes, num_channels=num_channels)

        elif model_name =="convnext_resmamba_concat":
            return create_convnextmamba(fusion_type="concat",num_classes=num_classes, num_channels=num_channels)

        elif model_name =="convnext_resmamba_attention":
            return create_convnextmamba(fusion_type="attention",num_classes=num_classes, num_channels=num_channels)


        elif model_name == "convnext_resmamba_udaf":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_single_udaf":

            return create_test_convnextmamba(

                fusion_type="single_udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_dice_ce":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_dice_ce_focal":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_dice_ce_boundary":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_dice_ce_focal_tversky":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_align005":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_align01":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_align02":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_se":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="se",

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_eca":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="eca",

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_cbam":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="cbam",

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_danet":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="danet",

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_coord":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="coord",

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_epsa":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="epsa",

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_triplet":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="triplet",

                pretrained=pretrained,

                **kwargs

            )


        elif model_name == "convnext_resmamba_udaf_paea":

            return create_test_convnextmamba(

                fusion_type="udaf",

                num_classes=num_classes,

                num_channels=num_channels,

                post_attn_type="paea",

                pretrained=pretrained,

                **kwargs

            )

        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

    except Exception as e:
        print(f"创建模型 {model_name} 时发生错误: {e}")
        print(f"参数: num_classes={num_classes}, num_channels={num_channels}, pretrained={pretrained}")
        print("建议检查对应模型的create_model函数签名")
        raise


def _get_model_function_signature(model_name, create_func):
    """
    获取模型创建函数的签名信息，用于调试

    Args:
        model_name: 模型名称
        create_func: 模型创建函数

    Returns:
        签名信息字典
    """
    try:
        sig = inspect.signature(create_func)
        params = list(sig.parameters.keys())
        return {
            'model_name': model_name,
            'parameters': params,
            'required_params': [p for p in params if sig.parameters[p].default == inspect.Parameter.empty]
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'error': str(e)
        }


def print_model_signatures():
    """
    打印所有模型创建函数的签名信息，用于调试
    """
    print("模型创建函数签名信息:")
    print("-" * 80)

    models = {
        'unet': create_unet,
        'unetpp': create_unetpp,
        'attunet': create_attunet,
        'resunet': create_resunet,
        'ours': create_ours,
        'acc_unet': create_accunet,
        'swinunet': create_swinunet,
        'transunet': create_transunet,
    }

    for name, func in models.items():
        sig_info = _get_model_function_signature(name, func)
        if 'error' in sig_info:
            print(f"{name}: 获取签名失败 - {sig_info['error']}")
        else:
            print(f"{name}:")
            print(f"  参数列表: {', '.join(sig_info['parameters'])}")
            print(f"  必需参数: {', '.join(sig_info['required_params'])}")
        print()


# 更新freeze_encoder函数，添加更详细的错误处理
def freeze_encoder(model, freeze=True):
    """
    冻结/解冻模型编码器部分

    Args:
        model: 模型实例
        freeze: 是否冻结

    Returns:
        冻结状态
    """
    model_name = model.__class__.__name__.lower()
    status = "冻结" if freeze else "解冻"

    try:
        # 根据不同模型类型实现冻结逻辑
        if hasattr(model, 'encoder'):
            # 通用编码器冻结
            for param in model.encoder.parameters():
                param.requires_grad = not freeze
            print(f"{model_name}编码器已{status}")

        elif 'resnet' in model_name:
            # ResNet基础模型
            layers_to_freeze = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
            frozen_count = 0

            for layer_name in layers_to_freeze:
                if hasattr(model, layer_name):
                    for param in getattr(model, layer_name).parameters():
                        param.requires_grad = not freeze
                    frozen_count += 1

            if frozen_count > 0:
                print(f"ResNet基础模型 {frozen_count} 个编码器层已{status}")
            else:
                # 尝试查找其他可能的编码器结构
                self._freeze_by_prefix(model, freeze, ['encoder', 'backbone', 'features'])

        elif 'swin' in model_name:
            # SwinUnet编码器
            for name, param in model.named_parameters():
                if 'patch_embed' in name or 'layers' in name:
                    param.requires_grad = not freeze
            print(f"SwinUnet编码器已{status}")

        elif 'trans' in model_name:
            # TransUnet编码器
            for name, param in model.named_parameters():
                if 'transformer' in name or 'embeddings' in name:
                    param.requires_grad = not freeze
            print(f"TransUnet编码器已{status}")

        else:
            # 默认冻结前半部分网络
            total_params = sum(1 for _ in model.parameters())
            frozen_params = 0

            for i, param in enumerate(model.parameters()):
                if i < total_params // 2:
                    param.requires_grad = not freeze
                    frozen_params += 1
                else:
                    param.requires_grad = freeze

            print(f"{model_name}的前{total_params // 2}个参数已{status}")

        return True

    except Exception as e:
        print(f"{model_name}编码器{status}失败: {e}")
        print(f"模型结构: {[name for name, _ in model.named_children()]}")
        return False


def _freeze_by_prefix(model, freeze, prefixes):
    """
    根据参数名前缀冻结参数

    Args:
        model: 模型
        freeze: 是否冻结
        prefixes: 前缀列表
    """
    for name, param in model.named_parameters():
        for prefix in prefixes:
            if name.startswith(prefix):
                param.requires_grad = not freeze
                break


# 更新get_model_params函数，添加更详细的信息
def get_model_params(model):
    """
    获取模型参数量信息

    Args:
        model: 模型实例

    Returns:
        参数量统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 按层类型统计参数
    param_details = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0] if '.' in name else 'other'
        if layer_type not in param_details:
            param_details[layer_type] = {
                'total': 0,
                'trainable': 0
            }
        param_details[layer_type]['total'] += param.numel()
        if param.requires_grad:
            param_details[layer_type]['trainable'] += param.numel()

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'param_details': param_details
    }


if __name__ == "__main__":
    # 首先打印所有模型的签名信息
    print_model_signatures()

    # 测试模型创建
    test_models = [
        'unet', 'resnet34_unet', 'unetpp', 'attunet', 'resunet',
        'ours', 'acc_unet', 'swinunet', 'transunet'
    ]

    print("测试模型创建接口...")

    for model_name in test_models:
        try:
            print(f"\n创建模型: {model_name}")

            # 对于ResUNet使用不同的参数尝试
            if model_name == 'resunet':
                # 尝试多种调用方式
                try:
                    model = create_model(model_name, num_classes=1, num_channels=1, pretrained=False)
                except TypeError as e:
                    if "num_classes" in str(e):
                        print(f"  ResUNet可能不支持num_classes参数，尝试备用方案...")
                        # 手动调用resunet的create_model函数
                        model = create_resunet(num_channels=1, pretrained=False)
            else:
                model = create_model(model_name, num_classes=1, num_channels=1, pretrained=False)

            params = get_model_params(model)
            print(f"  总参数量: {params['total_params']:,}")
            print(f"  可训练参数量: {params['trainable_params']:,}")

            # 测试前向传播
            dummy_input = torch.randn(1, 1, 256, 256).to(config.DEVICE)
            output = model(dummy_input)

            if isinstance(output, tuple):
                # 处理deep supervision情况
                for i, out in enumerate(output):
                    print(f"  输出{i + 1}形状: {out.shape}")
            else:
                print(f"  输出形状: {output.shape}")

            print(f"  模型创建成功!")

        except Exception as e:
            print(f"  模型创建失败: {e}")
            import traceback

            traceback.print_exc()