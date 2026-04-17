import torch
from torch import nn


def create_model(model_type="swinunet", num_classes=1, num_channels=1, img_size=256):
    """
    创建SwinUnet模型
    
    Args:
        model_type: 模型类型
        num_classes: 输出通道数
        num_channels: 输入通道数
        img_size: 输入图像大小
    
    Returns:
        配置好的Swin_Unet模型
    """
    if model_type == "swinunet":
        # 从swin_transformer_unet_skip_expand_decoder_sys导入完整的模型实现
        from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
        
        # 创建Swin_Unet模型实例，使用默认参数配置
        model = SwinTransformerSys(
            img_size=img_size,
            num_classes=num_classes,
            in_chans=num_channels,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            drop_path_rate=0.3,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        return model
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 测试代码
if __name__ == "__main__":
    import torch
    
    # 创建模型
    model = create_model("swinunet", num_classes=1, num_channels=1)
    
    # 测试输入
    x = torch.randn(2, 1, 256, 256)  # (batch_size, channels, height, width)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
