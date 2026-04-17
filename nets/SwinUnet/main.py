# test_swin_unet.py
import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 模拟config对象
class MockConfig:
    class DATA:
        IMG_SIZE = 256

    class MODEL:
        SWIN = type('obj', (object,), {
            'PATCH_SIZE': 4,
            'IN_CHANS': 3,
            'EMBED_DIM': 96,
            'DEPTHS': [2, 2, 2, 2],
            'NUM_HEADS': [3, 6, 12, 24],
            'WINDOW_SIZE': 8,
            'MLP_RATIO': 4.0,
            'QKV_BIAS': True,
            'QK_SCALE': None,
            'APE': False,
            'PATCH_NORM': True
        })()
        DROP_RATE = 0.0
        DROP_PATH_RATE = 0.1
        PRETRAIN_CKPT = None

    class TRAIN:
        USE_CHECKPOINT = False


def test_swin_unet():
    print("测试 SwinUnet 模型...")

    # 创建配置
    config = MockConfig()

    # 创建模型
    from vision_transformer import SwinUnet
    model = SwinUnet(config, img_size=256, num_classes=1)

    # 将模型设置为评估模式
    model.eval()

    # 创建随机输入数据 [batch_size, channels, height, width]
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 256, 256)

    print(f"输入形状: {input_tensor.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)

    print(f"输出形状: {output.shape}")

    # 验证形状
    expected_input_shape = torch.Size([2, 1, 256, 256])
    expected_output_shape = torch.Size([2, 1, 256, 256])

    if input_tensor.shape == expected_input_shape:
        print("✓ 输入形状正确")
    else:
        print(f"✗ 输入形状错误: 期望 {expected_input_shape}, 实际 {input_tensor.shape}")

    if output.shape == expected_output_shape:
        print("✓ 输出形状正确")
    else:
        print(f"✗ 输出形状错误: 期望 {expected_output_shape}, 实际 {output.shape}")

    # 测试单通道
    print("\n测试...")
    single_channel_input = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        single_output = model(single_channel_input)
    print(f"单通道输入形状: {single_channel_input.shape}")
    print(f"单通道输出形状: {single_output.shape}")


    return model, input_tensor, output


if __name__ == "__main__":
    # 运行测试
    model, input_tensor, output = test_swin_unet()

    # 打印模型信息
    print(f"\n模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 保存测试数据供后续使用
    torch.save({
        'model_state_dict': model.state_dict(),
        'input': input_tensor,
        'output': output
    }, 'swin_unet_test_data.pth')

    print("\n测试完成！")