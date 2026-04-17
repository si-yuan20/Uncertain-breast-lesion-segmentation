import os
import torch


class Config:
    """配置文件"""

    # 路径设置
    RAW_DATA_DIR = "E:/datasets/BreastDatasets/datasets/datasets(1)"
    PROCESSED_DATA_DIR = "./datasets"
    IMAGE_DIR = os.path.join(PROCESSED_DATA_DIR, "images")  # 原图存储目录
    MASK_DIR = os.path.join(PROCESSED_DATA_DIR, "masks")  # 掩码存储目录
    MODEL_SAVE_DIR = "./models"
    LOG_DIR = "./logs"
    VISUAL_DIR = "./visual"

    # 创建目录
    for dir_path in [PROCESSED_DATA_DIR, IMAGE_DIR, MASK_DIR, MODEL_SAVE_DIR, LOG_DIR, VISUAL_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # 数据参数
    IMAGE_SIZE = (256, 256)  # 输入图像尺寸
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    # 模型参数
    MODEL_TYPE = "vit_b"  # vit_b, vit_l, vit_h
    CHECKPOINT_PATH = None  # 预训练权重路径

    # 训练参数
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping耐心

    # 设备设置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 日志设置
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5

    # 数据集信息
    NUM_CLASSES = 2  # 背景和病灶

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("训练配置信息")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)


# 实例化配置
config = Config()