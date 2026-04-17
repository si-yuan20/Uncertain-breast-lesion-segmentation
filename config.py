import os
import torch


class Config:
    """配置文件"""

    # 路径设置
    RAW_DATA_DIR = "/home/medical/datasets/BreastDatasets"
    # RAW_DATA_DIR = "E:/datasets/BreastDatasets/datasets"

    PROCESSED_DATA_DIR = "./datasets"
    IMAGE_DIR = os.path.join(PROCESSED_DATA_DIR, "images")  # 原图存储目录
    MASK_DIR = os.path.join(PROCESSED_DATA_DIR, "masks")  # 掩码存储目录
    MODEL_SAVE_DIR = "./models/model"
    LOG_DIR = "./logs"
    VISUAL_DIR = "./visual"

    USE_PRETRAINED = False  # 新增：是否使用预训练权重

    # 创建目录
    for dir_path in [PROCESSED_DATA_DIR, IMAGE_DIR, MASK_DIR, MODEL_SAVE_DIR, LOG_DIR, VISUAL_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # 训练序列
    SEQ_MODE = "ALL"

    # 数据参数
    IMAGE_SIZE = (256, 256)  # 输入图像尺寸
    BATCH_SIZE = 8
    NUM_WORKERS = 8
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    DATASET_PREPARE_WORKERS = 8

    # 优化器参数
    OPTIMIZER_TYPE = "AdamW"  # AdamW, Adam, SGD
    INITIAL_LR = 3e-4
    WEIGHT_DECAY = 3e-5

    # 学习率策略
    LR_SCHEDULER = "cosine"  # cosine, step, plateau
    LR_WARMUP_EPOCHS = 5  # 学习率预热轮数
    MIN_LR = 1e-6  # 最小学习率

    # 梯度裁剪
    GRAD_CLIP = 1.0

    # 训练参数
    EPOCHS = 300
    PATIENCE = 100   # Early stopping耐心

    # 断点续训设置
    RESUME_TRAINING = False  # 是否断点续训
    RESUME_CHECKPOINT_PATH = "models/model/convnext_resmamba_udaf_align02/checkpoints/best_model.pth"  # 续训的权重路径

    # 设备设置
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 日志设置
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    IN_CHANNELS = 1
    # 数据集信息
    NUM_CLASSES = 2  # 背景和病灶

    # ==================== 新增损失函数配置 ====================
    # 损失函数类型
    LOSS_ABLATION = "dice_ce"  # 对应你的(1.1)
    ALIGN_LAMBDA = 0.0  # (1.5)默认关闭，只有 *_align + lambda>0 才启用

    # Dice损失参数
    DICE_SMOOTH = 1e-6

    # BCE损失参数
    BCE_POS_WEIGHT = 10.0  # 正样本权重，用于处理类别不平衡

    # 加权交叉熵参数
    WCE_LESION_WEIGHT = 10.0  # 病灶权重
    WCE_BACKGROUND_WEIGHT = 1.0  # 背景权重

    # base weights
    DICE_WEIGHT = 0.7
    BCE_WEIGHT = 0.3

    # focal tversky hyper
    FT_ALPHA = 0.7
    FT_BETA = 0.3
    FT_GAMMA = 0.75

    # 梯度累积
    USE_GRADIENT_ACCUMULATION = False
    ACCUMULATION_STEPS = 4

    @classmethod
    def print_runtime_config(self):
        """打印运行时配置（实例属性优先，真实生效值）"""
        print("=" * 50)
        print("运行时配置信息 (instance)")
        print("=" * 50)

        # 先列出 Config 类里定义过的字段，读取实例上的值（若被覆盖则显示覆盖后的）
        for key in sorted(
                [k for k in Config.__dict__.keys() if not k.startswith("_") and not callable(getattr(Config, k))]):
            print(f"{key}: {getattr(self, key)}")

        print("=" * 50)

    @classmethod
    def validate_resume_config(cls):
        """验证断点续训配置"""
        if cls.RESUME_TRAINING:
            if not cls.RESUME_CHECKPOINT_PATH:
                raise ValueError("断点续训已启用，但未指定RESUME_CHECKPOINT_PATH")
            if not os.path.exists(cls.RESUME_CHECKPOINT_PATH):
                raise FileNotFoundError(f"续训权重文件不存在: {cls.RESUME_CHECKPOINT_PATH}")


# 实例化配置
config = Config()