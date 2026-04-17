import argparse
import os
import sys
from config import config

import warnings
warnings.filterwarnings("ignore")

def setup_environment():
    """设置运行环境"""
    print("设置运行环境...")
    
    # 检查必要的目录
    required_dirs = [
        config.PROCESSED_DATA_DIR,
        config.MODEL_SAVE_DIR, 
        config.LOG_DIR,
        config.VISUAL_DIR
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

def data_preparation():
    """数据准备流程"""
    print("\n开始数据准备...")
    from data_prepare import main as prepare_main
    prepare_main()

def model_training():
    """模型训练流程"""
    print("\n开始模型训练...")
    from train import main as train_main
    train_main()

def model_evaluation(model_path: str = None):
    """模型评估流程"""
    print("\n开始模型评估...")
    from val import main as eval_main
    return eval_main(model_path)

def interactive_visualization():
    """交互式可视化"""
    print("\n启动交互式可视化...")
    try:
        # 这里可以添加交互式可视化代码
        print("交互式可视化功能开发中...")
    except Exception as e:
        print(f"可视化失败: {e}")

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='乳腺癌病灶分割系统')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['all', 'prepare', 'train', 'eval', 'visualize'],
                       help='运行模式: all(全流程), prepare(数据准备), train(训练), eval(评估), visualize(可视化)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='评估时指定的模型路径')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("="*60)
    print("       乳腺癌病灶分割系统")
    print("="*60)
    
    # 设置环境
    setup_environment()
    
    try:
        if args.mode == 'all':
            # 全流程执行
            data_preparation()
            model_training()
            model_evaluation()
            interactive_visualization()
            
        elif args.mode == 'prepare':
            data_preparation()
            
        elif args.mode == 'train':
            model_training()
            
        elif args.mode == 'eval':
            model_evaluation(args.model_path)
            
        elif args.mode == 'visualize':
            interactive_visualization()
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n程序执行完成!")

if __name__ == "__main__":
    main()