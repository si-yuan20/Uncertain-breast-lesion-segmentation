import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict

from config import config
from nets.resunet_pro import create_model
from data_process import create_data_loaders
from utils import Metrics, Visualizer


class Evaluator:
    """评估器类"""

    def __init__(self, model_path: str = None):
        self.device = config.DEVICE

        # 加载模型
        self.model = create_model("resnet50_unet_seg")
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 查找最佳模型
            best_model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')
            if os.path.exists(best_model_path):
                self.load_model(best_model_path)
            else:
                raise FileNotFoundError("未找到训练好的模型")

        self.model.eval()

        # 数据加载器
        _, _, self.test_loader = create_data_loaders()

        # 可视化工具
        self.visualizer = Visualizer()

        # 评估结果
        self.results = {}

    def load_model(self, model_path: str):
        """加载模型权重"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {model_path}")
        print(f"训练轮次: {checkpoint.get('epoch', '未知')}")
        print(f"验证Dice: {checkpoint.get('dice', '未知'):.4f}")

    def evaluate(self) -> Dict[str, float]:
        """全面评估模型"""
        print("开始评估模型...")

        all_dice = []
        all_iou = []
        all_precision = []
        all_recall = []

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.test_loader, desc="评估")):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)

                # 计算指标
                dice = Metrics.calculate_dice(outputs, masks)
                iou = Metrics.calculate_iou(outputs, masks)
                precision, recall, _  = Metrics.calculate_precision_recall_accuracy(outputs, masks)

                all_dice.append(dice)
                all_iou.append(iou)
                all_precision.append(precision)
                all_recall.append(recall)

                # 可视化部分结果
                if batch_idx < 3:  # 只可视化前3个批次
                    self._visualize_batch(images, outputs, masks, batch_idx)

        # 计算平均指标
        self.results = {
            'dice': np.mean(all_dice),
            'iou': np.mean(all_iou),
            'precision': np.mean(all_precision),
            'recall': np.mean(all_recall),
            'f1_score': 2 * (np.mean(all_precision) * np.mean(all_recall)) /
                        (np.mean(all_precision) + np.mean(all_recall) + 1e-6)
        }

        self._print_results()
        self._plot_metrics()

        return self.results

    def _visualize_batch(self, images: torch.Tensor, preds: torch.Tensor,
                         targets: torch.Tensor, batch_idx: int):
        """可视化批次结果"""
        batch_size = images.shape[0]
        for i in range(min(2, batch_size)):  # 每个批次可视化2个样本
            self.visualizer.visualize_prediction(
                images[i].unsqueeze(0),
                preds[i].unsqueeze(0),
                targets[i].unsqueeze(0),
                f'test_batch{batch_idx}',
                i
            )

    def _print_results(self):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)
        for metric, value in self.results.items():
            print(f"{metric.upper():<12}: {value:.4f}")
        print("=" * 60)

    def _plot_metrics(self):
        """绘制评估指标雷达图"""
        metrics_names = ['Dice', 'IoU', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            self.results['dice'],
            self.results['iou'],
            self.results['precision'],
            self.results['recall'],
            self.results['f1_score']
        ]

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        metrics_values += metrics_values[:1]  # 闭合图形
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, metrics_values, 'o-', linewidth=2, label='性能指标')
        ax.fill(angles, metrics_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', size=14, fontweight='bold')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(config.VISUAL_DIR, 'metrics_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """生成评估报告"""
        report_path = os.path.join(config.LOG_DIR, 'evaluation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("乳腺癌病灶分割模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write("配置信息:\n")
            for key, value in config.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    f.write(f"  {key}: {value}\n")

            f.write("\n评估结果:\n")
            for metric, value in self.results.items():
                f.write(f"  {metric}: {value:.4f}\n")

            f.write(f"\n测试集大小: {len(self.test_loader.dataset)} 个切片\n")
            f.write(f"评估时间: {len(self.test_loader)} 个批次\n")

        print(f"评估报告已保存: {report_path}")


def main(model_path: str = None):
    """评估主函数"""
    evaluator = Evaluator(model_path)
    results = evaluator.evaluate()
    evaluator.generate_report()

    return results


if __name__ == "__main__":
    # 可以指定模型路径，如果不指定则使用最佳模型
    model_path = None  # 例如: "./models/best_model.pth"
    main(model_path)
