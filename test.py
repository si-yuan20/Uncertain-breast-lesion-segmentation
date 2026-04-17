import torch
import torch.nn.functional as F
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
import cv2
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import nrrd
from typing import List, Tuple, Optional, Dict, Any

from config import config
from nets.resunet import create_model


class MedicalImagePredictor:
    """医学图像预测器"""

    def __init__(self, model_path=None):
        self.device = config.DEVICE
        self.model = create_model()

        # 加载模型权重
        if model_path is None:
            model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print("模型加载成功!")

        # 创建结果保存目录
        self.result_dir = "./test_results"
        self.prediction_dir = os.path.join(self.result_dir, "predictions")
        self.visualization_dir = os.path.join(self.result_dir, "visualizations")
        self.mask_dir = os.path.join(self.result_dir, "masks")

        for dir_path in [self.result_dir, self.prediction_dir,
                         self.visualization_dir, self.mask_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def is_volume_file(self, file_path: str) -> bool:
        """判断是否为体积文件（序列）"""
        volume_extensions = ['.nii', '.nii.gz', '.nrrd']
        return any(file_path.lower().endswith(ext) for ext in volume_extensions)

    def is_single_image_file(self, file_path: str) -> bool:
        """判断是否为单张图像文件"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        return any(file_path.lower().endswith(ext) for ext in image_extensions)

    def load_volume_data(self, file_path: str) -> Tuple[np.ndarray, str]:
        """加载体积文件数据"""
        try:
            if file_path.endswith('.nrrd'):
                # 加载NRRD文件
                data, header = nrrd.read(file_path)
                file_type = 'nrrd'
            else:
                # 加载NIfTI文件
                nifti_img = nib.load(file_path)
                data = nifti_img.get_fdata()
                file_type = 'nifti'

            return data, file_type
        except Exception as e:
            raise ValueError(f"加载体积文件失败 {file_path}: {str(e)}")

    def predict_single_image(self, image_path: str, save_result: bool = True) -> Tuple[
        Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        """预测单张图像或单个切片"""
        try:
            # 检查文件类型
            if self.is_volume_file(image_path):
                print(f"检测到体积文件: {image_path}")
                return self._predict_volume_slice(image_path, save_result)
            elif self.is_single_image_file(image_path):
                print(f"检测到单张图像文件: {image_path}")
                return self._predict_single_file(image_path, save_result)
            else:
                # 尝试作为图像文件处理
                print(f"未知文件类型，尝试作为图像处理: {image_path}")
                return self._predict_single_file(image_path, save_result)

        except Exception as e:
            print(f"预测图像 {image_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _predict_single_file(self, image_path: str, save_result: bool = True) -> Tuple[
        Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        """预测单个图像文件"""
        try:
            # 如果是体积文件，直接返回空结果（因为已经在 predict_single_image 中处理了）
            if self.is_volume_file(image_path):
                return None, None, None

            # 加载图像（仅处理非体积文件）
            image_slice = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image_slice is None:
                raise ValueError(f"无法读取图像文件: {image_path}")

            # 预处理
            processed_image = self._preprocess_image(image_slice)

            # 预测
            with torch.no_grad():
                input_tensor = processed_image.unsqueeze(0).to(self.device)
                seg_output, cls_output = self.model(input_tensor)

                # 后处理
                seg_pred = torch.sigmoid(seg_output).squeeze().cpu().numpy()
                cls_pred = torch.softmax(cls_output, dim=1).squeeze().cpu().numpy()

                # 二值化分割结果
                seg_binary = (seg_pred > 0.5).astype(np.uint8)

                # 分类结果
                lesion_prob = cls_pred[1] if len(cls_pred) > 1 else cls_pred[0]
                has_lesion = lesion_prob > 0.5

            # 保存结果
            result_info = {
                'image_path': image_path,
                'has_lesion': bool(has_lesion),
                'lesion_probability': float(lesion_prob),
                'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'lesion_area_pixels': int(np.sum(seg_binary)),
                'lesion_area_percentage': float(np.sum(seg_binary) / (seg_binary.shape[0] * seg_binary.shape[1]) * 100),
                'file_type': 'single_image'
            }

            if save_result:
                self._save_single_result(image_slice, seg_pred, seg_binary, result_info)

            return result_info, seg_binary, seg_pred

        except Exception as e:
            print(f"预测单文件 {image_path} 时出错: {str(e)}")
            return None, None, None

    def _predict_volume_slice(self, volume_path: str, save_result: bool = True, slice_idx: Optional[int] = None) -> \
    Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        """预测体积文件中的单个切片"""
        try:
            # 加载体积数据
            volume_data, file_type = self.load_volume_data(volume_path)

            # 确定切片索引
            if slice_idx is None:
                if len(volume_data.shape) > 2:
                    slice_idx = volume_data.shape[2] // 2
                else:
                    slice_idx = 0

            # 提取切片
            if len(volume_data.shape) > 2:
                image_slice = volume_data[:, :, slice_idx]
            else:
                image_slice = volume_data

            # 预处理
            processed_image = self._preprocess_image(image_slice)

            # 预测
            with torch.no_grad():
                input_tensor = processed_image.unsqueeze(0).to(self.device)
                seg_output, cls_output = self.model(input_tensor)

                # 后处理
                seg_pred = torch.sigmoid(seg_output).squeeze().cpu().numpy()
                cls_pred = torch.softmax(cls_output, dim=1).squeeze().cpu().numpy()

                # 二值化分割结果
                seg_binary = (seg_pred > 0.5).astype(np.uint8)

                # 分类结果
                lesion_prob = cls_pred[1] if len(cls_pred) > 1 else cls_pred[0]
                has_lesion = lesion_prob > 0.5

            # 保存结果
            result_info = {
                'image_path': volume_path,
                'slice_index': slice_idx,
                'total_slices': volume_data.shape[2] if len(volume_data.shape) > 2 else 1,
                'has_lesion': bool(has_lesion),
                'lesion_probability': float(lesion_prob),
                'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'lesion_area_pixels': int(np.sum(seg_binary)),
                'lesion_area_percentage': float(np.sum(seg_binary) / (seg_binary.shape[0] * seg_binary.shape[1]) * 100),
                'file_type': 'volume_slice',
                'volume_type': file_type
            }

            if save_result:
                base_name = self._get_base_name(volume_path)
                slice_suffix = f"_slice_{slice_idx:03d}"
                self._save_single_result(image_slice, seg_pred, seg_binary, result_info,
                                         base_name + slice_suffix)

            return result_info, seg_binary, seg_pred

        except Exception as e:
            print(f"预测体积切片 {volume_path} (切片 {slice_idx}) 时出错: {str(e)}")
            return None, None, None

    def predict_volume(self, volume_path: str, output_name: Optional[str] = None) -> List[Dict]:
        """预测整个体积文件的所有切片"""
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"体积文件不存在: {volume_path}")

        print(f"开始预测体积文件: {volume_path}")

        try:
            # 加载体积数据
            volume_data, file_type = self.load_volume_data(volume_path)

            if len(volume_data.shape) <= 2:
                print("警告: 文件似乎不是3D体积，将作为2D图像处理")
                result_info, seg_binary, seg_pred = self._predict_volume_slice(volume_path)
                return [result_info] if result_info else []

            total_slices = volume_data.shape[2]
            print(f"体积文件包含 {total_slices} 个切片")

            # 预测所有切片
            all_results = []
            successful_predictions = 0

            for slice_idx in tqdm(range(total_slices), desc="预测切片进度"):
                result_info, seg_binary, seg_pred = self._predict_volume_slice(
                    volume_path, save_result=True, slice_idx=slice_idx
                )

                if result_info is not None:
                    all_results.append(result_info)
                    successful_predictions += 1

            # 保存总体结果
            if output_name is None:
                output_name = self._get_base_name(volume_path)

            self._save_volume_results(all_results, output_name, volume_path)

            # 生成统计报告
            self._generate_volume_statistics(all_results, output_name)

            print(f"体积预测完成! 成功处理 {successful_predictions}/{total_slices} 个切片")
            return all_results

        except Exception as e:
            print(f"预测体积文件 {volume_path} 时出错: {str(e)}")
            return []

    def predict_image_sequence(self, sequence_dir: str, output_name: Optional[str] = None) -> List[Dict]:
        """预测图像序列文件夹"""
        if not os.path.exists(sequence_dir):
            raise FileNotFoundError(f"图像序列目录不存在: {sequence_dir}")

        print(f"开始预测图像序列: {sequence_dir}")

        # 获取所有支持的文件
        all_files = []

        # 查找体积文件
        for ext in ['*.nii', '*.nii.gz', '*.nrrd']:
            all_files.extend(self._find_files(sequence_dir, ext))

        # 查找单张图像文件
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
            all_files.extend(self._find_files(sequence_dir, ext))

        if not all_files:
            raise ValueError(f"在目录 {sequence_dir} 中未找到支持的图像文件")

        print(f"找到 {len(all_files)} 个文件")

        # 分类处理文件
        volume_files = [f for f in all_files if self.is_volume_file(f)]
        single_image_files = [f for f in all_files if self.is_single_image_file(f)]

        print(f"体积文件: {len(volume_files)} 个")
        print(f"单张图像文件: {len(single_image_files)} 个")

        all_results = []
        successful_predictions = 0

        # 处理体积文件
        for volume_file in volume_files:
            print(f"处理体积文件: {volume_file}")
            volume_results = self.predict_volume(volume_file, output_name)
            all_results.extend(volume_results)
            successful_predictions += len(volume_results)

        # 处理单张图像文件
        for image_file in tqdm(single_image_files, desc="预测单张图像"):
            result_info, seg_binary, seg_pred = self.predict_single_image(image_file, save_result=True)

            if result_info is not None:
                all_results.append(result_info)
                successful_predictions += 1

        # 保存总体结果
        if output_name is None:
            output_name = os.path.basename(sequence_dir.rstrip('/\\'))

        self._save_sequence_results(all_results, output_name)

        # 生成统计报告
        self._generate_statistics_report(all_results, output_name)

        print(f"序列预测完成! 成功处理 {successful_predictions} 个图像/切片")
        return all_results

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理"""
        # 归一化
        image = image.astype(np.float32)
        if np.max(image) > 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # 调整尺寸
        if image.shape != config.IMAGE_SIZE:
            image = resize(image, config.IMAGE_SIZE, order=3, preserve_range=True, anti_aliasing=True)

        # 转换为Tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]

        return image_tensor

    def _get_base_name(self, file_path: str) -> str:
        """获取文件基础名称"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]
        if base_name.endswith('.nrrd'):
            base_name = os.path.splitext(base_name)[0]
        return base_name

    def _find_files(self, directory: str, pattern: str) -> List[str]:
        """查找匹配模式的文件"""
        import glob
        search_pattern = os.path.join(directory, '**', pattern)
        return glob.glob(search_pattern, recursive=True)

    def _save_single_result(self, original_image: np.ndarray, seg_prob: np.ndarray,
                            seg_binary: np.ndarray, result_info: Dict,
                            custom_base_name: Optional[str] = None):
        """保存单张图像的结果"""
        if custom_base_name:
            base_name = custom_base_name
        else:
            base_name = self._get_base_name(result_info['image_path'])

        # 保存概率图
        prob_map_path = os.path.join(self.prediction_dir, f"{base_name}_prob.npy")
        np.save(prob_map_path, seg_prob)

        # 保存二值掩码
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, seg_binary * 255)

        # 保存可视化结果
        self._create_visualization(original_image, seg_prob, seg_binary, result_info, base_name)

        # 保存结果信息
        info_path = os.path.join(self.prediction_dir, f"{base_name}_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, indent=2, ensure_ascii=False)

    def _create_visualization(self, original_image: np.ndarray, seg_prob: np.ndarray,
                              seg_binary: np.ndarray, result_info: Dict, base_name: str):
        """创建可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 原始图像
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')

        # 概率图
        prob_map = axes[0, 1].imshow(seg_prob, cmap='hot')
        axes[0, 1].set_title('病灶概率图')
        axes[0, 1].axis('off')
        plt.colorbar(prob_map, ax=axes[0, 1])

        # 二值分割结果
        axes[1, 0].imshow(seg_binary, cmap='gray')
        axes[1, 0].set_title('二值分割结果')
        axes[1, 0].axis('off')

        # 叠加显示
        axes[1, 1].imshow(original_image, cmap='gray')
        axes[1, 1].imshow(seg_prob, cmap='hot', alpha=0.5)
        axes[1, 1].set_title('原始图像+病灶概率')
        axes[1, 1].axis('off')

        # 添加文本信息
        lesion_status = "有病灶" if result_info['has_lesion'] else "无病灶"
        prob_text = f"病灶概率: {result_info['lesion_probability']:.3f}\n"
        area_text = f"病灶面积: {result_info['lesion_area_pixels']} 像素 ({result_info['lesion_area_percentage']:.2f}%)"

        # 如果是体积切片，添加切片信息
        slice_info = ""
        if 'slice_index' in result_info:
            slice_info = f"切片: {result_info['slice_index'] + 1}/{result_info['total_slices']}\n"

        plt.figtext(0.5, 0.01, f"诊断结果: {lesion_status}\n{slice_info}{prob_text}{area_text}",
                    ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

        plt.tight_layout()

        # 保存图像
        viz_path = os.path.join(self.visualization_dir, f"{base_name}_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_single_result(self, original_image: np.ndarray, seg_prob: np.ndarray,
                            seg_binary: np.ndarray, result_info: Dict,
                            custom_base_name: Optional[str] = None):
        """保存单张图像的结果"""
        if custom_base_name:
            base_name = custom_base_name
        else:
            base_name = self._get_base_name(result_info['image_path'])

        # 保存概率图
        prob_map_path = os.path.join(self.prediction_dir, f"{base_name}_prob.npy")
        np.save(prob_map_path, seg_prob)

        # 保存二值掩码
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, seg_binary * 255)

        # 保存可视化结果
        self._create_visualization(original_image, seg_prob, seg_binary, result_info, base_name)

        # 保存结果信息
        info_path = os.path.join(self.prediction_dir, f"{base_name}_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, indent=2, ensure_ascii=False)

    def _save_volume_results(self, all_results: List[Dict], output_name: str, volume_path: str):
        """保存体积文件预测结果"""
        # 保存为CSV
        csv_path = os.path.join(self.result_dir, f"{output_name}_volume_predictions.csv")

        df_data = []
        for result in all_results:
            row_data = {
                '文件路径': result['image_path'],
                '切片索引': result.get('slice_index', 'N/A'),
                '总切片数': result.get('total_slices', 'N/A'),
                '是否有病灶': '是' if result['has_lesion'] else '否',
                '病灶概率': f"{result['lesion_probability']:.4f}",
                '病灶像素数': result['lesion_area_pixels'],
                '病灶面积百分比': f"{result['lesion_area_percentage']:.2f}%",
                '预测时间': result['prediction_time']
            }
            df_data.append(row_data)

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存为JSON
        json_path = os.path.join(self.result_dir, f"{output_name}_volume_predictions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"体积预测结果已保存至: {csv_path}")
        print(f"详细结果已保存至: {json_path}")

    def _save_sequence_results(self, all_results: List[Dict], output_name: str):
        """保存序列预测结果"""
        # 保存为CSV
        csv_path = os.path.join(self.result_dir, f"{output_name}_predictions.csv")

        df_data = []
        for result in all_results:
            row_data = {
                '文件路径': result['image_path'],
                '文件类型': result.get('file_type', 'unknown'),
                '是否有病灶': '是' if result['has_lesion'] else '否',
                '病灶概率': f"{result['lesion_probability']:.4f}",
                '病灶像素数': result['lesion_area_pixels'],
                '病灶面积百分比': f"{result['lesion_area_percentage']:.2f}%",
                '预测时间': result['prediction_time']
            }

            # 添加切片信息（如果是体积文件）
            if 'slice_index' in result:
                row_data['切片索引'] = result['slice_index']
                row_data['总切片数'] = result['total_slices']

            df_data.append(row_data)

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 保存为JSON
        json_path = os.path.join(self.result_dir, f"{output_name}_predictions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"预测结果已保存至: {csv_path}")
        print(f"详细结果已保存至: {json_path}")

    def _generate_volume_statistics(self, all_results: List[Dict], output_name: str):
        """生成体积文件统计报告"""
        if not all_results:
            return

        # 计算统计信息
        total_slices = len(all_results)
        lesion_slices = sum(1 for r in all_results if r['has_lesion'])
        no_lesion_slices = total_slices - lesion_slices

        lesion_probs = [r['lesion_probability'] for r in all_results]
        lesion_areas = [r['lesion_area_percentage'] for r in all_results if r['has_lesion']]

        stats = {
            '总体统计': {
                '总切片数': total_slices,
                '有病灶切片数': lesion_slices,
                '无病灶切片数': no_lesion_slices,
                '病灶检出率': f"{(lesion_slices / total_slices * 100):.2f}%"
            },
            '概率统计': {
                '平均病灶概率': f"{np.mean(lesion_probs):.4f}",
                '最大病灶概率': f"{np.max(lesion_probs):.4f}",
                '最小病灶概率': f"{np.min(lesion_probs):.4f}",
                '概率标准差': f"{np.std(lesion_probs):.4f}"
            }
        }

        if lesion_areas:
            stats['面积统计'] = {
                '平均病灶面积百分比': f"{np.mean(lesion_areas):.2f}%",
                '最大病灶面积': f"{np.max(lesion_areas):.2f}%",
                '最小病灶面积': f"{np.min(lesion_areas):.2f}%",
                '面积标准差': f"{np.std(lesion_areas):.2f}%"
            }

        # 保存统计报告
        stats_path = os.path.join(self.result_dir, f"{output_name}_volume_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 打印统计信息
        print("\n" + "=" * 50)
        print("体积文件预测统计报告")
        print("=" * 50)
        for category, category_stats in stats.items():
            print(f"\n{category}:")
            for key, value in category_stats.items():
                print(f"  {key}: {value}")
        print("=" * 50)

        # 创建统计图表
        self._create_volume_statistics_plots(all_results, output_name)

    def _generate_statistics_report(self, all_results: List[Dict], output_name: str):
        """生成统计报告"""
        if not all_results:
            return

        # 计算统计信息
        total_images = len(all_results)
        lesion_images = sum(1 for r in all_results if r['has_lesion'])
        no_lesion_images = total_images - lesion_images

        lesion_probs = [r['lesion_probability'] for r in all_results]
        lesion_areas = [r['lesion_area_percentage'] for r in all_results if r['has_lesion']]

        stats = {
            '总图像数': total_images,
            '有病灶图像数': lesion_images,
            '无病灶图像数': no_lesion_images,
            '病灶检出率': f"{(lesion_images / total_images * 100):.2f}%",
            '平均病灶概率': f"{np.mean(lesion_probs):.4f}",
            '最大病灶概率': f"{np.max(lesion_probs):.4f}",
            '最小病灶概率': f"{np.min(lesion_probs):.4f}",
        }

        if lesion_areas:
            stats.update({
                '平均病灶面积百分比': f"{np.mean(lesion_areas):.2f}%",
                '最大病灶面积': f"{np.max(lesion_areas):.2f}%",
                '最小病灶面积': f"{np.min(lesion_areas):.2f}%",
            })

        # 保存统计报告
        stats_path = os.path.join(self.result_dir, f"{output_name}_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 打印统计信息
        print("\n" + "=" * 50)
        print("预测统计报告")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 50)

        # 创建统计图表
        self._create_statistics_plots(all_results, output_name)

    def _create_volume_statistics_plots(self, all_results: List[Dict], output_name: str):
        """创建体积文件统计图表"""
        # 病灶概率分布直方图
        lesion_probs = [r['lesion_probability'] for r in all_results]

        plt.figure(figsize=(12, 8))

        # 概率分布
        plt.subplot(2, 2, 1)
        plt.hist(lesion_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('病灶概率')
        plt.ylabel('切片数量')
        plt.title('病灶概率分布直方图')
        plt.grid(True, alpha=0.3)

        # 按切片索引的概率变化
        plt.subplot(2, 2, 2)
        slice_indices = [r.get('slice_index', i) for i, r in enumerate(all_results)]
        plt.scatter(slice_indices, lesion_probs, alpha=0.6, color='coral')
        plt.xlabel('切片索引')
        plt.ylabel('病灶概率')
        plt.title('各切片病灶概率分布')
        plt.grid(True, alpha=0.3)

        # 病灶面积分布（仅对有病灶的图像）
        lesion_areas = [r['lesion_area_percentage'] for r in all_results if r['has_lesion']]
        if lesion_areas:
            plt.subplot(2, 2, 3)
            plt.hist(lesion_areas, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('病灶面积百分比 (%)')
            plt.ylabel('切片数量')
            plt.title('病灶面积分布直方图')
            plt.grid(True, alpha=0.3)

        # 病灶检测情况
        plt.subplot(2, 2, 4)
        lesion_status = [1 if r['has_lesion'] else 0 for r in all_results]
        plt.plot(lesion_status, 'o-', alpha=0.7, color='green')
        plt.xlabel('切片索引')
        plt.ylabel('是否有病灶 (1=有, 0=无)')
        plt.title('各切片病灶检测情况')
        plt.grid(True, alpha=0.3)
        plt.yticks([0, 1])

        plt.tight_layout()

        prob_plot_path = os.path.join(self.result_dir, f"{output_name}_volume_statistics.png")
        plt.savefig(prob_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_statistics_plots(self, all_results: List[Dict], output_name: str):
        """创建统计图表"""
        # 病灶概率分布直方图
        lesion_probs = [r['lesion_probability'] for r in all_results]

        plt.figure(figsize=(10, 6))
        plt.hist(lesion_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('病灶概率')
        plt.ylabel('图像数量')
        plt.title('病灶概率分布直方图')
        plt.grid(True, alpha=0.3)

        prob_plot_path = os.path.join(self.result_dir, f"{output_name}_probability_distribution.png")
        plt.savefig(prob_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 病灶面积分布（仅对有病灶的图像）
        lesion_areas = [r['lesion_area_percentage'] for r in all_results if r['has_lesion']]
        if lesion_areas:
            plt.figure(figsize=(10, 6))
            plt.hist(lesion_areas, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('病灶面积百分比 (%)')
            plt.ylabel('图像数量')
            plt.title('病灶面积分布直方图')
            plt.grid(True, alpha=0.3)

            area_plot_path = os.path.join(self.result_dir, f"{output_name}_area_distribution.png")
            plt.savefig(area_plot_path, dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """主测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description='医学图像分割预测')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像文件路径、体积文件路径或文件夹路径')
    parser.add_argument('--model', type=str, default=None,
                        help='模型权重路径 (默认为最佳模型)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='输出结果名称')
    # 移除 process_volume 参数，因为现在对于体积文件默认处理整个序列
    # parser.add_argument('--process_volume', action='store_true',
    #                    help='强制将输入文件作为体积文件处理（对所有切片进行预测）')

    args = parser.parse_args()

    # 创建预测器
    predictor = MedicalImagePredictor(model_path=args.model)

    # 执行预测
    if os.path.isfile(args.input):
        if predictor.is_volume_file(args.input):
            # 体积文件，处理所有切片（默认行为）
            print(f"预测体积文件（所有切片）: {args.input}")
            all_results = predictor.predict_volume(args.input, args.output_name)

            if all_results:
                print(f"\n体积文件预测完成! 共处理 {len(all_results)} 个切片")
                lesion_slices = sum(1 for r in all_results if r['has_lesion'])
                print(f"其中有病灶的切片: {lesion_slices} 个")
                print(f"结果保存至: {predictor.result_dir}")

        else:
            # 单文件预测（非体积文件）
            print(f"预测单文件: {args.input}")
            result_info, seg_binary, seg_prob = predictor.predict_single_image(args.input)

            if result_info:
                print("\n预测结果:")
                print(f"  文件类型: {result_info.get('file_type', 'unknown')}")
                if 'slice_index' in result_info:
                    print(f"  切片索引: {result_info['slice_index'] + 1}/{result_info['total_slices']}")
                print(f"  是否有病灶: {'是' if result_info['has_lesion'] else '否'}")
                print(f"  病灶概率: {result_info['lesion_probability']:.4f}")
                print(f"  病灶面积: {result_info['lesion_area_pixels']} 像素 ({result_info['lesion_area_percentage']:.2f}%)")
                print(f"  结果保存至: {predictor.result_dir}")

    elif os.path.isdir(args.input):
        # 图像序列预测
        predictor.predict_image_sequence(args.input, args.output_name)

    else:
        print(f"输入路径不存在: {args.input}")


if __name__ == "__main__":
    # 示例用法
    # python test.py --input "path/to/single/image.png"  # 单张图像
    # python test.py --input "path/to/volume.nii.gz"     # 体积文件（仅中间切片）
    # python test.py --input "path/to/volume.nrrd" --process_volume  # 体积文件（所有切片）
    # python test.py --input "path/to/your/images" --output_name "my_predictions"  # 文件夹

    main()