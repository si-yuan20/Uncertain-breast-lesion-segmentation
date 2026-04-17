import os
import numpy as np
import nibabel as nib
import nrrd
from typing import List, Tuple, Dict
import tempfile
import shutil
from config import config
import re
import SimpleITK as sitk
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import cv2
from PIL import Image
import random


class NRRDToPNGConverter:
    """批量NRRD到PNG转换器 - 支持多序列数据，按病例划分数据集"""

    def __init__(self):
        self.raw_data_dir = config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.image_dir = config.IMAGE_DIR  # 原图存储目录
        self.mask_dir = config.MASK_DIR  # 掩码存储目录

        # 定义要处理的文件模式
        self.file_patterns = {
            'c2': ('img_c2.nrrd', 'seg_c2.nrrd'),
            'c5': ('img_c5.nrrd', 'seg_c5.nrrd'),
            't2': ('img_t2.nrrd', 'seg_t2.nrrd')
        }

        # 缓存已转换的文件，避免重复处理
        self._converted_cache = set()

    def find_nrrd_files(self) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """
        查找所有NRRD文件
        新的文件结构：
        RAW_DATA_DIR/
        ├── 文件夹1/
        │   ├── 病人ID1/
        │   │   ├── img_c2.nrrd
        │   │   ├── seg_c2.nrrd
        │   │   ├── img_c5.nrrd
        │   │   ├── seg_c5.nrrd
        │   │   ├── img_t2.nrrd
        │   │   └── seg_t2.nrrd
        │   └── 病人ID2/
        │       ├── ...
        ├── 文件夹2/
        │   └── ...
        └── ...
        """
        patient_files = {}

        print("正在扫描NRRD文件...")

        # 获取所有文件夹
        folders = [f for f in os.listdir(self.raw_data_dir)
                   if os.path.isdir(os.path.join(self.raw_data_dir, f))]

        for folder in tqdm(folders, desc="扫描文件夹"):
            folder_path = os.path.join(self.raw_data_dir, folder)

            # 获取所有病人目录
            patient_dirs = [p for p in os.listdir(folder_path)
                            if os.path.isdir(os.path.join(folder_path, p))]

            for patient_dir in patient_dirs:
                patient_path = os.path.join(folder_path, patient_dir)

                # 生成唯一病人ID（包含文件夹信息避免重名）
                patient_id = f"{folder}_{patient_dir}"
                patient_files[patient_id] = {}

                # 检查每个序列的文件是否存在
                for seq_name, (img_file, seg_file) in self.file_patterns.items():
                    img_path = os.path.join(patient_path, img_file)
                    seg_path = os.path.join(patient_path, seg_file)

                    if os.path.exists(img_path) and os.path.exists(seg_path):
                        patient_files[patient_id][seq_name] = (img_path, seg_path)

                # 如果该病人没有任何有效文件对，则从字典中移除
                if not patient_files[patient_id]:
                    del patient_files[patient_id]

        return patient_files

    def split_patients_by_case(self, patient_files: Dict) -> Tuple[List[str], List[str], List[str]]:
        """按病例划分训练集、验证集、测试集"""
        patient_ids = list(patient_files.keys())
        random.shuffle(patient_ids)  # 随机打乱病人顺序

        total_patients = len(patient_ids)
        train_count = int(total_patients * config.TRAIN_RATIO)
        val_count = int(total_patients * config.VAL_RATIO)

        train_patients = patient_ids[:train_count]
        val_patients = patient_ids[train_count:train_count + val_count]
        test_patients = patient_ids[train_count + val_count:]

        print(f"病例划分: 训练集 {len(train_patients)} 个病例, "
              f"验证集 {len(val_patients)} 个病例, "
              f"测试集 {len(test_patients)} 个病例")

        return train_patients, val_patients, test_patients

    def convert_batch(self, use_parallel=True, max_workers=None):
        """批量转换NRRD文件到PNG格式，按病例划分数据集"""
        patient_files = self.find_nrrd_files()

        total_sequences = sum(len(seqs) for seqs in patient_files.values())
        print(f"找到 {len(patient_files)} 个病人，共 {total_sequences} 个序列")

        if len(patient_files) == 0:
            print("错误: 未找到任何NRRD文件对，请检查数据路径和文件命名")
            return {}

        # 按病例划分数据集
        train_patients, val_patients, test_patients = self.split_patients_by_case(patient_files)

        # 构建所有需要转换的任务
        conversion_tasks = []
        for patient_id, sequences in patient_files.items():
            # 确定病例所属的数据集
            if patient_id in train_patients:
                dataset_type = 'train'
            elif patient_id in val_patients:
                dataset_type = 'val'
            else:
                dataset_type = 'test'

            for seq_name, (img_nrrd, mask_nrrd) in sequences.items():
                conversion_tasks.append({
                    'patient_id': patient_id,
                    'seq_name': seq_name,
                    'img_nrrd': img_nrrd,
                    'mask_nrrd': mask_nrrd,
                    'dataset_type': dataset_type
                })

        # 使用并行处理或顺序处理
        if use_parallel and len(conversion_tasks) > 1:
            converted_results = self._convert_parallel(conversion_tasks, max_workers)
        else:
            converted_results = self._convert_sequential(conversion_tasks)

        # 按数据集类型整理结果
        dataset_slices = {'train': [], 'val': [], 'test': []}
        for result in converted_results:
            dataset_type = result['dataset_type']
            dataset_slices[dataset_type].extend(result['slices'])

        # 保存文件列表
        self.save_file_list(dataset_slices)
        return dataset_slices

    def _convert_sequential(self, conversion_tasks):
        """顺序转换"""
        converted_results = []

        for task in tqdm(conversion_tasks, desc="转换序列", unit="seq"):
            try:
                slices = self._convert_single_task(task)
                converted_results.append({
                    'dataset_type': task['dataset_type'],
                    'slices': slices
                })
            except Exception as e:
                print(f"\n转换失败 {task['patient_id']} 的 {task['seq_name']} 序列: {str(e)}")
                continue

        return converted_results

    def _convert_parallel(self, conversion_tasks, max_workers=None):
        """并行转换"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), config.DATASET_PREPARE_WORKERS)

        converted_results = []

        print(f"使用 {max_workers} 个线程进行并行转换...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._convert_single_task, task): task
                for task in conversion_tasks
            }

            # 使用tqdm显示进度
            for future in tqdm(as_completed(future_to_task),
                               total=len(conversion_tasks),
                               desc="并行转换", unit="seq"):
                task = future_to_task[future]
                try:
                    slices = future.result()
                    converted_results.append({
                        'dataset_type': task['dataset_type'],
                        'slices': slices
                    })
                except Exception as e:
                    print(f"\n转换失败 {task['patient_id']} 的 {task['seq_name']} 序列: {str(e)}")
                    continue

        return converted_results

    def _convert_single_task(self, task):
        """转换单个任务 - 返回所有有病灶的切片路径对"""
        patient_id = task['patient_id']
        seq_name = task['seq_name']
        img_nrrd = task['img_nrrd']
        mask_nrrd = task['mask_nrrd']
        dataset_type = task['dataset_type']

        try:
            # 加载图像和掩码数据
            img_data = self.load_nrrd_data(img_nrrd)
            mask_data = self.load_nrrd_data(mask_nrrd)

            if img_data is None or mask_data is None:
                return []

            # 验证维度一致性
            if img_data.shape != mask_data.shape:
                print(
                    f"警告: {patient_id} 的 {seq_name} 序列图像和掩码维度不一致: {img_data.shape} vs {mask_data.shape}")
                return []

            converted_pairs = []

            # 遍历所有切片
            for slice_idx in range(img_data.shape[2]):
                img_slice = img_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]

                # 只保存有病灶的切片
                if np.any(mask_slice > 0):
                    # 生成文件名
                    base_name = f"{patient_id}_{seq_name}_slice_{slice_idx:03d}"
                    img_png = os.path.join(self.image_dir, f"{base_name}_image.png")
                    mask_png = os.path.join(self.mask_dir, f"{base_name}_mask.png")

                    # 保存为PNG
                    self.save_slice_as_png(img_slice, img_png, is_mask=False)
                    self.save_slice_as_png(mask_slice, mask_png, is_mask=True)

                    converted_pairs.append((img_png, mask_png))

            return converted_pairs

        except Exception as e:
            raise e

    def load_nrrd_data(self, nrrd_path: str) -> np.ndarray:
        """加载NRRD文件数据"""
        try:
            # 优先使用SimpleITK
            image = sitk.ReadImage(nrrd_path)
            data = sitk.GetArrayFromImage(image)
            # 调整维度顺序为 (H, W, D)
            data = np.transpose(data, (1, 2, 0))
            return data
        except Exception as e:
            print(f"SimpleITK加载失败 {os.path.basename(nrrd_path)}: {str(e)}")
            try:
                # 回退到nrrd方法
                data, _ = nrrd.read(nrrd_path)
                return data
            except Exception as e2:
                print(f"NRRD加载失败 {os.path.basename(nrrd_path)}: {str(e2)}")
                return None

    def save_slice_as_png(self, slice_data: np.ndarray, output_path: str, is_mask: bool = False):
        """保存切片为PNG格式"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 归一化图像数据
        if not is_mask:
            # 图像数据归一化到0-255
            slice_data = slice_data.astype(np.float32)
            if np.max(slice_data) > np.min(slice_data):
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
            else:
                slice_data = np.zeros_like(slice_data)
            slice_data = slice_data.astype(np.uint8)
        else:
            # 掩码数据二值化
            slice_data = (slice_data > 0).astype(np.uint8) * 255

        # 调整尺寸到配置的大小
        if slice_data.shape != config.IMAGE_SIZE:
            slice_data = cv2.resize(slice_data, config.IMAGE_SIZE,
                                    interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)

        # 保存为PNG
        cv2.imwrite(output_path, slice_data)

    def save_file_list(self, dataset_slices: Dict[str, List[Tuple[str, str]]]):
        """按数据集划分保存文件列表"""
        # 保存总体文件列表
        list_file = os.path.join(self.processed_dir, "file_list.txt")
        with open(list_file, 'w', encoding='utf-8') as f:
            for dataset_type in ['train', 'val', 'test']:
                for img_path, mask_path in dataset_slices[dataset_type]:
                    f.write(f"{img_path},{mask_path}\n")

        # 分别保存训练集、验证集、测试集文件列表
        train_list_file = os.path.join(self.processed_dir, "train_list.txt")
        val_list_file = os.path.join(self.processed_dir, "val_list.txt")
        test_list_file = os.path.join(self.processed_dir, "test_list.txt")

        with open(train_list_file, 'w', encoding='utf-8') as f:
            for img_path, mask_path in dataset_slices['train']:
                f.write(f"{img_path},{mask_path}\n")

        with open(val_list_file, 'w', encoding='utf-8') as f:
            for img_path, mask_path in dataset_slices['val']:
                f.write(f"{img_path},{mask_path}\n")

        with open(test_list_file, 'w', encoding='utf-8') as f:
            for img_path, mask_path in dataset_slices['test']:
                f.write(f"{img_path},{mask_path}\n")

        print(f"\n文件列表已保存:")
        print(f"  总体文件列表: {list_file}")
        print(f"  训练集文件列表: {train_list_file}")
        print(f"  验证集文件列表: {val_list_file}")
        print(f"  测试集文件列表: {test_list_file}")

        # 统计各数据集和各序列数量
        print(f"\n数据集统计:")
        for dataset_type in ['train', 'val', 'test']:
            count = len(dataset_slices[dataset_type])
            print(f"  {dataset_type}集: {count} 对PNG切片")

        # 按序列类型统计
        seq_count = {'train': {}, 'val': {}, 'test': {}}
        for dataset_type in ['train', 'val', 'test']:
            for img_path, _ in dataset_slices[dataset_type]:
                filename = os.path.basename(img_path)
                parts = filename.split('_')
                if len(parts) >= 3:
                    seq_name = parts[1]
                    seq_count[dataset_type][seq_name] = seq_count[dataset_type].get(seq_name, 0) + 1

        print(f"\n序列统计:")
        for dataset_type in ['train', 'val', 'test']:
            print(f"  {dataset_type}集:")
            for seq_name, count in seq_count[dataset_type].items():
                print(f"    {seq_name}序列: {count} 对PNG切片")


def main():
    """数据预处理主函数"""
    print("开始数据预处理...")
    print(f"原始数据目录: {config.RAW_DATA_DIR}")
    print(f"处理数据目录: {config.PROCESSED_DATA_DIR}")
    print(f"原图存储目录: {config.IMAGE_DIR}")
    print(f"掩码存储目录: {config.MASK_DIR}")

    start_time = time.time()

    # 检查原始目录是否存在
    if not os.path.exists(config.RAW_DATA_DIR):
        print(f"错误: 原始数据目录不存在: {config.RAW_DATA_DIR}")
        return

    # 创建输出目录
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    os.makedirs(config.MASK_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    converter = NRRDToPNGConverter()

    # 根据数据量决定是否使用并行处理
    dataset_slices = converter.convert_batch(use_parallel=True)

    end_time = time.time()
    processing_time = end_time - start_time

    total_slices = sum(len(slices) for slices in dataset_slices.values())
    print(f"\n数据预处理完成!")
    print(f"共转换 {total_slices} 个PNG切片文件对")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"平均每个文件: {processing_time / total_slices:.2f} 秒" if total_slices > 0 else "N/A")


if __name__ == "__main__":
    main()