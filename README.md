# Uncertainty-Guided Alignment Feature Fusion for Robust Breast Tumor Segmentation in Multi-Sequence MRI

## 📄 Overview

Accurate breast tumor segmentation in multi-sequence MRI is challenged by irregular morphology, indistinct boundaries, small lesion size, and high modality uncertainty. We propose an **uncertainty-guided dual‑stream feature fusion network** that integrates:

- A **ConvNeXt–Mamba hybrid encoder** for joint local detail extraction and efficient global context modeling.
- An **Uncertainty‑Guided Alignment Fusion (UGAF) module** with deformable spatial alignment, pixel‑wise uncertainty estimation, and gated interaction enhancement.
- A composite loss function combining Dice‑BCE, Focal Tversky, and alignment regularization.

**Key Results on a private multi‑sequence breast MRI dataset (296 patients):**

| Metric        | Value         |
|---------------|---------------|
| mDice         | 78.55% ± 2.40 |
| mIoU          | 68.62% ± 1.70 |
| Sensitivity   | 79.66% ± 2.90 |
| Precision     | 82.04% ± 3.10 |
| HD95 (↓)      | 12.20 ± 2.17  |
| ASSD (↓)      | 6.93 ± 1.34   |

---

## 🧠 Method Highlights

<p align="center">
  <img src="assets/architecture.png" alt="Network Architecture" width="800"/>
</p>

1. **ConvNeXt Local Branch** – extracts multi‑scale local features with large kernels and depthwise separable convolutions.  
2. **Mamba Global Branch** – models long‑range dependencies with linear complexity via Residual Vision Mamba (RVM) blocks.  
3. **UGAF Module** – aligns and fuses dual‑stream features adaptively using:
   - Deformable flow‑based spatial alignment  
   - Pixel‑wise uncertainty estimation for confidence‑weighted fusion  
   - Gated bidirectional residual injection  
4. **Composite Loss** – Dice‑BCE + Focal Tversky + alignment loss (L1 + smoothness) for robust training.

---

## 📁 Repository Structure

```
Uncertain-breast-lesion-segmentation/
├── datasets/               # Dataset loading and preprocessing utilities
├── models/                 # Model definition (encoder, decoder, UGAF)
├── nets/                   # Network components (ConvNeXt, Mamba blocks, etc.)
├── logs/                   # Training logs and checkpoints (created automatically)
├── config.py               # Configuration file (hyperparameters, paths)
├── data_prepare.py         # Data splitting and preparation
├── data_process.py         # Augmentation, normalization, and preprocessing
├── train.py                # Training script
├── val.py                  # Validation script
├── test.py                 # Testing and inference script
├── main.py                 # Main entry point for training/evaluation
├── main.sh                 # Shell script to launch training
├── utils.py                # Helper functions (metrics, visualization, etc.)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/si-yuan20/Uncertain-breast-lesion-segmentation.git
cd Uncertain-breast-lesion-segmentation
```

### 2. Create a virtual environment (recommended)
```bash
conda create -n breast_mri python=3.9 -y
conda activate breast_mri
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

*Key dependencies:* `torch>=1.12`, `torchvision`, `numpy`, `opencv-python`, `scikit-learn`, `tqdm`, `tensorboard`, `SimpleITK`, `monai`.

---

## 🚀 Usage

### Data Preparation

Organize your multi‑sequence MRI dataset as follows:
```
/path/to/dataset/
├── images/
│   ├── patient_001_T2.nii.gz
│   ├── patient_001_C2.nii.gz
│   ├── patient_001_C5.nii.gz
│   └── ...
└── masks/
    ├── patient_001.nii.gz
    └── ...
```

Update dataset paths in `config.py`.

### Training

Run the training pipeline with default settings:
```bash
python main.py --mode train --config config.py
```

Or use the provided shell script:
```bash
bash main.sh
```

**Training hyperparameters (configurable in `config.py`):**
- Optimizer: AdamW (`lr=1e-3`, weight decay `3e-5`)
- Batch size: 8
- Epochs: 200 (early stopping patience = 30)
- Image size: 256×256
- Loss weights: Dice‑BCE + Focal Tversky + Alignment loss (λ=0.1)

### Validation & Testing

```bash
# Validate on the validation split
python val.py --checkpoint /path/to/best_model.pth

# Test on the test set and generate metrics
python test.py --checkpoint /path/to/best_model.pth --save_results
```
---

## 📈 Experimental Results (Highlights)

| Method        | mDice (%) | mIoU (%) | Sensitivity (%) | Precision (%) | HD95 ↓ | ASSD ↓ |
|---------------|-----------|----------|-----------------|---------------|--------|--------|
| UNet          | 70.71     | 61.63    | 73.14           | 74.69         | 13.15  | 7.26   |
| Swin‑UNet     | 67.38     | 56.50    | 69.23           | 71.10         | 15.21  | 7.65   |
| LightM‑UNet   | 73.82     | 63.17    | 72.79           | 77.83         | 11.96  | 6.51   |
| **Ours**      | **78.55** | **68.62**| **79.66**       | **82.04**     | **12.20** | **6.93** |

*Full comparison tables and ablation studies are available in the paper.*

---


## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhao2026uncertainty,
  title={Uncertainty-Guided Alignment Feature Fusion for Robust Breast Tumor Segmentation in Multi-Sequence MRI},
  author={Zhao, Sichao and Feng, Kanghua and Chen, Junjun and Su, Luowei and Li, Minghao and Zheng, Kehong and Lai, Wanting and Gao, Rong and Li, Weidong and Liu, Ying and Qiu, Xuejun},
  journal={/},
  year={2026},
}
```

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
