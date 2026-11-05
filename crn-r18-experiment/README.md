# CRN Front-Only: Camera-Radar Fusion Sensor Ablation (ResNet-18)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9.1-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An empirical sensor ablation study and lightweight forward-perception adaptation of **CRN (Camera Radar Net)** for autonomous driving and advanced driver assistance systems (ADAS).

This repository investigates the efficiency-accuracy trade-offs of restricting full $360^\circ$ surround multi-camera sensor suites down to a **front-facing 3-camera setup** (`CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`) combined with perspective-view **radar point cloud fusion**, backed by a lightweight **ResNet-18** image encoder.

---

## 📌 Key Architectural & Engineering Highlights

1. **Sensor Ablation (3C + Radar):**
   * Drops the 3 rear cameras (`CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`), focusing compute on the frontal $180^\circ$ operational design domain (ODD).
   * Transforms sparse radar sweeps into perspective-view depth and feature maps mapped directly onto the active forward cameras.
   * Fuses perspective camera features and radar representations in Bird's-Eye-View (BEV) via Multimodal Feature Aggregation (MFA) deformable cross-attention.

2. **Lightweight Image Backbone (ResNet-18):**
   * Replaces the standard ResNet-50 image backbone with a torchvision-pretrained **ResNet-18** (`depth=18`).
   * Reconfigures the `SECONDFPN` neck input channels (`[64, 128, 256, 512]`) to match BasicBlock stage outputs while maintaining standard 512-channel BEV depth projections.

3. **Canonical Ego Reference Frame Stabilization:**
   * Resolves origin drift: original CRN computed the vehicle reference frame by averaging extrinsic poses across all configured cameras (`cams`). Reducing cameras from 6 to 3 front cameras artificially biased the ego origin forward.
   * Anchors ground-truth translations and evaluation strictly to `LIDAR_TOP['ego_pose']`.

4. **Front-Hemisphere Ground Truth Filtering:**
   * Implements a configurable `front_only` supervision flag in `NuscDatasetRadarDet`.
   * Automatically filters out rear ground-truth bounding boxes during training so the forward-only network is not penalized for unobserved rear areas.

5. **Hardware & Training Throughput Optimization:**
   * **TF32 Precision:** Injected `allow_tf32 = True` for Ampere/Ada Lovelace architectures (e.g. RTX 3090), achieving significant speedups in matrix multiplications.
   * **DataLoader Tuning:** Tuned DataLoader with 12 workers, `persistent_workers=True`, and `pin_memory=True`.
   * **Transfer Learning CLI:** Patched `exps/base_cli.py` to allow `--ckpt_path` to load weights only (`strict=False`) with a fresh optimizer and learning rate schedule, with an explicit `--resume` flag for full trainer-state resumption.

---

## 📊 Experimental Results

Evaluated on the **nuScenes validation split**:

### 360° Validation Benchmark Metrics
*Note: Evaluated across the full 360° nuScenes validation set. Ground-truth objects in the unobserved rear region naturally establish an expected coverage upper bound of ~50%.*

| Metric | Score | Description |
|:-------|:-----:|:------------|
| **NDS** | **30.03%** | NuScenes Detection Score |
| **mAP** | **12.80%** | Mean Average Precision across distance thresholds |
| **mATE** | **0.723 m** | Mean Average Translation Error |
| **mASE** | **0.306** | Mean Average Scale Error |
| **mAOE** | **0.860 rad** | Mean Average Orientation Error |
| **mAVE** | **0.566 m/s** | Mean Average Velocity Error |
| **mAAE** | **0.183** | Mean Average Attribute Error |

Detailed class-wise APs and error breakdowns are preserved under [`results/`](results/).

---

## 🛠️ Repository Structure

```text
crn-r18-experiment/
├── exps/
│   ├── base_cli.py                     # CLI launcher with flexible weights-only / resume loading
│   ├── base_exp.py                     # Lightning module with front_only DataLoader support
│   └── det/
│       ├── CRN_r18_256x704_128x128_4key_front_only.py  # Active 3C+Radar R18 ablation config
│       ├── CRN_r18_256x704_128x128_4key.py             # 360° R18 baseline config
│       └── CRN_r50_256x704_128x128_4key.py             # 360° R50 baseline config
├── datasets/
│   └── nusc_det_dataset.py             # nuScenes dataset with front-only box filtering & canonical ego frame
├── layers/                             # RVT-LSS-FPN, Radar voxelization, MFA fuser, BEVDepth head
├── models/                             # CameraRadarNetDet architecture
├── ops/                                # CUDA C++ voxel pooling extensions (setup via setup.py develop)
├── scripts/                            # Data generation (depth GT, radar BEV, radar PV)
├── tools/                              # Evaluation & diagnostic utilities
│   ├── count_rear_predictions.py       # Checks rear prediction leakage
│   ├── evaluate_comprehensive_front_180.py # Dedicated 180° front evaluation
│   ├── filter_infos.py                 # Multithreaded sample verification tool
│   └── plot_losses.py                  # Headless loss curve plotter
├── patches/                            # Documented patch history from upstream CRN
├── results/                            # Benchmark metrics & JSON summaries
├── run_training.py                     # Single-command GPU training launcher
├── setup.py
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/MahdiRoshani/crn-r18-front-ablation.git
cd crn-r18-front-ablation

# Create and activate conda environment
conda env create --file CRN.yaml
conda activate CRN

# Install PyTorch and core dependencies
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.6.0
mim install mmcv==1.6.0
mim install mmsegmentation==0.28.0
mim install mmdet==2.25.2

# Install mmdetection3d
cd mmdetection3d
pip install -v -e .
cd ..

# Build CUDA voxel pooling extensions
python setup.py develop
```

### 2. Data Preparation

Follow standard nuScenes preparation instructions:
1. Symlink nuScenes dataset to `./data/nuScenes/`.
2. Generate annotations and depth ground-truth:
   ```bash
   python scripts/gen_info.py
   python scripts/gen_depth_gt.py
   ```
3. Generate radar sweeps in BEV and project to camera perspective view:
   ```bash
   python scripts/gen_radar_bev.py
   python scripts/gen_radar_pv.py
   ```
4. Verify dataset integrity:
   ```bash
   python tools/filter_infos.py
   ```

### 3. Training

Launch fine-tuning using the single-command training script:
```bash
python run_training.py
```

Or via direct configuration invocation:
```bash
python exps/det/CRN_r18_256x704_128x128_4key_front_only.py \
    --batch_size_per_device 4 \
    --max_epochs 24 \
    --precision 16 \
    --accelerator gpu \
    --devices 1
```

To warm-start from pre-trained weights (fresh optimizer & schedule):
```bash
python exps/det/CRN_r18_256x704_128x128_4key_front_only.py \
    --ckpt_path checkpoints/CRN_r18_pretrained.pth \
    --batch_size_per_device 4 \
    --max_epochs 24 \
    --precision 16 \
    --accelerator gpu \
    --devices 1
```

To resume full trainer state from an intermediate Lightning checkpoint:
```bash
python exps/det/CRN_r18_256x704_128x128_4key_front_only.py \
    --ckpt_path outputs/det/CRN_r18_256x704_128x128_4key_front_only/lightning_logs/version_0/checkpoints/last.ckpt \
    --resume \
    --batch_size_per_device 4 \
    --precision 16 \
    --accelerator gpu \
    --devices 1
```

### 4. Evaluation & Analysis

Generate standard nuScenes validation predictions:
```bash
python exps/det/CRN_r18_256x704_128x128_4key_front_only.py \
    -e \
    --ckpt_path <path-to-ckpt> \
    -b 4 \
    --gpus 1
```

Evaluate Front $180^\circ$ performance:
```bash
python tools/evaluate_comprehensive_front_180.py \
    --res_path outputs/det/CRN_r18_256x704_128x128_4key_front_only/results_nusc.json \
    --axis y
```

Inspect prediction leakage in rear hemisphere:
```bash
python tools/count_rear_predictions.py \
    --res_path outputs/det/CRN_r18_256x704_128x128_4key_front_only/results_nusc.json
```

Plot training and validation loss curves headlessly:
```bash
python tools/plot_losses.py outputs/det/CRN_r18_256x704_128x128_4key_front_only/lightning_logs/version_0
```

---

## 📜 Acknowledgements & Citation

This codebase builds upon:
* [CRN (ICCV 2023)](https://github.com/youngskkim/CRN)
* [BEVDepth (AAAI 2023)](https://github.com/Megvii-BaseDetection/BEVDepth)
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

```bibtex
@inproceedings{kim2023crn,
    title={Crn: Camera radar net for accurate, robust, efficient 3d perception},
    author={Kim, Youngseok and Shin, Juyeb and Kim, Sanmin and Lee, In-Jae and Choi, Jun Won and Kum, Dongsuk},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={17615--17626},
    year={2023}
}
```
