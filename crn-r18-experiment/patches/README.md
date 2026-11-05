# Reproducibility & Patch Documentation

This directory contains the automated patch scripts used to programmatically transform the upstream [CRN codebase](https://github.com/youngskkim/CRN) into this front-only sensor ablation experiment.

> **Note:** All patches in this directory have **already been applied** to this repository. These scripts are archived here for complete research transparency and reproduction from upstream.

### Applied Patches:
1. `patch_config_and_dataloader.py`: Configures front-only 3-camera setup (`CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`), injects TF32 optimizations, and optimizes DataLoader throughput with 12 workers, pin memory, and persistent workers.
2. `patch_r18_backbone.py`: Swaps the image backbone from ResNet-50 to ResNet-18 (`depth=18`, torchvision checkpoint) and adjusts SECONDFPN neck input channels (`[64, 128, 256, 512]`).
3. `patch_dataloader_front_only.py`: Adds `self.front_only` support to `exps/base_exp.py` to propagate front hemisphere filtering down to dataset loaders.
4. `patch_gt_front_only.py`: Filters 3D ground truth bounding boxes in `datasets/nusc_det_dataset.py` during training so the loss function only supervises the front hemisphere.
5. `patch_ego_frame_fix.py`: Fixes the canonical ego vehicle frame calculation in `datasets/nusc_det_dataset.py` to strictly use `LIDAR_TOP['ego_pose']`, preventing ego origin drift when reducing camera count from 6 to 3.
6. `patch_ckpt_loading.py`: Modifies `exps/base_cli.py` to support warm-starting weights (`strict=False`) with a fresh optimizer and learning rate schedule for fine-tuning.
7. `patch_resume_flag.py`: Introduces an explicit `--resume` CLI flag to distinguish between full trainer resumption vs. transfer learning warm-starts.
8. `patch_eval_weights_only.py`: Ensures evaluation mode (`-e`) seamlessly loads checkpoints regardless of format.
