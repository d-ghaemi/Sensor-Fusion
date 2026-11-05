# Evaluation & Diagnostic Tools

This directory contains utility scripts developed for training, validating, and analyzing the front-only sensor ablation experiment:

### Scripts Overview:
1. `evaluate_comprehensive_front_180.py`:
   * Evaluates 3D object detection metrics (mAP, NDS, True Positive errors) specifically within the vehicle's forward 180° field of view.
   * Usage:
     ```bash
     python tools/evaluate_comprehensive_front_180.py --res_path outputs/det/CRN_r18_256x704_128x128_4key_front_only/results_nusc.json --axis y
     ```
2. `count_rear_predictions.py`:
   * Diagnostic script that inspects 360° validation predictions to check whether the front-only trained model hallucinates or bleeds false positives into the unobserved rear region.
   * Usage:
     ```bash
     python tools/count_rear_predictions.py --res_path outputs/det/CRN_r18_256x704_128x128_4key_front_only/results_nusc.json
     ```
3. `plot_losses.py`:
   * Headless TensorBoard parser that extracts loss curves (`train/detection`, `train/depth`, `train/heatmap`, `train/bbox`, and validation metrics) into CSV and generates an aggregate PNG figure without requiring a browser or web server.
   * Usage:
     ```bash
     python tools/plot_losses.py outputs/det/CRN_r18_256x704_128x128_4key_front_only/lightning_logs/version_0
     ```
4. `filter_infos.py`:
   * Multi-threaded preprocessing tool (16 threads) that audits `nuscenes_infos_{train,val}.pkl` and strips sample tokens referencing missing front camera frames to guarantee crash-free training pipelines.
   * Usage:
     ```bash
     python tools/filter_infos.py
     ```
