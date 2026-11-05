# Evaluation Results (Sensor Ablation Study)

This directory preserves the evaluation metrics for the **Front-Only 3C + 1R (ResNet-18)** ablation model evaluated on the nuScenes `val` split.

### Overall Benchmark Metrics
Evaluated on full nuScenes validation split (with sensor inputs constrained to the 3 front cameras and perspective-view radar):

| Metric | Score | Description |
|:-------|:-----:|:------------|
| **mAP** | **12.80%** | Mean Average Precision (over 0.5m, 1.0m, 2.0m, 4.0m) |
| **NDS** | **30.03%** | NuScenes Detection Score |
| **mATE** | **0.723 m** | Mean Average Translation Error |
| **mASE** | **0.306** | Mean Average Scale Error |
| **mAOE** | **0.860 rad** | Mean Average Orientation Error |
| **mAVE** | **0.566 m/s** | Mean Average Velocity Error |
| **mAAE** | **0.183** | Mean Average Attribute Error |

### Class-wise Average Precision (AP)
| Class | AP |
|:------|:---:|
| **Car** | 22.33% |
| **Barrier** | 18.33% |
| **Bus** | 18.11% |
| **Traffic Cone** | 16.87% |
| **Pedestrian** | 14.57% |
| **Motorcycle** | 12.71% |
| **Bicycle** | 10.08% |
| **Truck** | 9.25% |
| **Trailer** | 4.48% |
| **Construction Vehicle** | 1.30% |

*(Note: NuScenes official benchmark metrics evaluate across the complete 360° field around the vehicle. Because this ablation model intentionally operates with forward-only sensors, ground truth objects behind the vehicle cannot be detected, establishing the expected 360° coverage upper-bound of ~50% before factoring in frontal performance.)*
