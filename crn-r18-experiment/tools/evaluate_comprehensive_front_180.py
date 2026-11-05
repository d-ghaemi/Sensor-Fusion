import argparse
import json
import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
import math
from pathlib import Path

# 1. Parse arguments and automatically find the JSON
parser = argparse.ArgumentParser(description="Evaluate Front 180° mAP/NDS")
parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the .ckpt file')
parser.add_argument('--res_path', type=str, default=None, help='Direct path to results_nusc.json')
parser.add_argument('--data_root', type=str, default='data/nuScenes', help='Path to nuScenes dataset')
parser.add_argument('--axis', type=str, default='y', choices=['x', 'y'],
                    help="Forward axis in ego frame: 'y' for nuScenes vehicle frame (default), 'x' for LiDAR frame")
args = parser.parse_args()

if args.res_path:
    res_path = Path(args.res_path).resolve()
elif args.ckpt_path:
    ckpt_path = Path(args.ckpt_path).resolve()
    # Navigate up the directory tree: checkpoints -> version_X -> lightning_logs -> exp_name
    exp_dir = ckpt_path.parents[3]
    res_path = exp_dir / "results_nusc.json"
else:
    # Try default experiment path
    default_candidates = [
        Path('outputs/det/CRN_r18_256x704_128x128_4key_front_only/results_nusc.json'),
        Path('outputs/det/CRN_r50_256x704_128x128_4key_front_only_3090/results_nusc.json'),
    ]
    res_path = next((p for p in default_candidates if p.exists()), None)

if res_path is None or not res_path.exists():
    print(f"❌ Cannot find predictions file.")
    print("Please specify --res_path <path-to-results_nusc.json> or --ckpt_path <path-to-ckpt>")
    exit(1)

print(f"✅ Found predictions at: {res_path}")
data_root = args.data_root

# 2. Load Data
print("1. Loading NuScenes v1.0-trainval and predictions...")
nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)
with open(res_path, 'r') as f:
    results_data = json.load(f)
predictions = results_data.get('results', {})

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
DIST_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]

metrics = {cls: {
    'gt_count': 0,
    'pr_curves': {t: [] for t in DIST_THRESHOLDS},
    'errors': {'ate': [], 'ase': [], 'aoe': [], 'ave': [], 'aae': []}
} for cls in CLASSES}

def get_yaw(quat): return quat.yaw_pitch_roll[0]
def calc_aoe(yaw1, yaw2):
    diff = abs(yaw1 - yaw2)
    return min(diff, 2 * math.pi - diff)
def calc_ase(pred_size, gt_size):
    vol_pred = pred_size[0] * pred_size[1] * pred_size[2]
    vol_gt = gt_size[0] * gt_size[1] * gt_size[2]
    inter = min(pred_size[0], gt_size[0]) * min(pred_size[1], gt_size[1]) * min(pred_size[2], gt_size[2])
    union = vol_pred + vol_gt - inter
    return 1 - (inter / union) if union > 0 else 1.0

# 3. Evaluate Front 180°
axis_idx = 1 if args.axis == 'y' else 0
axis_name = 'Y > 0 (nuScenes vehicle front)' if args.axis == 'y' else 'X > 0 (LiDAR frame front)'
print(f"2. Evaluating Front 180° ({axis_name})...")
for sample_token, pred_list in predictions.items():
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_trans = np.array(ego_pose['translation'])
    ego_rot = Quaternion(ego_pose['rotation']).inverse

    filtered_preds = []
    for p in pred_list:
        global_center = np.array(p['translation'])
        ego_center = ego_rot.rotate(global_center - ego_trans)
        if ego_center[axis_idx] > 0:
            filtered_preds.append({
                'center': ego_center, 'class': p['detection_name'], 'score': p['detection_score'],
                'size': p['size'], 'rotation': Quaternion(p['rotation']),
                'velocity': p.get('velocity', [0.0, 0.0]), 'attribute': p.get('attribute_name', '')
            })

    gt_boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        gt_center = np.array(ann['translation'])
        gt_ego_center = ego_rot.rotate(gt_center - ego_trans)
        if gt_ego_center[axis_idx] > 0:
            cat_name = ann['category_name']
            gt_class = None
            if 'human.pedestrian' in cat_name: gt_class = 'pedestrian'
            elif 'vehicle.car' in cat_name: gt_class = 'car'
            elif 'vehicle.truck' in cat_name: gt_class = 'truck'
            elif 'vehicle.bus' in cat_name: gt_class = 'bus'
            elif 'vehicle.trailer' in cat_name: gt_class = 'trailer'
            elif 'vehicle.construction' in cat_name: gt_class = 'construction_vehicle'
            elif 'vehicle.bicycle' in cat_name: gt_class = 'bicycle'
            elif 'vehicle.motorcycle' in cat_name: gt_class = 'motorcycle'
            elif 'movable_object.barrier' in cat_name: gt_class = 'barrier'
            elif 'movable_object.trafficcone' in cat_name: gt_class = 'traffic_cone'
            
            if gt_class in CLASSES:
                gt_boxes.append({
                    'center': gt_ego_center, 'class': gt_class, 'size': ann['size'],
                    'rotation': Quaternion(ann['rotation']), 'velocity': ann.get('velocity', [0.0, 0.0]),
                    'attribute': ann.get('attribute_name', '')
                })
                metrics[gt_class]['gt_count'] += 1

    for thresh in DIST_THRESHOLDS:
        for cls in CLASSES:
            cls_preds = sorted([p for p in filtered_preds if p['class'] == cls], key=lambda x: x['score'], reverse=True)
            cls_gts = [gt for gt in gt_boxes if gt['class'] == cls]
            gt_matched = [False] * len(cls_gts)
            
            for pred in cls_preds:
                best_dist = float('inf')
                best_gt_idx = -1
                for i, gt in enumerate(cls_gts):
                    if not gt_matched[i]:
                        dist = np.linalg.norm(pred['center'] - gt['center'])
                        if dist < best_dist:
                            best_dist = dist
                            best_gt_idx = i
                
                if best_gt_idx != -1 and best_dist <= thresh:
                    metrics[cls]['pr_curves'][thresh].append((pred['score'], True))
                    gt_matched[best_gt_idx] = True
                    if thresh == 2.0:
                        gt = cls_gts[best_gt_idx]
                        metrics[cls]['errors']['ate'].append(best_dist)
                        metrics[cls]['errors']['aoe'].append(calc_aoe(get_yaw(pred['rotation']), get_yaw(gt['rotation'])))
                        metrics[cls]['errors']['ase'].append(calc_ase(pred['size'], gt['size']))
                        if cls in ['barrier', 'traffic_cone']: metrics[cls]['errors']['ave'].append(0.0)
                        else:
                            v_err = np.linalg.norm(np.array(pred['velocity']) - np.array(gt['velocity']))
                            metrics[cls]['errors']['ave'].append(v_err)
                        aae = 0.0 if pred['attribute'] == gt['attribute'] else 1.0
                        metrics[cls]['errors']['aae'].append(aae)
                else:
                    metrics[cls]['pr_curves'][thresh].append((pred['score'], False))

# 4. Calculate and Print Results
print("\n" + "="*105)
print(f"{'Class':<22} | {'mAP':<6} | {'ATE':<6} | {'ASE':<6} | {'AOE':<6} | {'AVE':<6} | {'AAE':<6} | {'NDS':<6}")
print("-" * 105)

total_map = 0.0
total_nds = 0.0
valid_classes = 0

for cls in CLASSES:
    m = metrics[cls]
    gt_count = m['gt_count']
    if gt_count == 0:
        print(f"{cls:<22} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6}")
        continue
    
    valid_classes += 1
    aps = []
    for thresh in DIST_THRESHOLDS:
        pr_curve = m['pr_curves'][thresh]
        pr_curve.sort(key=lambda x: x[0], reverse=True)
        tp_accum = 0; fp_accum = 0; precisions = []; recalls = []
        for score, is_tp in pr_curve:
            if is_tp: tp_accum += 1
            else: fp_accum += 1
            recall = tp_accum / gt_count
            precision = tp_accum / (tp_accum + fp_accum) if (tp_accum + fp_accum) > 0 else 0.0
            recalls.append(recall); precisions.append(precision)
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])
        
        ap = 0.0
        for r in np.linspace(0, 1, 101):
            valid_precisions = [p for rec, p in zip(recalls, precisions) if rec >= r]
            if valid_precisions: ap += max(valid_precisions)
        ap /= 101.0
        aps.append(ap)
    
    map_cls = np.mean(aps)
    total_map += map_cls
    
    def mean_err(err_list): return np.mean(err_list) if len(err_list) > 0 else 0.0
    ate = mean_err(m['errors']['ate']); ase = mean_err(m['errors']['ase'])
    aoe = mean_err(m['errors']['aoe']); ave = mean_err(m['errors']['ave']); aae = mean_err(m['errors']['aae'])
    
    norm_ate = min(1.0, ate / 2.0); norm_ase = min(1.0, ase)
    norm_aoe = min(1.0, aoe / np.pi); norm_ave = min(1.0, ave / 2.0); norm_aae = min(1.0, aae)
    mean_err_score = (1 - norm_ate + 1 - norm_ase + 1 - norm_aoe + 1 - norm_ave + 1 - norm_aae) / 5.0
    nds = 0.5 * map_cls + 0.5 * mean_err_score
    total_nds += nds
    
    print(f"{cls:<22} | {map_cls:<6.4f} | {ate:<6.3f} | {ase:<6.3f} | {aoe:<6.3f} | {ave:<6.3f} | {aae:<6.3f} | {nds:<6.4f}")

print("-" * 105)
final_map = total_map / valid_classes if valid_classes > 0 else 0.0
final_nds = total_nds / valid_classes if valid_classes > 0 else 0.0
print(f"{'OVERALL (Front 180°)':<22} | {final_map:<6.4f} | {'':<6} | {'':<6} | {'':<6} | {'':<6} | {'':<6} | {final_nds:<6.4f}")
print("=" * 105)
