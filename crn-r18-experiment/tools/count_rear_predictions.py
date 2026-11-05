import argparse
import json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from pathlib import Path

parser = argparse.ArgumentParser(description="Analyze predictions outside the front FoV")
parser.add_argument('--res_path', type=str, default=None,
                    help='Path to results_nusc.json')
parser.add_argument('--data_root', type=str, default='data/nuScenes',
                    help='Path to nuScenes dataset')
args = parser.parse_args()

# Candidate paths
candidate_paths = [
    args.res_path,
    'outputs/det/CRN_r18_256x704_128x128_4key_front_only/results_nusc.json',
    'outputs/det/CRN_r50_256x704_128x128_4key_front_only_3090/results_nusc.json',
]
res_path = None
for p in candidate_paths:
    if p and Path(p).exists():
        res_path = Path(p)
        break

if res_path is None:
    print("❌ Cannot find results_nusc.json. Please run evaluation (-e) first or specify --res_path.")
    exit(1)

print(f"Loading predictions from: {res_path}")
nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_root, verbose=False)
with open(res_path, 'r') as f:
    results_data = json.load(f)
predictions = results_data.get('results', {})

total_predictions = 0
rear_y_predictions = 0   # nuScenes vehicle frame: Y <= 0 (longitudinal rear)
rear_x_predictions = 0   # LiDAR frame convention: X <= 0
rear_by_class = {}

print("Analyzing prediction locations...")
for sample_token, pred_list in predictions.items():
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    ego_trans = np.array(ego_pose['translation'])
    ego_rot = Quaternion(ego_pose['rotation']).inverse

    for p in pred_list:
        total_predictions += 1
        global_center = np.array(p['translation'])
        ego_center = ego_rot.rotate(global_center - ego_trans)
        
        # In nuScenes vehicle frame: +Y is Forward, +X is Right, +Z is Up
        if ego_center[1] <= 0:
            rear_y_predictions += 1
            cls = p['detection_name']
            rear_by_class[cls] = rear_by_class.get(cls, 0) + 1
            
        if ego_center[0] <= 0:
            rear_x_predictions += 1

print("\n" + "="*65)
print("📊 PREDICTION DISTRIBUTION ANALYSIS (Front-Only Model)")
print("="*65)
print(f"Total Predictions (360° Val Set)          : {total_predictions}")
print(f"Rear Predictions (nuScenes Y <= 0)         : {rear_y_predictions} ({100 * rear_y_predictions / max(1, total_predictions):.2f}%)")
print(f"Lateral Negative Predictions (X <= 0)     : {rear_x_predictions} ({100 * rear_x_predictions / max(1, total_predictions):.2f}%)")
print("-" * 65)
print("Rear Predictions Breakdown by Class (Y <= 0):")

if not rear_by_class:
    print("  🎉 ZERO rear predictions! The model is perfectly constrained to the front.")
else:
    for cls, count in sorted(rear_by_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<22}: {count}")
print("="*65)
