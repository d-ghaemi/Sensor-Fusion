"""
Fix datasets/nusc_det_dataset.py's get_gt() to use a single, fixed ego
reference frame (the LIDAR_TOP ego pose) instead of averaging over
whatever cameras happen to be configured.

Why: get_gt() previously computed the ego frame as:

    np.mean([info['cam_infos'][cam]['ego_pose'][...] for cam in cams], 0)

`cams` is whatever camera list is active (self.ida_aug_conf['cams']).
When that list changed from 6 cameras to 3 front-only cameras, this
silently shifted the computed ego origin for every training-time GT
box -- to a DIFFERENT frame than both the original 6-camera model's
training convention and evaluate_comprehensive_front_180.py's
convention (which uses the single LIDAR_TOP ego pose, confirmed via
gen_depth_gt.py's proven usage: info['lidar_infos']['LIDAR_TOP']['ego_pose']).

After this fix: GT boxes (and the front/rear cutoff) are always
expressed in the LIDAR_TOP ego frame, regardless of camera count --
consistent with evaluation, and stable if camera config changes again
in the future.

Run from the CRN repo root:

    python patch_ego_frame_fix.py

Requires patch_gt_front_only.py to have already been applied (this
patch targets a different part of the same function and is independent
of it, but expects the same file/function to exist).
"""
from pathlib import Path

target = Path("datasets/nusc_det_dataset.py")
if not target.exists():
    raise SystemExit(f"❌ Could not find {target} — run this from the CRN repo root.")

code = target.read_text()

if "lidar_infos']['LIDAR_TOP']['ego_pose']" in code:
    raise SystemExit("❌ File already patched (found LIDAR_TOP ego_pose lookup). Doing nothing.")

anchor = (
    "        ego2global_rotation = np.mean(\n"
    "            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],\n"
    "            0)\n"
    "        ego2global_translation = np.mean([\n"
    "            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams\n"
    "        ], 0)\n"
    "        trans = -np.array(ego2global_translation)\n"
    "        rot = Quaternion(ego2global_rotation).inverse\n"
)
replacement = (
    "        # Use the single LIDAR_TOP ego pose as the canonical ego frame.\n"
    "        # This matches evaluate_comprehensive_front_180.py's convention and\n"
    "        # is independent of which/how-many cameras are configured, unlike\n"
    "        # averaging over `cams` (which shifted the frame when Ncams changed\n"
    "        # from 6 to 3 for the front-only experiment).\n"
    "        lidar_ego_pose = info['lidar_infos']['LIDAR_TOP']['ego_pose']\n"
    "        trans = -np.array(lidar_ego_pose['translation'])\n"
    "        rot = Quaternion(lidar_ego_pose['rotation']).inverse\n"
)
if anchor not in code:
    raise SystemExit("❌ Could not find the ego2global averaging anchor. Aborting without changes.")
code = code.replace(anchor, replacement, 1)

target.write_text(code)

print("✅ Patched datasets/nusc_det_dataset.py")
print("   - get_gt() now uses info['lidar_infos']['LIDAR_TOP']['ego_pose'] as the ego frame")
print("   - No longer depends on which/how-many cameras are configured")
