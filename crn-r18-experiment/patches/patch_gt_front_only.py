"""
Patch datasets/nusc_det_dataset.py to add a `front_only` option.

What this does:
  1. Adds a `front_only=False` constructor parameter to NuscDatasetRadarDet,
     stored as `self.front_only`.
  2. In get_gt(), after each box is transformed into the ego frame, skips
     it if front_only=True and the box's ego-x coordinate is <= 0 (i.e.
     it's behind the car). This uses the exact same convention as
     evaluate_comprehensive_front_180.py (ego_center[0] > 0 = front).

Run this from the CRN repo root (same place you run your other patch
scripts from):

    python patch_gt_front_only.py

It edits datasets/nusc_det_dataset.py in place and prints a summary of
what changed. It refuses to run twice (won't double-patch).
"""
import re
from pathlib import Path

target = Path("datasets/nusc_det_dataset.py")
if not target.exists():
    raise SystemExit(f"❌ Could not find {target} — run this from the CRN repo root.")

code = target.read_text()

if "self.front_only" in code:
    raise SystemExit("❌ File already patched (found `self.front_only`). Doing nothing.")

# ------------------------------------------------------------------
# 1. Add `front_only=False` to the constructor signature
# ------------------------------------------------------------------
sig_anchor = "                 gt_for_radar_only=False,\n                 sweep_idxes=list(),"
sig_replacement = "                 gt_for_radar_only=False,\n                 front_only=False,\n                 sweep_idxes=list(),"

if sig_anchor not in code:
    raise SystemExit("❌ Could not find constructor signature anchor. Aborting without changes.")
code = code.replace(sig_anchor, sig_replacement, 1)

# ------------------------------------------------------------------
# 2. Store it as self.front_only
# ------------------------------------------------------------------
init_anchor = "        self.gt_for_radar_only = gt_for_radar_only\n"
init_replacement = "        self.gt_for_radar_only = gt_for_radar_only\n        self.front_only = front_only\n"

if init_anchor not in code:
    raise SystemExit("❌ Could not find __init__ body anchor. Aborting without changes.")
code = code.replace(init_anchor, init_replacement, 1)

# ------------------------------------------------------------------
# 3. Filter rear boxes in get_gt()
# ------------------------------------------------------------------
gt_anchor = "            box_xyz = np.array(box.center)\n            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]"
gt_replacement = (
    "            box_xyz = np.array(box.center)\n"
    "            if self.front_only and box_xyz[0] <= 0:\n"
    "                continue\n"
    "            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]"
)

if gt_anchor not in code:
    raise SystemExit("❌ Could not find get_gt() anchor. Aborting without changes.")
code = code.replace(gt_anchor, gt_replacement, 1)

target.write_text(code)

print("✅ Patched datasets/nusc_det_dataset.py")
print("   - Added `front_only=False` constructor parameter")
print("   - Added `self.front_only = front_only`")
print("   - get_gt() now skips boxes with ego_x <= 0 when front_only=True")
