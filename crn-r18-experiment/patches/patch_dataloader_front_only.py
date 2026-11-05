"""
Patch exps/base_exp.py to add a `self.front_only` flag (default False)
and pass it into the train/val NuscDatasetRadarDet instances.

Run from the CRN repo root:

    python patch_dataloader_front_only.py

Won't run twice (refuses if already patched).
"""
from pathlib import Path

target = Path("exps/base_exp.py")
if not target.exists():
    raise SystemExit(f"❌ Could not find {target} — run this from the CRN repo root.")

code = target.read_text()

if "self.front_only" in code:
    raise SystemExit("❌ File already patched (found `self.front_only`). Doing nothing.")

# ------------------------------------------------------------------
# 1. Default flag in __init__ (base class default: no filtering)
# ------------------------------------------------------------------
init_anchor = "        self.remove_z_axis = True\n"
init_replacement = "        self.remove_z_axis = True\n        self.front_only = False\n"

if init_anchor not in code:
    raise SystemExit("❌ Could not find __init__ anchor (`self.remove_z_axis = True`). Aborting without changes.")
code = code.replace(init_anchor, init_replacement, 1)

# ------------------------------------------------------------------
# 2. train_dataloader(): pass front_only into NuscDatasetRadarDet
# ------------------------------------------------------------------
train_anchor = (
    "            remove_z_axis=self.remove_z_axis,\n"
    "            depth_path='depth_gt',\n"
    "            radar_pv_path='radar_pv_filter'\n"
    "        )"
)
train_replacement = (
    "            remove_z_axis=self.remove_z_axis,\n"
    "            front_only=self.front_only,\n"
    "            depth_path='depth_gt',\n"
    "            radar_pv_path='radar_pv_filter'\n"
    "        )"
)
if train_anchor not in code:
    raise SystemExit("❌ Could not find train_dataloader() anchor. Aborting without changes.")
code = code.replace(train_anchor, train_replacement, 1)

# ------------------------------------------------------------------
# 3. val_dataloader(): pass front_only into NuscDatasetRadarDet
# ------------------------------------------------------------------
val_anchor = (
    "            remove_z_axis=self.remove_z_axis,\n"
    "            radar_pv_path='radar_pv_filter',\n"
    "        )"
)
val_replacement = (
    "            remove_z_axis=self.remove_z_axis,\n"
    "            front_only=self.front_only,\n"
    "            radar_pv_path='radar_pv_filter',\n"
    "        )"
)
if val_anchor not in code:
    raise SystemExit("❌ Could not find val_dataloader() anchor. Aborting without changes.")
code = code.replace(val_anchor, val_replacement, 1)

target.write_text(code)

print("✅ Patched exps/base_exp.py")
print("   - Added `self.front_only = False` default")
print("   - train_dataloader() now passes front_only=self.front_only")
print("   - val_dataloader() now passes front_only=self.front_only")
