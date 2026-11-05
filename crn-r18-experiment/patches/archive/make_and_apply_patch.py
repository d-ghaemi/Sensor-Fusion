import difflib
import os

target_file = "exps/det/CRN_r50_256x704_128x128_4key_front_only_3090.py"
patch_file_name = "front_only_fix.patch"

if not os.path.exists(target_file):
    raise FileNotFoundError(f"Could not find {target_file}. Make sure you are running from the clone root.")

with open(target_file, "r", encoding="utf-8") as f:
    original_lines = f.readlines()

modified_lines = []
for line in original_lines:
    # 1. Update image backbone x_bound
    if "'x_bound': [-51.2, 51.2, 0.8]" in line:
        line = line.replace("[-51.2, 51.2, 0.8]", "[0.0, 51.2, 0.8]")
    
    # 2. Update bbox_coder post_center_range and pc_range
    elif "post_center_range=[-61.2," in line:
        line = line.replace("[-61.2, -61.2", "[0.0, -61.2")
    elif "pc_range=[-51.2," in line:
        line = line.replace("[-51.2, -51.2", "[0.0, -51.2")
        
    # 3. Update train_cfg point_cloud_range
    elif "point_cloud_range=[-51.2," in line:
        line = line.replace("[-51.2, -51.2", "[0.0, -51.2")
        
    # 4. Update test_cfg post_center_limit_range
    elif "post_center_limit_range=[-61.2," in line:
        line = line.replace("[-61.2, -61.2", "[0.0, -61.2")
        
    modified_lines.append(line)

# Generate unified diff
diff = list(difflib.unified_diff(
    original_lines,
    modified_lines,
    fromfile=f"a/{target_file}",
    tofile=f"b/{target_file}",
    n=3
))

# Write out the patch file
with open(patch_file_name, "w", encoding="utf-8") as f:
    f.writelines(diff)
print(f"[+] Successfully generated patch file: {patch_file_name}")

# Write out the modified configuration file
with open(target_file, "w", encoding="utf-8") as f:
    f.writelines(modified_lines)
print(f"[+] Successfully applied changes directly to: {target_file}")
