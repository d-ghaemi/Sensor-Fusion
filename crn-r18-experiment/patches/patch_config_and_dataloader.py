import re
from pathlib import Path

# 1. Read the original config
config_path = Path("exps/det/CRN_r50_256x704_128x128_4key.py")
code = config_path.read_text()

# 2. Inject TF32 optimizations at the very top (Massive speedup for RTX 30-series)
tf32_injection = "import torch\ntorch.backends.cuda.matmul.allow_tf32 = True\ntorch.backends.cudnn.allow_tf32 = True\n\n"
code = tf32_injection + code

# 3. Patch Cameras to Front-Only (3 Cameras)
code = re.sub(r"'cams':\s*\[[^\]]+\]", "'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']", code)
code = re.sub(r"'Ncams':\s*\d+", "'Ncams': 3", code)

# 4. Update the experiment name so it saves to a new folder
code = re.sub(
    r"run_cli\(CRNLightningModel,\s*'det/CRN_r50_256x704_128x128_4key'\)",
    "run_cli(CRNLightningModel, 'det/CRN_r50_256x704_128x128_4key_front_only_3090')",
    code
)

# Save the new config
new_config_path = Path("exps/det/CRN_r50_256x704_128x128_4key_front_only_3090.py")
new_config_path.write_text(code)
print(f"✅ Created optimized config: {new_config_path.name}")

# 5. Patch the DataLoader in base_exp.py for maximum 16-core throughput
base_exp_path = Path("exps/base_exp.py")
base_code = base_exp_path.read_text()

# Replace num_workers=8 (or whatever it is) with 12, and add persistent_workers & pin_memory
base_code = re.sub(r"num_workers\s*=\s*\d+", "num_workers=12, persistent_workers=True, pin_memory=True", base_code)

base_exp_path.write_text(base_code)
print("✅ Patched DataLoader: 12 workers, persistent_workers=True, pin_memory=True")
