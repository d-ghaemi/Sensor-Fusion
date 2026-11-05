"""
Swap the image backbone from ResNet50 to ResNet18 in
CRN_r50_256x704_128x128_4key_front_only_3090.py.

This changes two things that MUST move together:
  1. img_backbone_conf: depth 50 -> 18, and the ImageNet init checkpoint
     from torchvision://resnet50 -> torchvision://resnet18.
  2. img_neck_conf: in_channels [256, 512, 1024, 2048] (ResNet50's
     Bottleneck stage outputs) -> [64, 128, 256, 512] (ResNet18's
     BasicBlock stage outputs). Without this, the neck will crash on
     a shape mismatch the moment it consumes the backbone's features.

Everything else (img_neck's out_channels, depth_net_conf, the BEV
backbone in head_conf which is already a separate ResNet-18, fuser_conf,
backbone_pts_conf) is untouched -- SECONDFPN's out_channels stays
[128,128,128,128] regardless of backbone, so depth_net_conf's
in_channels=512 (4*128) remains correct either way.

Per the CRN paper (Table 1, C+R, 256x704): R50 gets ~49.0 mAP / 56.0
NDS at 20.4 FPS; R18 gets ~44.8 mAP / 54.3 NDS at 27.9 FPS. Expect a
similar trade-off here: faster training, somewhat lower ceiling.

Run from the CRN repo root:

    python patch_r18_backbone.py
"""
from pathlib import Path

target = Path("exps/det/CRN_r50_256x704_128x128_4key_front_only_3090.py")
if not target.exists():
    raise SystemExit(f"❌ Could not find {target} — run this from the CRN repo root.")

code = target.read_text()

if "torchvision://resnet18" in code:
    raise SystemExit("❌ File already patched (found resnet18). Doing nothing.")

# ------------------------------------------------------------------
# 1. img_backbone_conf: depth + init checkpoint
# ------------------------------------------------------------------
backbone_anchor = (
    "            'img_backbone_conf': dict(\n"
    "                type='ResNet',\n"
    "                depth=50,\n"
    "                frozen_stages=0,\n"
    "                out_indices=[0, 1, 2, 3],\n"
    "                norm_eval=False,\n"
    "                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),\n"
    "            ),\n"
)
backbone_replacement = (
    "            'img_backbone_conf': dict(\n"
    "                type='ResNet',\n"
    "                depth=18,\n"
    "                frozen_stages=0,\n"
    "                out_indices=[0, 1, 2, 3],\n"
    "                norm_eval=False,\n"
    "                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),\n"
    "            ),\n"
)
if backbone_anchor not in code:
    raise SystemExit("❌ Could not find img_backbone_conf anchor. Aborting without changes.")
code = code.replace(backbone_anchor, backbone_replacement, 1)

# ------------------------------------------------------------------
# 2. img_neck_conf: in_channels must match ResNet18's stage outputs
# ------------------------------------------------------------------
neck_anchor = (
    "            'img_neck_conf': dict(\n"
    "                type='SECONDFPN',\n"
    "                in_channels=[256, 512, 1024, 2048],\n"
)
neck_replacement = (
    "            'img_neck_conf': dict(\n"
    "                type='SECONDFPN',\n"
    "                in_channels=[64, 128, 256, 512],\n"
)
if neck_anchor not in code:
    raise SystemExit("❌ Could not find img_neck_conf anchor. Aborting without changes.")
code = code.replace(neck_anchor, neck_replacement, 1)

target.write_text(code)

print("✅ Patched exps/det/CRN_r50_256x704_128x128_4key_front_only_3090.py")
print("   - img_backbone_conf: ResNet50 -> ResNet18 (depth + ImageNet init checkpoint)")
print("   - img_neck_conf.in_channels: [256,512,1024,2048] -> [64,128,256,512]")
print("   - Nothing else changed (neck out_channels, depth_net_conf, BEV backbone all unaffected)")
