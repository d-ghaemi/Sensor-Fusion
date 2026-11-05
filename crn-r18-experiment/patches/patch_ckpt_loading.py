"""
Patch exps/base_cli.py so that a PLAIN training run (no -e/-p flag)
loads `--ckpt_path` as WEIGHTS ONLY, instead of asking PyTorch
Lightning to fully resume trainer state (epoch count, optimizer,
LR schedule) from it.

Why: trainer.fit(model, ckpt_path=...) expects a full Lightning
checkpoint dict (state_dict + epoch + global_step + optimizer_states +
lr_schedulers). CRN_r50_256x704_128x128_4key.pth is the originally
released pretrained checkpoint, which may not have that structure --
in the best case it silently resumes from whatever epoch/step is baked
into that file (possibly already >= your requested --max_epochs,
meaning training does ~nothing), in the worst case it errors.

After this patch: weights are loaded into the model (strict=False, so
it won't crash on partial key mismatches) and then trainer.fit(model)
starts a completely fresh run: epoch 0, fresh optimizer, fresh LR
schedule -- exactly what you want when fine-tuning into a different
sensor configuration.

Run from the CRN repo root:

    python patch_ckpt_loading.py
"""
from pathlib import Path

target = Path("exps/base_cli.py")
if not target.exists():
    raise SystemExit(f"❌ Could not find {target} — run this from the CRN repo root.")

code = target.read_text()

if "Loading pretrained weights from" in code:
    raise SystemExit("❌ File already patched. Doing nothing.")

# ------------------------------------------------------------------
# 1. Make sure torch is imported
# ------------------------------------------------------------------
import_anchor = "import pytorch_lightning as pl\n"
if import_anchor not in code:
    raise SystemExit("❌ Could not find import anchor. Aborting without changes.")
if "import torch\n" not in code:
    code = code.replace(import_anchor, "import torch\n" + import_anchor, 1)

# ------------------------------------------------------------------
# 2. Replace the full-resume call with a weights-only load + fresh fit
# ------------------------------------------------------------------
train_anchor = (
    "    else:\n"
    "        if args.ckpt_path:\n"
    "            trainer.fit(model, ckpt_path=args.ckpt_path)\n"
    "        else:\n"
    "            trainer.fit(model)\n"
)
train_replacement = (
    "    else:\n"
    "        if args.ckpt_path:\n"
    "            print(f'Loading pretrained weights from {args.ckpt_path} '\n"
    "                  f'(weights only -- fresh optimizer/LR schedule/epoch count)')\n"
    "            state = torch.load(args.ckpt_path, map_location='cpu')\n"
    "            state_dict = state.get('state_dict', state) if isinstance(state, dict) else state\n"
    "            missing, unexpected = model.load_state_dict(state_dict, strict=False)\n"
    "            print(f'  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')\n"
    "        trainer.fit(model)\n"
)
if train_anchor not in code:
    raise SystemExit("❌ Could not find the plain-training else-branch anchor. Aborting without changes.")
code = code.replace(train_anchor, train_replacement, 1)

target.write_text(code)

print("✅ Patched exps/base_cli.py")
print("   - `--ckpt_path` on a plain training run now loads weights only (strict=False)")
print("   - training always starts fresh: epoch 0, new optimizer, new LR schedule")
print("   - -e/-p (evaluate/predict) behavior is unchanged")
