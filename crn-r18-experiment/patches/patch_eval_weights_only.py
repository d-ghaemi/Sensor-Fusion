"""
Make -e (evaluate) mode use the same proven weights-only loader as
training, instead of PyTorch Lightning's built-in trainer.test(ckpt_path=...).

Why: we already confirmed trainer.fit(ckpt_path=...) can silently fail
to load the original released .pth pretrained checkpoint (it expects a
full Lightning checkpoint dict). trainer.test(ckpt_path=...) may have
the same assumption -- and we've never actually tested -e against that
raw .pth file, only against genuine Lightning .ckpt files (which worked
fine either way). This patch removes the guesswork: it always uses the
same load_state_dict(strict=False) approach that gave us a confirmed
"missing keys: 0, unexpected keys: 0" result earlier, for both file
types.

Run from the CRN repo root:

    python patch_eval_weights_only.py
"""
from pathlib import Path

target = Path("exps/base_cli.py")
if not target.exists():
    raise SystemExit(f"❌ Could not find {target} — run this from the CRN repo root.")

code = target.read_text()

if "Loading weights for evaluation from" in code:
    raise SystemExit("❌ File already patched. Doing nothing.")

anchor = (
    "    if args.evaluate:\n"
    "        trainer.test(model, ckpt_path=args.ckpt_path)\n"
)
replacement = (
    "    if args.evaluate:\n"
    "        if args.ckpt_path:\n"
    "            print(f'Loading weights for evaluation from {args.ckpt_path}')\n"
    "            state = torch.load(args.ckpt_path, map_location='cpu')\n"
    "            state_dict = state.get('state_dict', state) if isinstance(state, dict) else state\n"
    "            missing, unexpected = model.load_state_dict(state_dict, strict=False)\n"
    "            print(f'  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')\n"
    "        trainer.test(model)\n"
)
if anchor not in code:
    raise SystemExit("❌ Could not find the evaluate-branch anchor. Aborting without changes.")
code = code.replace(anchor, replacement, 1)

target.write_text(code)

print("✅ Patched exps/base_cli.py")
print("   - -e (evaluate) now loads --ckpt_path via the same weights-only loader as training")
print("   - Works identically for both raw .pth files and genuine Lightning .ckpt files")
