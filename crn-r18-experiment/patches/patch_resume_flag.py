"""
Add a `--resume` flag to exps/base_cli.py so you can choose, per run:

  --ckpt_path X                (no --resume)  -> WEIGHTS-ONLY load, fresh
                                                  optimizer/LR-schedule/epoch
                                                  count (what we used to warm-
                                                  start from the original
                                                  pretrained .pth)

  --ckpt_path X --resume                      -> FULL trainer-state resume:
                                                  continues epoch count,
                                                  optimizer state, and LR
                                                  schedule from X. Use this
                                                  to keep training on top of
                                                  a checkpoint produced by a
                                                  PREVIOUS run of this same
                                                  experiment (a real Lightning
                                                  .ckpt, not the original
                                                  released .pth).

Run from the CRN repo root:

    python patch_resume_flag.py
"""
from pathlib import Path

target = Path("exps/base_cli.py")
if not target.exists():
    raise SystemExit(f"❌ Could not find {target} — run this from the CRN repo root.")

code = target.read_text()

if "--resume" in code:
    raise SystemExit("❌ File already patched (found `--resume`). Doing nothing.")

# ------------------------------------------------------------------
# 1. Add the --resume CLI flag
# ------------------------------------------------------------------
arg_anchor = "    parent_parser.add_argument('--ckpt_path', type=str)\n"
arg_replacement = (
    "    parent_parser.add_argument('--ckpt_path', type=str)\n"
    "    parent_parser.add_argument('--resume', action='store_true',\n"
    "                               help='Full trainer-state resume (epoch/optimizer/LR schedule) '\n"
    "                                    'from --ckpt_path, instead of a weights-only fresh start.')\n"
)
if arg_anchor not in code:
    raise SystemExit("❌ Could not find --ckpt_path argument anchor. Aborting without changes.")
code = code.replace(arg_anchor, arg_replacement, 1)

# ------------------------------------------------------------------
# 2. Branch on --resume in the plain-training else block
# ------------------------------------------------------------------
train_anchor = (
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
train_replacement = (
    "    else:\n"
    "        if args.ckpt_path and args.resume:\n"
    "            print(f'Resuming FULL trainer state from {args.ckpt_path} '\n"
    "                  f'(epoch count, optimizer state, and LR schedule all continue)')\n"
    "            trainer.fit(model, ckpt_path=args.ckpt_path)\n"
    "        elif args.ckpt_path:\n"
    "            print(f'Loading pretrained weights from {args.ckpt_path} '\n"
    "                  f'(weights only -- fresh optimizer/LR schedule/epoch count)')\n"
    "            state = torch.load(args.ckpt_path, map_location='cpu')\n"
    "            state_dict = state.get('state_dict', state) if isinstance(state, dict) else state\n"
    "            missing, unexpected = model.load_state_dict(state_dict, strict=False)\n"
    "            print(f'  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')\n"
    "            trainer.fit(model)\n"
    "        else:\n"
    "            trainer.fit(model)\n"
)
if train_anchor not in code:
    raise SystemExit("❌ Could not find the plain-training else-branch anchor. Aborting without changes.")
code = code.replace(train_anchor, train_replacement, 1)

target.write_text(code)

print("✅ Patched exps/base_cli.py")
print("   - Added --resume flag")
print("   - --ckpt_path alone            -> weights-only, fresh start (unchanged from before)")
print("   - --ckpt_path together with --resume -> full trainer-state resume")
