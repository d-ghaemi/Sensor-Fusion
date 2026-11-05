"""
Dump a Lightning run's tfevents scalars to CSV and a quick PNG plot,
without needing TensorBoard's web UI (useful on a remote/headless server).

Usage:
    pip install tensorboard matplotlib   # if not already present
    python plot_losses.py outputs/det/CRN_r18_256x704_128x128_4key_front_only/lightning_logs/version_0

Writes:
    <logdir>/scalars.csv   -- every logged scalar, one row per step
    <logdir>/losses.png    -- train/detection, train/heatmap, train/bbox,
                              train/depth, val/detection, val/heatmap,
                              val/bbox, val/depth (whichever are present)
"""
import sys
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if len(sys.argv) != 2:
    raise SystemExit("Usage: python plot_losses.py <path-to-version_N-dir>")

logdir = Path(sys.argv[1])
if not logdir.exists():
    raise SystemExit(f"❌ {logdir} does not exist")

ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
ea.Reload()

tags = ea.Tags().get("scalars", [])
if not tags:
    raise SystemExit(f"❌ No scalar data found in {logdir}")

print(f"Found {len(tags)} scalar tags: {tags}")

# ---- CSV dump ----
csv_path = logdir / "scalars.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "step", "wall_time", "value"])
    for tag in tags:
        for event in ea.Scalars(tag):
            writer.writerow([tag, event.step, event.wall_time, event.value])
print(f"✅ Wrote {csv_path}")

# ---- Quick plot ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interesting = [t for t in tags if t.startswith("train/") or t.startswith("val/")]
    if interesting:
        fig, axes = plt.subplots(len(interesting), 1, figsize=(8, 3 * len(interesting)), sharex=False)
        if len(interesting) == 1:
            axes = [axes]
        for ax, tag in zip(axes, interesting):
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values)
            ax.set_title(tag)
            ax.set_xlabel("step")
        plt.tight_layout()
        png_path = logdir / "losses.png"
        plt.savefig(png_path, dpi=100)
        print(f"✅ Wrote {png_path}")
    else:
        print("No train/ or val/ tags found to plot.")
except ImportError:
    print("matplotlib not installed -- skipping plot (CSV was still written). Run: pip install matplotlib")
