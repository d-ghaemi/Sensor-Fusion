from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


# ============================================================
# تنظیمات
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

NUSCENES_ROOT = Path(r"D:\nuscene")

NUSCENES_VERSION = "v1.0-mini"

SCENE_NAME = "scene-0061"

RADAR_CHANNEL = "RADAR_FRONT"

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "radar_fusion_outputs"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# محدوده نمایش BEV بر حسب متر
MAX_FORWARD_DISTANCE = 80.0

MAX_SIDE_DISTANCE = 30.0


# ============================================================
# بارگذاری nuScenes
# ============================================================

print("=" * 65)

print("STEP 1: RADAR BIRD'S-EYE VIEW")

print("=" * 65)

print("Loading nuScenes-mini...")

nusc = NuScenes(
    version=NUSCENES_VERSION,
    dataroot=str(NUSCENES_ROOT),
    verbose=True,
)


# ============================================================
# پیدا کردن scene-0061
# ============================================================

selected_scene = None

for scene in nusc.scene:

    if scene["name"] == SCENE_NAME:

        selected_scene = scene

        break


if selected_scene is None:

    raise ValueError(
        f"Scene not found: {SCENE_NAME}"
    )


print()

print(
    "Selected scene:",
    selected_scene["name"],
)

print(
    "Description:",
    selected_scene["description"],
)

print(
    "Number of samples:",
    selected_scene["nbr_samples"],
)


# ============================================================
# دریافت اولین Sample
# ============================================================

sample = nusc.get(
    "sample",
    selected_scene["first_sample_token"],
)


if RADAR_CHANNEL not in sample["data"]:

    raise KeyError(
        f"{RADAR_CHANNEL} is not available "
        "in this sample."
    )


radar_token = sample["data"][RADAR_CHANNEL]


radar_sample_data = nusc.get(
    "sample_data",
    radar_token,
)


radar_file_path = (
    NUSCENES_ROOT
    / radar_sample_data["filename"]
)


print()

print(
    "Radar channel:",
    RADAR_CHANNEL,
)

print(
    "Radar file:",
    radar_file_path,
)


if not radar_file_path.is_file():

    raise FileNotFoundError(
        f"Radar file not found:\n"
        f"{radar_file_path}"
    )


# ============================================================
# خواندن Point Cloud رادار
# ============================================================

radar_pc = RadarPointCloud.from_file(
    str(radar_file_path)
)


number_of_raw_points = (
    radar_pc.points.shape[1]
)


print()

print(
    "Number of raw radar points:",
    number_of_raw_points,
)


if number_of_raw_points == 0:

    raise ValueError(
        "The radar point cloud is empty."
    )


# ============================================================
# استخراج موقعیت و سرعت در مختصات رادار
# ============================================================

# موقعیت نقاط در دستگاه مختصات رادار
radar_x = radar_pc.points[0, :].copy()

radar_y = radar_pc.points[1, :].copy()


# سرعت جبران‌شده رادار
velocity_x_comp = (
    radar_pc.points[8, :].copy()
)

velocity_y_comp = (
    radar_pc.points[9, :].copy()
)


# سرعت شعاعی جبران‌شده
raw_range = np.sqrt(
    radar_x ** 2
    + radar_y ** 2
)


safe_range = np.maximum(
    raw_range,
    1e-6,
)


radial_velocity = (
    radar_x * velocity_x_comp
    + radar_y * velocity_y_comp
) / safe_range


# ============================================================
# تبدیل مختصات رادار به مختصات Ego Vehicle
# ============================================================

calibrated_sensor = nusc.get(
    "calibrated_sensor",
    radar_sample_data[
        "calibrated_sensor_token"
    ],
)


rotation_matrix = Quaternion(
    calibrated_sensor["rotation"]
).rotation_matrix


translation_vector = np.array(
    calibrated_sensor["translation"]
)


# موقعیت اولیه نقاط رادار
radar_positions = (
    radar_pc.points[0:3, :].copy()
)


# انتقال از مختصات رادار به مختصات خودرو
ego_positions = (
    rotation_matrix @ radar_positions
    + translation_vector.reshape(3, 1)
)


# در دستگاه مختصات خودرو
x_forward = ego_positions[0, :]

y_left = ego_positions[1, :]

z_up = ego_positions[2, :]


# فاصله افقی نسبت به خودروی حامل حسگر
planar_range = np.sqrt(
    x_forward ** 2
    + y_left ** 2
)


# زاویه Azimuth نسبت به جهت جلو
azimuth_degree = np.degrees(
    np.arctan2(
        y_left,
        x_forward,
    )
)


# ============================================================
# فیلتر محدوده نمایش
# ============================================================

valid_mask = (
    (x_forward > 0.0)
    & (
        x_forward
        <= MAX_FORWARD_DISTANCE
    )
    & (
        np.abs(y_left)
        <= MAX_SIDE_DISTANCE
    )
    & np.isfinite(radial_velocity)
)


x_forward_valid = (
    x_forward[valid_mask]
)

y_left_valid = (
    y_left[valid_mask]
)

range_valid = (
    planar_range[valid_mask]
)

azimuth_valid = (
    azimuth_degree[valid_mask]
)

velocity_valid = (
    radial_velocity[valid_mask]
)


number_of_valid_points = (
    x_forward_valid.size
)


print(
    "Number of displayed points:",
    number_of_valid_points,
)


if number_of_valid_points == 0:

    raise ValueError(
        "No valid radar point remained "
        "after filtering."
    )


# ============================================================
# قرارداد نمایش BEV
# ============================================================

# محور افقی:
# مثبت به سمت راست خودرو
bev_horizontal = -y_left_valid

# محور عمودی:
# مثبت به سمت جلوی خودرو
bev_forward = x_forward_valid


# ============================================================
# آمار نقاط
# ============================================================

print()

print(
    "Minimum range:",
    f"{np.min(range_valid):.2f} m",
)

print(
    "Maximum range:",
    f"{np.max(range_valid):.2f} m",
)

print(
    "Minimum radial velocity:",
    f"{np.min(velocity_valid):.2f} m/s",
)

print(
    "Maximum radial velocity:",
    f"{np.max(velocity_valid):.2f} m/s",
)

print(
    "Mean radial velocity:",
    f"{np.mean(velocity_valid):.2f} m/s",
)


# ============================================================
# رسم Bird's-Eye View
# ============================================================

figure, axis = plt.subplots(
    figsize=(10, 12)
)


scatter = axis.scatter(
    bev_horizontal,
    bev_forward,
    c=velocity_valid,
    cmap="coolwarm",
    vmin=-15,
    vmax=15,
    s=70,
    edgecolors="black",
    linewidths=0.4,
    alpha=0.9,
)


# نمایش محل رادار/خودرو
axis.scatter(
    0,
    0,
    marker="^",
    s=280,
    color="black",
    label="Ego vehicle / radar",
    zorder=5,
)


# رسم خطوط زاویه دید تقریبی
field_of_view_degree = 60.0

field_of_view_radian = np.radians(
    field_of_view_degree
)


for sign in [-1, 1]:

    side_position = (
        sign
        * MAX_FORWARD_DISTANCE
        * np.tan(
            field_of_view_radian / 2
        )
    )

    axis.plot(
        [0, side_position],
        [0, MAX_FORWARD_DISTANCE],
        linestyle="--",
        linewidth=1.5,
        color="gray",
        alpha=0.7,
    )


# خط مرکزی خودرو
axis.axvline(
    0,
    color="gray",
    linestyle=":",
    linewidth=1,
)


axis.set_xlim(
    -MAX_SIDE_DISTANCE,
    MAX_SIDE_DISTANCE,
)

axis.set_ylim(
    0,
    MAX_FORWARD_DISTANCE,
)


axis.set_aspect(
    "equal",
    adjustable="box",
)


axis.set_xlabel(
    "Lateral position (m)\n"
    "Left ←     → Right"
)

axis.set_ylabel(
    "Forward distance (m)"
)

axis.set_title(
    "nuScenes RADAR_FRONT Bird's-Eye View\n"
    "scene-0061 — first sample"
)

axis.grid(
    True,
    linestyle="--",
    alpha=0.35,
)

axis.legend(
    loc="upper right"
)


colorbar = figure.colorbar(
    scatter,
    ax=axis,
    pad=0.02,
)

colorbar.set_label(
    "Compensated radial velocity (m/s)"
)


# توضیح قرارداد رنگ
axis.text(
    0.02,
    0.02,
    "Negative velocity: approaching\n"
    "Positive velocity: receding",
    transform=axis.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.8,
    },
)


figure.tight_layout()


# ============================================================
# ذخیره خروجی
# ============================================================

output_path = (
    OUTPUT_FOLDER
    / "step1_scene0061_radar_bev.png"
)


figure.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)


print()

print(
    "BEV image saved at:"
)

print(output_path)


print()

print(
    "STEP 1 FINISHED."
)


plt.show()
