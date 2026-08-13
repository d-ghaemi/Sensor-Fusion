from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nuscenes.nuscenes import NuScenes


# ============================================================
# تنظیمات
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

NUSCENES_ROOT = Path(r"D:\nuscene")

NUSCENES_VERSION = "v1.0-mini"

SCENE_NAME = "scene-0061"

CAMERA_CHANNEL = "CAM_FRONT"

RADAR_CHANNEL = "RADAR_FRONT"


OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "radar_fusion_outputs"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# شروع برنامه
# ============================================================

print("=" * 70)

print(
    "STEP 2: PROJECT RADAR POINTS "
    "ONTO CAM_FRONT"
)

print("=" * 70)


# ============================================================
# بارگذاری nuScenes
# ============================================================

print(
    "Loading nuScenes-mini..."
)

nusc = NuScenes(
    version=NUSCENES_VERSION,
    dataroot=str(NUSCENES_ROOT),
    verbose=True,
)


# ============================================================
# پیدا کردن سناریوی انتخاب‌شده
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
    selected_scene[
        "first_sample_token"
    ],
)


if CAMERA_CHANNEL not in sample["data"]:

    raise KeyError(
        f"{CAMERA_CHANNEL} is not "
        "available in the sample."
    )


if RADAR_CHANNEL not in sample["data"]:

    raise KeyError(
        f"{RADAR_CHANNEL} is not "
        "available in the sample."
    )


camera_token = sample["data"][
    CAMERA_CHANNEL
]

radar_token = sample["data"][
    RADAR_CHANNEL
]


camera_sample_data = nusc.get(
    "sample_data",
    camera_token,
)

radar_sample_data = nusc.get(
    "sample_data",
    radar_token,
)


camera_path = (
    NUSCENES_ROOT
    / camera_sample_data["filename"]
)

radar_path = (
    NUSCENES_ROOT
    / radar_sample_data["filename"]
)


print()

print(
    "Camera file:"
)

print(camera_path)


print()

print(
    "Radar file:"
)

print(radar_path)


if not camera_path.is_file():

    raise FileNotFoundError(
        f"Camera image not found:\n"
        f"{camera_path}"
    )


if not radar_path.is_file():

    raise FileNotFoundError(
        f"Radar file not found:\n"
        f"{radar_path}"
    )


# ============================================================
# فرافکنی Point Cloud رادار روی دوربین
# ============================================================

print()

print(
    "Projecting RADAR_FRONT points "
    "onto CAM_FRONT..."
)


projected_points, depths, camera_image = (
    nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=radar_token,
        camera_token=camera_token,
        min_dist=1.0,
    )
)


number_of_projected_points = (
    projected_points.shape[1]
)


print(
    "Number of projected radar points:",
    number_of_projected_points,
)


if number_of_projected_points == 0:

    raise ValueError(
        "No radar point was projected "
        "onto the camera image."
    )


# مختصات پیکسلی نقاط
pixel_u = projected_points[0, :]

pixel_v = projected_points[1, :]


# depths فاصله نقاط در راستای محور دوربین است
depths = np.asarray(depths)


print()

print(
    "Minimum projected depth:",
    f"{np.min(depths):.2f} m",
)

print(
    "Maximum projected depth:",
    f"{np.max(depths):.2f} m",
)


# ============================================================
# رسم نقاط رادار روی تصویر
# ============================================================

figure, axis = plt.subplots(
    figsize=(16, 9)
)


axis.imshow(camera_image)


scatter = axis.scatter(
    pixel_u,
    pixel_v,
    c=depths,
    cmap="turbo_r",
    s=75,
    edgecolors="black",
    linewidths=0.7,
    alpha=0.95,
)


axis.set_xlim(
    0,
    camera_image.size[0],
)

axis.set_ylim(
    camera_image.size[1],
    0,
)


axis.set_title(
    "RADAR_FRONT projected onto CAM_FRONT\n"
    "scene-0061 — first sample",
    fontsize=16,
)


axis.axis("off")


colorbar = figure.colorbar(
    scatter,
    ax=axis,
    fraction=0.025,
    pad=0.02,
)


colorbar.set_label(
    "Radar point depth (m)",
    fontsize=12,
)


figure.tight_layout()


# ============================================================
# ذخیره تصویر
# ============================================================

output_path = (
    OUTPUT_FOLDER
    / (
        "step2_scene0061_"
        "radar_on_camera.png"
    )
)


figure.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)


print()

print(
    "Projected radar image saved at:"
)

print(output_path)


print()

print(
    "STEP 2 FINISHED."
)


plt.show()
