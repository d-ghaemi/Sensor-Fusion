from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion


# ============================================================
# USER SETTINGS
# ============================================================

DATAROOT = Path(r"D:\nuscene")
VERSION = "v1.0-mini"

# scene-0061
SCENE_INDEX = 0

# اولین فریم
FRAME_INDEX = 0

RADAR_CHANNEL = "RADAR_FRONT"
CAMERA_CHANNEL = "CAM_FRONT"

PROJECT_ROOT = Path(__file__).resolve().parent

YOLO_LABEL = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "radar_fusion_0061"
    / "labels"
    / "00_scene-0061_frame_000.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "radar_fusion_outputs"
)

# وزن YOLOPv2 مورد استفاده، وسایل نقلیه را با ID=3 ذخیره کرده است.
CLASS_NAMES = {
    3: "vehicle"
}

MIN_DEPTH = 1.0

# حاشیه اطراف باکس YOLO برای ارتباط با نقاط راداری
BOX_MARGIN_PX = 5

# اگر فاصله دو نقطه متوالی بیشتر از این مقدار باشد،
# آن‌ها در دو خوشه جدا قرار می‌گیرند.
RANGE_CLUSTER_GAP_M = 3.0

# حداقل تعداد نقاط برای تشکیل خوشه معتبر
MIN_CLUSTER_POINTS = 2

# محدوده نمایش BEV
BEV_X_LIMIT = (-35.0, 35.0)
BEV_Y_LIMIT = (0.0, 90.0)


# ============================================================
# TRANSFORMATION MATRIX
# ============================================================

def transform_matrix(rotation, translation, inverse=False):

    matrix = np.eye(
        4,
        dtype=np.float64
    )

    rotation_matrix = Quaternion(
        rotation
    ).rotation_matrix

    translation = np.asarray(
        translation,
        dtype=np.float64
    )

    if inverse:

        matrix[:3, :3] = (
            rotation_matrix.T
        )

        matrix[:3, 3] = (
            -rotation_matrix.T
            @ translation
        )

    else:

        matrix[:3, :3] = (
            rotation_matrix
        )

        matrix[:3, 3] = (
            translation
        )

    return matrix


# ============================================================
# READ YOLO LABELS
# ============================================================

def read_yolo_labels(
    label_path,
    image_width,
    image_height
):

    detections = []

    if not label_path.exists():

        raise FileNotFoundError(
            "\nYOLO label file was not found:\n"
            f"{label_path}\n\n"
            "ابتدا YOLOPv2 را با --save-txt "
            "و --save-conf اجرا کن."
        )

    with label_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) not in (5, 6):

                raise ValueError(
                    f"Invalid YOLO row at line "
                    f"{line_number}:\n{line}"
                )

            class_id = int(
                float(parts[0])
            )

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            if len(parts) == 6:
                confidence = float(parts[5])
            else:
                confidence = np.nan

            x1 = int(
                round(
                    (
                        x_center
                        - box_width / 2.0
                    )
                    * image_width
                )
            )

            y1 = int(
                round(
                    (
                        y_center
                        - box_height / 2.0
                    )
                    * image_height
                )
            )

            x2 = int(
                round(
                    (
                        x_center
                        + box_width / 2.0
                    )
                    * image_width
                )
            )

            y2 = int(
                round(
                    (
                        y_center
                        + box_height / 2.0
                    )
                    * image_height
                )
            )

            x1 = int(
                np.clip(
                    x1,
                    0,
                    image_width - 1
                )
            )

            y1 = int(
                np.clip(
                    y1,
                    0,
                    image_height - 1
                )
            )

            x2 = int(
                np.clip(
                    x2,
                    0,
                    image_width - 1
                )
            )

            y2 = int(
                np.clip(
                    y2,
                    0,
                    image_height - 1
                )
            )

            class_name = CLASS_NAMES.get(
                class_id,
                f"class_{class_id}"
            )

            detections.append(
                {
                    "id": len(detections) + 1,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }
            )

    return detections


# ============================================================
# SELECT SCENE AND FRAME
# ============================================================

def get_scene_sample(
    nusc,
    scene_index,
    frame_index
):

    if not 0 <= scene_index < len(nusc.scene):

        raise IndexError(
            f"SCENE_INDEX must be between "
            f"0 and {len(nusc.scene) - 1}."
        )

    scene = nusc.scene[
        scene_index
    ]

    sample = nusc.get(
        "sample",
        scene["first_sample_token"]
    )

    for _ in range(frame_index):

        if not sample["next"]:

            raise IndexError(
                f"FRAME_INDEX={frame_index} "
                f"is not available in "
                f"{scene['name']}."
            )

        sample = nusc.get(
            "sample",
            sample["next"]
        )

    return scene, sample


# ============================================================
# LOAD AND PROJECT RADAR
# ============================================================

def load_and_project_radar(
    nusc,
    sample
):

    radar_sample_data = nusc.get(
        "sample_data",
        sample["data"][RADAR_CHANNEL]
    )

    camera_sample_data = nusc.get(
        "sample_data",
        sample["data"][CAMERA_CHANNEL]
    )

    radar_path = (
        Path(nusc.dataroot)
        / radar_sample_data["filename"]
    )

    camera_path = (
        Path(nusc.dataroot)
        / camera_sample_data["filename"]
    )

    radar_point_cloud = (
        RadarPointCloud.from_file(
            str(radar_path)
        )
    )

    original_points = (
        radar_point_cloud.points.copy()
    )

    raw_point_count = (
        original_points.shape[1]
    )

    radar_calibration = nusc.get(
        "calibrated_sensor",
        radar_sample_data[
            "calibrated_sensor_token"
        ]
    )

    radar_pose = nusc.get(
        "ego_pose",
        radar_sample_data[
            "ego_pose_token"
        ]
    )

    camera_calibration = nusc.get(
        "calibrated_sensor",
        camera_sample_data[
            "calibrated_sensor_token"
        ]
    )

    camera_pose = nusc.get(
        "ego_pose",
        camera_sample_data[
            "ego_pose_token"
        ]
    )

    radar_to_ego = transform_matrix(
        radar_calibration["rotation"],
        radar_calibration["translation"]
    )

    radar_ego_to_global = (
        transform_matrix(
            radar_pose["rotation"],
            radar_pose["translation"]
        )
    )

    global_to_camera_ego = (
        transform_matrix(
            camera_pose["rotation"],
            camera_pose["translation"],
            inverse=True
        )
    )

    camera_ego_to_camera = (
        transform_matrix(
            camera_calibration["rotation"],
            camera_calibration[
                "translation"
            ],
            inverse=True
        )
    )

    radar_to_camera = (
        camera_ego_to_camera
        @ global_to_camera_ego
        @ radar_ego_to_global
        @ radar_to_ego
    )

    homogeneous_points = np.vstack(
        (
            original_points[:3, :],
            np.ones(
                (
                    1,
                    raw_point_count
                )
            )
        )
    )

    camera_points = (
        radar_to_camera
        @ homogeneous_points
    )[:3, :]

    depths = camera_points[
        2,
        :
    ]

    camera_intrinsic = np.asarray(
        camera_calibration[
            "camera_intrinsic"
        ],
        dtype=np.float64
    )

    pixel_homogeneous = (
        camera_intrinsic
        @ camera_points
    )

    pixels = (
        pixel_homogeneous[:2, :]
        / np.maximum(
            pixel_homogeneous[
                2:3,
                :
            ],
            1e-12
        )
    )

    image = cv2.imread(
        str(camera_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"Camera image could not "
            f"be opened:\n{camera_path}"
        )

    image_height, image_width = (
        image.shape[:2]
    )

    valid = (
        (depths > MIN_DEPTH)
        & (pixels[0, :] >= 0)
        & (
            pixels[0, :]
            < image_width
        )
        & (pixels[1, :] >= 0)
        & (
            pixels[1, :]
            < image_height
        )
    )

    pixels = pixels[:, valid]
    depths = depths[valid]
    points = original_points[:, valid]

    # در دستگاه مختصات رادار:
    # x = روبه‌جلو
    # y = چپ
    forward = points[0, :]
    lateral = -points[1, :]

    ranges = np.hypot(
        points[0, :],
        points[1, :]
    )

    # سرعت خام و نسبی رادار
    vx = points[6, :]
    vy = points[7, :]

    safe_range = np.maximum(
        ranges,
        1e-6
    )

    radial_velocity = (
        vx * points[0, :]
        + vy * points[1, :]
    ) / safe_range

    return {
        "image": image,
        "camera_path": camera_path,
        "radar_path": radar_path,
        "raw_count": raw_point_count,
        "pixels": pixels,
        "depths": depths,
        "points": points,
        "forward": forward,
        "lateral": lateral,
        "ranges": ranges,
        "radial_velocity": radial_velocity
    }


# ============================================================
# RANGE CLUSTERING
# ============================================================

def cluster_radar_points_by_range(
    candidate_indices,
    ranges,
    gap_threshold
):

    if len(candidate_indices) == 0:
        return []

    sorted_indices = (
        candidate_indices[
            np.argsort(
                ranges[
                    candidate_indices
                ]
            )
        ]
    )

    sorted_ranges = ranges[
        sorted_indices
    ]

    clusters = []

    current_cluster = [
        int(sorted_indices[0])
    ]

    for i in range(
        1,
        len(sorted_indices)
    ):

        range_difference = (
            sorted_ranges[i]
            - sorted_ranges[i - 1]
        )

        if (
            range_difference
            <= gap_threshold
        ):

            current_cluster.append(
                int(sorted_indices[i])
            )

        else:

            clusters.append(
                np.asarray(
                    current_cluster,
                    dtype=int
                )
            )

            current_cluster = [
                int(sorted_indices[i])
            ]

    clusters.append(
        np.asarray(
            current_cluster,
            dtype=int
        )
    )

    return clusters


# ============================================================
# YOLO-RADAR ASSOCIATION
# ============================================================

def associate_detections(
    detections,
    radar
):

    pixel_x = radar["pixels"][0, :]
    pixel_y = radar["pixels"][1, :]

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        inside_box = (
            (
                pixel_x
                >= x1 - BOX_MARGIN_PX
            )
            & (
                pixel_x
                <= x2 + BOX_MARGIN_PX
            )
            & (
                pixel_y
                >= y1 - BOX_MARGIN_PX
            )
            & (
                pixel_y
                <= y2 + BOX_MARGIN_PX
            )
        )

        candidate_indices = (
            np.flatnonzero(
                inside_box
            )
        )

        detection[
            "candidate_indices"
        ] = candidate_indices

        detection[
            "candidate_count"
        ] = len(candidate_indices)

        clusters = (
            cluster_radar_points_by_range(
                candidate_indices,
                radar["ranges"],
                RANGE_CLUSTER_GAP_M
            )
        )

        detection[
            "number_of_clusters"
        ] = len(clusters)

        valid_clusters = [
            cluster
            for cluster in clusters
            if (
                len(cluster)
                >= MIN_CLUSTER_POINTS
            )
        ]

        if len(valid_clusters) > 0:

            cluster_median_ranges = [
                float(
                    np.median(
                        radar["ranges"][
                            cluster
                        ]
                    )
                )
                for cluster
                in valid_clusters
            ]

            nearest_cluster_number = int(
                np.argmin(
                    cluster_median_ranges
                )
            )

            selected_indices = (
                valid_clusters[
                    nearest_cluster_number
                ]
            )

            detection[
                "cluster_status"
            ] = "valid_cluster"

        elif len(candidate_indices) > 0:

            nearest_position = int(
                np.argmin(
                    radar["ranges"][
                        candidate_indices
                    ]
                )
            )

            selected_indices = np.asarray(
                [
                    candidate_indices[
                        nearest_position
                    ]
                ],
                dtype=int
            )

            detection[
                "cluster_status"
            ] = "single_point"

        else:

            selected_indices = np.asarray(
                [],
                dtype=int
            )

            detection[
                "cluster_status"
            ] = "no_point"

        detection[
            "radar_indices"
        ] = selected_indices

        detection[
            "point_count"
        ] = len(selected_indices)

        detection[
            "removed_point_count"
        ] = (
            len(candidate_indices)
            - len(selected_indices)
        )

        if len(selected_indices) > 0:

            selected_ranges = (
                radar["ranges"][
                    selected_indices
                ]
            )

            selected_velocities = (
                radar[
                    "radial_velocity"
                ][selected_indices]
            )

            selected_lateral = (
                radar["lateral"][
                    selected_indices
                ]
            )

            selected_forward = (
                radar["forward"][
                    selected_indices
                ]
            )

            detection["range"] = float(
                np.median(
                    selected_ranges
                )
            )

            detection["velocity"] = float(
                np.median(
                    selected_velocities
                )
            )

            detection["lateral"] = float(
                np.median(
                    selected_lateral
                )
            )

            detection["forward"] = float(
                np.median(
                    selected_forward
                )
            )

            detection[
                "minimum_range"
            ] = float(
                np.min(
                    selected_ranges
                )
            )

            detection[
                "maximum_range"
            ] = float(
                np.max(
                    selected_ranges
                )
            )

            detection[
                "range_spread"
            ] = float(
                np.ptp(
                    selected_ranges
                )
            )

        else:

            detection["range"] = np.nan
            detection["velocity"] = np.nan
            detection["lateral"] = np.nan
            detection["forward"] = np.nan

            detection[
                "minimum_range"
            ] = np.nan

            detection[
                "maximum_range"
            ] = np.nan

            detection[
                "range_spread"
            ] = np.nan

    return detections


# ============================================================
# ASSOCIATION QUALITY
# ============================================================

def association_quality(
    point_count
):

    if point_count == 0:
        return "none"

    if point_count <= 2:
        return "low"

    if point_count <= 5:
        return "medium"

    return "strong"


# ============================================================
# DRAW CAMERA PANEL
# ============================================================

def draw_camera_panel(
    ax,
    radar,
    detections
):

    image_rgb = cv2.cvtColor(
        radar["image"],
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(
        image_rgb
    )

    number_of_points = (
        radar["pixels"].shape[1]
    )

    candidate_mask = np.zeros(
        number_of_points,
        dtype=bool
    )

    associated_mask = np.zeros(
        number_of_points,
        dtype=bool
    )

    for detection in detections:

        candidate_mask[
            detection[
                "candidate_indices"
            ]
        ] = True

        associated_mask[
            detection[
                "radar_indices"
            ]
        ] = True

    removed_mask = (
        candidate_mask
        & ~associated_mask
    )

    unrelated_mask = (
        ~candidate_mask
    )

    pixel_x = radar["pixels"][0, :]
    pixel_y = radar["pixels"][1, :]

    # نقاط خارج همه باکس‌ها
    ax.scatter(
        pixel_x[unrelated_mask],
        pixel_y[unrelated_mask],
        s=18,
        c="white",
        edgecolors="black",
        linewidths=0.4,
        label="Unassociated radar points",
        zorder=3
    )

    # نقاط حذف‌شده به عنوان پس‌زمینه
    ax.scatter(
        pixel_x[removed_mask],
        pixel_y[removed_mask],
        s=24,
        c="gray",
        marker="x",
        linewidths=0.9,
        label="Removed background points",
        zorder=4
    )

    # نقاط خوشه نهایی جسم
    ax.scatter(
        pixel_x[associated_mask],
        pixel_y[associated_mask],
        s=25,
        c="red",
        edgecolors="black",
        linewidths=0.4,
        label="Selected object points",
        zorder=5
    )

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        point_count = (
            detection["point_count"]
        )

        rectangle = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="yellow",
            linewidth=2.0,
            zorder=6
        )

        ax.add_patch(
            rectangle
        )

        if point_count > 0:

            quality = association_quality(
                point_count
            )

            label = (
                f"ID{detection['id']} "
                f"{detection['class_name']} "
                f"R={detection['range']:.1f}m "
                f"Vr={detection['velocity']:.1f}m/s "
                f"N={point_count} "
                f"({quality})"
            )

        else:

            label = (
                f"ID{detection['id']} "
                f"{detection['class_name']} "
                f"R=N/A Vr=N/A"
            )

        ax.text(
            x1,
            max(
                12,
                y1 - 5
            ),
            label,
            color="black",
            fontsize=6.5,
            fontweight="bold",
            bbox={
                "facecolor": "yellow",
                "edgecolor": "black",
                "pad": 1.2
            },
            zorder=7
        )

    ax.set_title(
        "Camera view: YOLOPv2 + clustered radar points",
        fontsize=14
    )

    ax.axis("off")


# ============================================================
# DRAW BEV PANEL
# ============================================================

def draw_bev_panel(
    ax,
    radar,
    detections
):

    scatter = ax.scatter(
        radar["lateral"],
        radar["forward"],
        c=np.clip(
            radar["radial_velocity"],
            -15,
            15
        ),
        cmap="coolwarm",
        vmin=-15,
        vmax=15,
        s=42,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.9
    )

    ax.scatter(
        0,
        0,
        marker="^",
        s=280,
        c="black",
        label="Ego vehicle",
        zorder=8
    )

    for detection in detections:

        indices = (
            detection["radar_indices"]
        )

        if len(indices) == 0:
            continue

        lateral_points = (
            radar["lateral"][
                indices
            ]
        )

        forward_points = (
            radar["forward"][
                indices
            ]
        )

        if len(indices) == 1:

            box_width = 2.0
            box_length = 4.0

            box_left = (
                lateral_points[0]
                - box_width / 2.0
            )

            box_bottom = (
                forward_points[0]
                - box_length / 2.0
            )

        else:

            box_left = float(
                np.min(
                    lateral_points
                )
                - 0.75
            )

            box_bottom = float(
                np.min(
                    forward_points
                )
                - 1.0
            )

            box_width = max(
                1.5,
                float(
                    np.ptp(
                        lateral_points
                    )
                    + 1.5
                )
            )

            box_length = max(
                3.0,
                float(
                    np.ptp(
                        forward_points
                    )
                    + 2.0
                )
            )

        rectangle = Rectangle(
            (
                box_left,
                box_bottom
            ),
            box_width,
            box_length,
            fill=False,
            edgecolor="yellow",
            linewidth=2.5,
            zorder=6
        )

        ax.add_patch(
            rectangle
        )

        ax.text(
            detection["lateral"],
            detection["forward"],
            f"ID{detection['id']}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox={
                "facecolor": "yellow",
                "edgecolor": "black",
                "pad": 1.5
            },
            zorder=7
        )

    ax.set_xlim(
        BEV_X_LIMIT[0],
        BEV_X_LIMIT[1]
    )

    ax.set_ylim(
        BEV_Y_LIMIT[0],
        BEV_Y_LIMIT[1]
    )

    ax.set_xlabel(
        "Lateral position (m)\n"
        "Left  ←     →  Right"
    )

    ax.set_ylabel(
        "Forward distance (m)"
    )

    ax.set_title(
        "Radar Bird's-Eye View\n"
        "Range-clustered association regions",
        fontsize=14
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    ax.legend(
        loc="upper right"
    )

    return scatter


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "STEP 3B: YOLO + RADAR ASSOCIATION "
        "+ RANGE CLUSTERING"
    )

    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    nusc = NuScenes(
        version=VERSION,
        dataroot=str(DATAROOT),
        verbose=True
    )

    scene, sample = get_scene_sample(
        nusc,
        SCENE_INDEX,
        FRAME_INDEX
    )

    radar = load_and_project_radar(
        nusc,
        sample
    )

    image_height, image_width = (
        radar["image"].shape[:2]
    )

    detections = read_yolo_labels(
        YOLO_LABEL,
        image_width,
        image_height
    )

    detections = associate_detections(
        detections,
        radar
    )

    print(
        f"\nSelected scene: "
        f"{scene['name']}"
    )

    print(
        f"Description: "
        f"{scene['description']}"
    )

    print(
        f"Frame index: "
        f"{FRAME_INDEX}"
    )

    print(
        f"Camera image size: "
        f"{image_width} x {image_height}"
    )

    print(
        f"Number of YOLO detections: "
        f"{len(detections)}"
    )

    print(
        f"Number of raw radar points: "
        f"{radar['raw_count']}"
    )

    print(
        f"Number of projected radar points: "
        f"{radar['pixels'].shape[1]}"
    )

    print(
        "\nAssociated objects "
        "after range clustering:"
    )

    for detection in detections:

        selected_count = (
            detection["point_count"]
        )

        candidate_count = (
            detection[
                "candidate_count"
            ]
        )

        removed_count = (
            detection[
                "removed_point_count"
            ]
        )

        if selected_count > 0:

            quality = association_quality(
                selected_count
            )

            print(
                f"ID {detection['id']}: "
                f"{detection['class_name']}, "
                f"class_id="
                f"{detection['class_id']}, "
                f"candidates="
                f"{candidate_count}, "
                f"selected="
                f"{selected_count}, "
                f"removed="
                f"{removed_count}, "
                f"clusters="
                f"{detection['number_of_clusters']}, "
                f"quality={quality}, "
                f"range="
                f"{detection['range']:.2f} m, "
                f"range_spread="
                f"{detection['range_spread']:.2f} m, "
                f"relative radial velocity="
                f"{detection['velocity']:.2f} m/s"
            )

        else:

            print(
                f"ID {detection['id']}: "
                f"{detection['class_name']}, "
                f"class_id="
                f"{detection['class_id']}, "
                f"candidates=0, "
                f"selected=0, "
                f"removed=0, "
                f"clusters=0, "
                f"quality=none, "
                f"range=N/A, "
                f"velocity=N/A"
            )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(19, 11),
        gridspec_kw={
            "width_ratios": [
                1.3,
                1.0
            ]
        }
    )

    figure.suptitle(
        f"YOLOPv2 and {RADAR_CHANNEL} Fusion — "
        f"{scene['name']}, "
        f"frame {FRAME_INDEX:03d}",
        fontsize=18
    )

    draw_camera_panel(
        axes[0],
        radar,
        detections
    )

    scatter = draw_bev_panel(
        axes[1],
        radar,
        detections
    )

    colorbar = figure.colorbar(
        scatter,
        ax=axes[1],
        fraction=0.045,
        pad=0.04
    )

    colorbar.set_label(
        "Relative radial velocity (m/s)"
    )

    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.96
        )
    )

    output_name = (
        f"step3b_"
        f"{scene['name'].replace('-', '')}_"
        f"frame{FRAME_INDEX:03d}_"
        f"range_clustered_fusion.png"
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    print(
        "\nFusion result saved at:\n"
        f"{output_path}"
    )

    print(
        "\nGray crosses are removed "
        "background radar points."
    )

    print(
        "Red points are selected "
        "object radar points."
    )

    print(
        "Yellow BEV rectangles are "
        "estimated association regions, "
        "not true 3D boxes."
    )

    print(
        "\nSTEP 3B FINISHED."
    )

    plt.show()


if __name__ == "__main__":
    main()
