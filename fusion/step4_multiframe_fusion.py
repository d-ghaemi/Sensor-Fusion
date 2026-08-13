from pathlib import Path
import csv

import cv2
import numpy as np

import step3_yolo_radar_fusion_corrected as step3


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SCENE_INDEX = 0

LABELS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "radar_fusion_0061"
    / "labels"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "radar_fusion_outputs"
    / "scene0061_multiframe"
)

CSV_PATH = (
    OUTPUT_DIR
    / "scene0061_radar_fusion_results.csv"
)

VIDEO_PATH = (
    OUTPUT_DIR
    / "scene0061_radar_fusion_video.mp4"
)

# اندازه بخش دوربین در ویدئو
CAMERA_PANEL_WIDTH = 1280
CAMERA_PANEL_HEIGHT = 720

# اندازه بخش BEV
BEV_PANEL_WIDTH = 640
BEV_PANEL_HEIGHT = 720

VIDEO_WIDTH = (
    CAMERA_PANEL_WIDTH
    + BEV_PANEL_WIDTH
)

VIDEO_HEIGHT = 720

# نرخ فریم ویدئوی خروجی
VIDEO_FPS = 5.0

# محدوده BEV
BEV_LATERAL_MIN = -35.0
BEV_LATERAL_MAX = 35.0
BEV_FORWARD_MIN = 0.0
BEV_FORWARD_MAX = 90.0

# اگر True باشد، تصویر هر فریم نیز جداگانه ذخیره می‌شود.
SAVE_INDIVIDUAL_FRAMES = True


# ============================================================
# COLOR OF RADAR VELOCITY
# ============================================================

def velocity_to_color(velocity):
    """
    سرعت منفی: آبی
    نزدیک صفر: سفید
    سرعت مثبت: قرمز
    """

    velocity = float(
        np.clip(
            velocity,
            -15.0,
            15.0
        )
    )

    if velocity < 0:

        ratio = abs(velocity) / 15.0

        blue = 255
        green = int(
            255 * (1.0 - ratio)
        )
        red = int(
            255 * (1.0 - ratio)
        )

    else:

        ratio = velocity / 15.0

        blue = int(
            255 * (1.0 - ratio)
        )
        green = int(
            255 * (1.0 - ratio)
        )
        red = 255

    # OpenCV uses BGR
    return (
        blue,
        green,
        red
    )


# ============================================================
# BEV COORDINATE CONVERSION
# ============================================================

def bev_to_pixel(
    lateral,
    forward
):
    """
    تبدیل مختصات واقعی BEV به پیکسل.
    """

    x_normalized = (
        (
            lateral
            - BEV_LATERAL_MIN
        )
        / (
            BEV_LATERAL_MAX
            - BEV_LATERAL_MIN
        )
    )

    y_normalized = (
        (
            forward
            - BEV_FORWARD_MIN
        )
        / (
            BEV_FORWARD_MAX
            - BEV_FORWARD_MIN
        )
    )

    pixel_x = int(
        x_normalized
        * (BEV_PANEL_WIDTH - 1)
    )

    # محور عمودی تصویر از بالا به پایین است.
    pixel_y = int(
        (
            1.0
            - y_normalized
        )
        * (BEV_PANEL_HEIGHT - 1)
    )

    return pixel_x, pixel_y


# ============================================================
# DRAW CAMERA PANEL
# ============================================================

def draw_camera_panel(
    radar,
    detections,
    frame_index
):

    original_image = (
        radar["image"].copy()
    )

    original_height, original_width = (
        original_image.shape[:2]
    )

    scale_x = (
        CAMERA_PANEL_WIDTH
        / original_width
    )

    scale_y = (
        CAMERA_PANEL_HEIGHT
        / original_height
    )

    image = cv2.resize(
        original_image,
        (
            CAMERA_PANEL_WIDTH,
            CAMERA_PANEL_HEIGHT
        )
    )

    number_of_points = (
        radar["pixels"].shape[1]
    )

    candidate_mask = np.zeros(
        number_of_points,
        dtype=bool
    )

    selected_mask = np.zeros(
        number_of_points,
        dtype=bool
    )

    for detection in detections:

        candidate_mask[
            detection["candidate_indices"]
        ] = True

        selected_mask[
            detection["radar_indices"]
        ] = True

    removed_mask = (
        candidate_mask
        & ~selected_mask
    )

    unrelated_mask = (
        ~candidate_mask
    )

    pixel_x = (
        radar["pixels"][0, :]
        * scale_x
    )

    pixel_y = (
        radar["pixels"][1, :]
        * scale_y
    )

    # نقاط خارج از باکس‌ها
    for index in np.flatnonzero(
        unrelated_mask
    ):

        center = (
            int(pixel_x[index]),
            int(pixel_y[index])
        )

        cv2.circle(
            image,
            center,
            3,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            image,
            center,
            3,
            (0, 0, 0),
            1
        )

    # نقاط حذف‌شده پس‌زمینه
    for index in np.flatnonzero(
        removed_mask
    ):

        center = (
            int(pixel_x[index]),
            int(pixel_y[index])
        )

        cv2.drawMarker(
            image,
            center,
            (128, 128, 128),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=7,
            thickness=1
        )

    # نقاط منتخب جسم
    for index in np.flatnonzero(
        selected_mask
    ):

        center = (
            int(pixel_x[index]),
            int(pixel_y[index])
        )

        cv2.circle(
            image,
            center,
            4,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            image,
            center,
            4,
            (0, 0, 0),
            1
        )

    # رسم باکس‌های YOLO
    for detection in detections:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        point_count = (
            detection["point_count"]
        )

        if point_count > 0:

            label = (
                f"ID{detection['id']} "
                f"vehicle "
                f"R={detection['range']:.1f}m "
                f"Vr={detection['velocity']:.1f}m/s "
                f"N={point_count}"
            )

        else:

            label = (
                f"ID{detection['id']} "
                f"vehicle "
                f"R=N/A Vr=N/A"
            )

        text_y = max(
            18,
            y1 - 5
        )

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1
        )[0]

        cv2.rectangle(
            image,
            (
                x1,
                text_y
                - text_size[1]
                - 5
            ),
            (
                min(
                    CAMERA_PANEL_WIDTH - 1,
                    x1 + text_size[0] + 4
                ),
                text_y + 3
            ),
            (0, 255, 255),
            -1
        )

        cv2.putText(
            image,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # عنوان فریم
    cv2.rectangle(
        image,
        (0, 0),
        (CAMERA_PANEL_WIDTH, 40),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        image,
        (
            f"scene-0061 | frame "
            f"{frame_index:03d} | "
            f"YOLOPv2 + RADAR_FRONT"
        ),
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


# ============================================================
# DRAW BEV PANEL
# ============================================================

def draw_bev_panel(
    radar,
    detections,
    frame_index
):

    bev = np.full(
        (
            BEV_PANEL_HEIGHT,
            BEV_PANEL_WIDTH,
            3
        ),
        245,
        dtype=np.uint8
    )

    # عنوان
    cv2.rectangle(
        bev,
        (0, 0),
        (BEV_PANEL_WIDTH, 45),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        bev,
        "Radar Bird's-Eye View",
        (145, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # خطوط شبکه افقی
    for forward in range(
        0,
        91,
        10
    ):

        x1, pixel_y = bev_to_pixel(
            BEV_LATERAL_MIN,
            forward
        )

        x2, _ = bev_to_pixel(
            BEV_LATERAL_MAX,
            forward
        )

        cv2.line(
            bev,
            (x1, pixel_y),
            (x2, pixel_y),
            (210, 210, 210),
            1
        )

        cv2.putText(
            bev,
            f"{forward}m",
            (5, max(55, pixel_y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (80, 80, 80),
            1,
            cv2.LINE_AA
        )

    # خطوط شبکه عمودی
    for lateral in range(
        -30,
        31,
        10
    ):

        pixel_x, y1 = bev_to_pixel(
            lateral,
            BEV_FORWARD_MIN
        )

        _, y2 = bev_to_pixel(
            lateral,
            BEV_FORWARD_MAX
        )

        cv2.line(
            bev,
            (pixel_x, y1),
            (pixel_x, y2),
            (210, 210, 210),
            1
        )

        cv2.putText(
            bev,
            str(lateral),
            (
                pixel_x - 10,
                BEV_PANEL_HEIGHT - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (80, 80, 80),
            1,
            cv2.LINE_AA
        )

    # محور مرکزی
    center_x, bottom_y = bev_to_pixel(
        0,
        0
    )

    _, top_y = bev_to_pixel(
        0,
        90
    )

    cv2.line(
        bev,
        (center_x, bottom_y),
        (center_x, top_y),
        (100, 100, 100),
        1
    )

    # تمام نقاط راداری
    for i in range(
        radar["points"].shape[1]
    ):

        lateral = radar["lateral"][i]
        forward = radar["forward"][i]

        if not (
            BEV_LATERAL_MIN
            <= lateral
            <= BEV_LATERAL_MAX
        ):
            continue

        if not (
            BEV_FORWARD_MIN
            <= forward
            <= BEV_FORWARD_MAX
        ):
            continue

        pixel = bev_to_pixel(
            lateral,
            forward
        )

        color = velocity_to_color(
            radar["radial_velocity"][i]
        )

        cv2.circle(
            bev,
            pixel,
            4,
            color,
            -1
        )

        cv2.circle(
            bev,
            pixel,
            4,
            (40, 40, 40),
            1
        )

    # نواحی اشیای مرتبط
    for detection in detections:

        indices = (
            detection["radar_indices"]
        )

        if len(indices) == 0:
            continue

        lateral_points = (
            radar["lateral"][indices]
        )

        forward_points = (
            radar["forward"][indices]
        )

        if len(indices) == 1:

            lateral_min = (
                lateral_points[0] - 1.0
            )

            lateral_max = (
                lateral_points[0] + 1.0
            )

            forward_min = (
                forward_points[0] - 2.0
            )

            forward_max = (
                forward_points[0] + 2.0
            )

        else:

            lateral_min = (
                np.min(lateral_points)
                - 0.75
            )

            lateral_max = (
                np.max(lateral_points)
                + 0.75
            )

            forward_min = (
                np.min(forward_points)
                - 1.0
            )

            forward_max = (
                np.max(forward_points)
                + 1.0
            )

        left, bottom = bev_to_pixel(
            lateral_min,
            forward_min
        )

        right, top = bev_to_pixel(
            lateral_max,
            forward_max
        )

        cv2.rectangle(
            bev,
            (
                min(left, right),
                min(top, bottom)
            ),
            (
                max(left, right),
                max(top, bottom)
            ),
            (0, 255, 255),
            2
        )

        center = bev_to_pixel(
            detection["lateral"],
            detection["forward"]
        )

        label = (
            f"ID{detection['id']} "
            f"R={detection['range']:.1f} "
            f"V={detection['velocity']:.1f}"
        )

        cv2.rectangle(
            bev,
            (
                center[0] - 42,
                center[1] - 19
            ),
            (
                center[0] + 70,
                center[1] + 4
            ),
            (0, 255, 255),
            -1
        )

        cv2.putText(
            bev,
            label,
            (
                center[0] - 39,
                center[1] - 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    # خودروی حامل حسگر
    ego_x, ego_y = bev_to_pixel(
        0,
        0
    )

    triangle = np.asarray(
        [
            [ego_x, ego_y - 20],
            [ego_x - 12, ego_y],
            [ego_x + 12, ego_y]
        ],
        dtype=np.int32
    )

    cv2.fillPoly(
        bev,
        [triangle],
        (0, 0, 0)
    )

    # راهنمای رنگ
    cv2.putText(
        bev,
        "Blue: approaching | Red: receding",
        (170, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (50, 50, 50),
        1,
        cv2.LINE_AA
    )

    return bev


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "STEP 4: MULTI-FRAME YOLO + RADAR FUSION"
    )
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    frames_output_dir = (
        OUTPUT_DIR
        / "frames"
    )

    if SAVE_INDIVIDUAL_FRAMES:

        frames_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    nusc = step3.NuScenes(
        version=step3.VERSION,
        dataroot=str(step3.DATAROOT),
        verbose=True
    )

    scene = nusc.scene[
        SCENE_INDEX
    ]

    print(
        f"\nSelected scene: "
        f"{scene['name']}"
    )

    print(
        f"Description: "
        f"{scene['description']}"
    )

    number_of_frames = int(
        scene["nbr_samples"]
    )

    print(
        f"Number of frames: "
        f"{number_of_frames}"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    video_writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        fourcc,
        VIDEO_FPS,
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT
        )
    )

    if not video_writer.isOpened():

        raise RuntimeError(
            "VideoWriter could not be opened."
        )

    csv_fields = [
        "frame_index",
        "sample_token",
        "object_id_in_frame",
        "class_id",
        "class_name",
        "confidence",
        "candidate_points",
        "selected_points",
        "removed_points",
        "number_of_clusters",
        "quality",
        "range_m",
        "relative_radial_velocity_mps",
        "lateral_position_m",
        "forward_position_m",
        "range_spread_m"
    ]

    total_objects = 0
    total_associated_objects = 0

    sample = nusc.get(
        "sample",
        scene["first_sample_token"]
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields
        )

        writer.writeheader()

        for frame_index in range(
            number_of_frames
        ):

            print(
                f"\nProcessing frame "
                f"{frame_index + 1}/"
                f"{number_of_frames}"
            )

            label_path = (
                LABELS_DIR
                / (
                    f"00_scene-0061_"
                    f"frame_{frame_index:03d}.txt"
                )
            )

            if not label_path.exists():

                print(
                    f"WARNING: label file missing: "
                    f"{label_path.name}"
                )

                if sample["next"]:
                    sample = nusc.get(
                        "sample",
                        sample["next"]
                    )

                continue

            radar = (
                step3.load_and_project_radar(
                    nusc,
                    sample
                )
            )

            image_height, image_width = (
                radar["image"].shape[:2]
            )

            detections = (
                step3.read_yolo_labels(
                    label_path,
                    image_width,
                    image_height
                )
            )

            detections = (
                step3.associate_detections(
                    detections,
                    radar
                )
            )

            total_objects += len(
                detections
            )

            for detection in detections:

                selected_points = (
                    detection["point_count"]
                )

                if selected_points > 0:

                    total_associated_objects += 1

                    quality = (
                        step3.association_quality(
                            selected_points
                        )
                    )

                    range_value = (
                        detection["range"]
                    )

                    velocity_value = (
                        detection["velocity"]
                    )

                    lateral_value = (
                        detection["lateral"]
                    )

                    forward_value = (
                        detection["forward"]
                    )

                    range_spread = (
                        detection["range_spread"]
                    )

                else:

                    quality = "none"
                    range_value = ""
                    velocity_value = ""
                    lateral_value = ""
                    forward_value = ""
                    range_spread = ""

                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "sample_token": sample[
                            "token"
                        ],
                        "object_id_in_frame": detection[
                            "id"
                        ],
                        "class_id": detection[
                            "class_id"
                        ],
                        "class_name": detection[
                            "class_name"
                        ],
                        "confidence": detection[
                            "confidence"
                        ],
                        "candidate_points": detection[
                            "candidate_count"
                        ],
                        "selected_points": selected_points,
                        "removed_points": detection[
                            "removed_point_count"
                        ],
                        "number_of_clusters": detection[
                            "number_of_clusters"
                        ],
                        "quality": quality,
                        "range_m": range_value,
                        "relative_radial_velocity_mps": velocity_value,
                        "lateral_position_m": lateral_value,
                        "forward_position_m": forward_value,
                        "range_spread_m": range_spread
                    }
                )

            camera_panel = draw_camera_panel(
                radar,
                detections,
                frame_index
            )

            bev_panel = draw_bev_panel(
                radar,
                detections,
                frame_index
            )

            combined_frame = np.hstack(
                (
                    camera_panel,
                    bev_panel
                )
            )

            video_writer.write(
                combined_frame
            )

            if SAVE_INDIVIDUAL_FRAMES:

                frame_path = (
                    frames_output_dir
                    / (
                        f"fusion_"
                        f"{frame_index:03d}.jpg"
                    )
                )

                cv2.imwrite(
                    str(frame_path),
                    combined_frame
                )

            associated_count = sum(
                detection["point_count"] > 0
                for detection in detections
            )

            print(
                f"YOLO objects="
                f"{len(detections)}, "
                f"radar-associated="
                f"{associated_count}"
            )

            if sample["next"]:

                sample = nusc.get(
                    "sample",
                    sample["next"]
                )

    video_writer.release()

    print("\n" + "=" * 72)
    print("MULTI-FRAME PROCESSING FINISHED")
    print("=" * 72)

    print(
        f"Total YOLO detections: "
        f"{total_objects}"
    )

    print(
        f"Radar-associated detections: "
        f"{total_associated_objects}"
    )

    if total_objects > 0:

        association_rate = (
            100.0
            * total_associated_objects
            / total_objects
        )

        print(
            f"Association rate: "
            f"{association_rate:.2f}%"
        )

    print(
        f"\nCSV saved at:\n"
        f"{CSV_PATH}"
    )

    print(
        f"\nVideo saved at:\n"
        f"{VIDEO_PATH}"
    )

    if SAVE_INDIVIDUAL_FRAMES:

        print(
            f"\nIndividual frames saved at:\n"
            f"{frames_output_dir}"
        )


if __name__ == "__main__":
    main()
