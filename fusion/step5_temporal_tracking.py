from pathlib import Path
import csv
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    PROJECT_ROOT
    / "radar_fusion_outputs"
    / "scene0061_multiframe"
    / "scene0061_radar_fusion_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "radar_fusion_outputs"
    / "scene0061_tracking"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "scene0061_tracked_objects.csv"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "scene0061_tracking_bev.mp4"
)

OUTPUT_PLOT = (
    OUTPUT_DIR
    / "scene0061_tracking_summary.png"
)

NUMBER_OF_FRAMES = 39

# فاصله زمانی تقریبی بین keyframeهای nuScenes
FRAME_PERIOD_SECONDS = 0.5

# بیشترین فاصله قابل قبول برای اتصال دو تشخیص
# در دو فریم متوالی
MAX_ASSOCIATION_DISTANCE_M = 8.0

# یک مسیر حداکثر چند فریم می‌تواند تشخیص داده نشود؟
MAX_MISSED_FRAMES = 3

# حداقل تعداد نقاط راداری برای ورود تشخیص به ردیاب
MIN_SELECTED_POINTS = 1

# حداقل تعداد مشاهده برای معتبرشدن مسیر
MIN_TRACK_HITS = 2

# ضریب نرم‌سازی سرعت موقعیت در BEV
VELOCITY_SMOOTHING = 0.6

# تنظیمات ویدئو
VIDEO_FPS = 5.0
VIDEO_WIDTH = 1000
VIDEO_HEIGHT = 800

# محدوده BEV
LATERAL_MIN = -35.0
LATERAL_MAX = 35.0
FORWARD_MIN = 0.0
FORWARD_MAX = 90.0

# تعداد نقاط قبلی که به عنوان دنباله مسیر نمایش داده می‌شوند
MAX_TRAIL_LENGTH = 15


# ============================================================
# TRACK CLASS
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        detection,
        frame_index
    ):

        self.track_id = track_id

        self.lateral = detection[
            "lateral_position_m"
        ]

        self.forward = detection[
            "forward_position_m"
        ]

        # سرعت تغییر موقعیت در صفحه BEV
        self.velocity_lateral = 0.0
        self.velocity_forward = 0.0

        self.last_frame = frame_index

        self.missed_frames = 0
        self.hits = 1

        self.range_m = detection[
            "range_m"
        ]

        self.radial_velocity = detection[
            "relative_radial_velocity_mps"
        ]

        self.selected_points = detection[
            "selected_points"
        ]

        self.quality = detection[
            "quality"
        ]

        self.history = [
            (
                frame_index,
                self.lateral,
                self.forward,
                self.range_m,
                self.radial_velocity
            )
        ]

    def predict(self, frame_index):
        """
        پیش‌بینی موقعیت مسیر در فریم جدید
        با مدل سرعت ثابت.
        """

        frame_difference = (
            frame_index
            - self.last_frame
        )

        predicted_lateral = (
            self.lateral
            + self.velocity_lateral
            * frame_difference
        )

        predicted_forward = (
            self.forward
            + self.velocity_forward
            * frame_difference
        )

        return np.asarray(
            [
                predicted_lateral,
                predicted_forward
            ],
            dtype=np.float64
        )

    def update(
        self,
        detection,
        frame_index
    ):
        """
        به‌روزرسانی مسیر با تشخیص جدید.
        """

        frame_difference = max(
            1,
            frame_index - self.last_frame
        )

        new_lateral = detection[
            "lateral_position_m"
        ]

        new_forward = detection[
            "forward_position_m"
        ]

        measured_velocity_lateral = (
            new_lateral
            - self.lateral
        ) / frame_difference

        measured_velocity_forward = (
            new_forward
            - self.forward
        ) / frame_difference

        self.velocity_lateral = (
            VELOCITY_SMOOTHING
            * self.velocity_lateral
            + (
                1.0
                - VELOCITY_SMOOTHING
            )
            * measured_velocity_lateral
        )

        self.velocity_forward = (
            VELOCITY_SMOOTHING
            * self.velocity_forward
            + (
                1.0
                - VELOCITY_SMOOTHING
            )
            * measured_velocity_forward
        )

        self.lateral = new_lateral
        self.forward = new_forward

        self.range_m = detection[
            "range_m"
        ]

        self.radial_velocity = detection[
            "relative_radial_velocity_mps"
        ]

        self.selected_points = detection[
            "selected_points"
        ]

        self.quality = detection[
            "quality"
        ]

        self.last_frame = frame_index
        self.missed_frames = 0
        self.hits += 1

        self.history.append(
            (
                frame_index,
                self.lateral,
                self.forward,
                self.range_m,
                self.radial_velocity
            )
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        result = float(value)
    except ValueError:
        return None

    if not np.isfinite(result):
        return None

    return result


def safe_int(value, default=0):

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def read_detections():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n"
            f"{INPUT_CSV}"
        )

    detections_by_frame = {
        frame_index: []
        for frame_index in range(
            NUMBER_OF_FRAMES
        )
    }

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            frame_index = safe_int(
                row["frame_index"]
            )

            selected_points = safe_int(
                row["selected_points"]
            )

            lateral = safe_float(
                row["lateral_position_m"]
            )

            forward = safe_float(
                row["forward_position_m"]
            )

            range_m = safe_float(
                row["range_m"]
            )

            radial_velocity = safe_float(
                row[
                    "relative_radial_velocity_mps"
                ]
            )

            if (
                selected_points
                < MIN_SELECTED_POINTS
            ):
                continue

            if (
                lateral is None
                or forward is None
                or range_m is None
                or radial_velocity is None
            ):
                continue

            detection = {
                "frame_index": frame_index,
                "object_id_in_frame": safe_int(
                    row["object_id_in_frame"]
                ),
                "class_id": safe_int(
                    row["class_id"]
                ),
                "class_name": row[
                    "class_name"
                ],
                "confidence": safe_float(
                    row["confidence"]
                ),
                "selected_points": selected_points,
                "quality": row["quality"],
                "range_m": range_m,
                "relative_radial_velocity_mps": (
                    radial_velocity
                ),
                "lateral_position_m": lateral,
                "forward_position_m": forward
            }

            if (
                frame_index
                in detections_by_frame
            ):

                detections_by_frame[
                    frame_index
                ].append(detection)

    return detections_by_frame


def distance_between(
    position_1,
    position_2
):

    return float(
        np.linalg.norm(
            position_1
            - position_2
        )
    )


# ============================================================
# TRACKING
# ============================================================

def run_tracking(
    detections_by_frame
):

    active_tracks = {}
    finished_tracks = {}

    next_track_id = 1

    assignments_by_frame = {
        frame_index: []
        for frame_index in range(
            NUMBER_OF_FRAMES
        )
    }

    for frame_index in range(
        NUMBER_OF_FRAMES
    ):

        detections = detections_by_frame[
            frame_index
        ]

        track_ids = list(
            active_tracks.keys()
        )

        matched_track_ids = set()
        matched_detection_indices = set()

        # ----------------------------------------------------
        # Hungarian matching
        # ----------------------------------------------------

        if (
            len(track_ids) > 0
            and len(detections) > 0
        ):

            cost_matrix = np.full(
                (
                    len(track_ids),
                    len(detections)
                ),
                1e6,
                dtype=np.float64
            )

            for track_row, track_id in enumerate(
                track_ids
            ):

                track = active_tracks[
                    track_id
                ]

                predicted_position = (
                    track.predict(
                        frame_index
                    )
                )

                for detection_column, detection in enumerate(
                    detections
                ):

                    detection_position = np.asarray(
                        [
                            detection[
                                "lateral_position_m"
                            ],
                            detection[
                                "forward_position_m"
                            ]
                        ],
                        dtype=np.float64
                    )

                    position_distance = (
                        distance_between(
                            predicted_position,
                            detection_position
                        )
                    )

                    # اختلاف سرعت شعاعی نیز در هزینه اثر دارد
                    velocity_difference = abs(
                        track.radial_velocity
                        - detection[
                            "relative_radial_velocity_mps"
                        ]
                    )

                    cost = (
                        position_distance
                        + 0.20
                        * velocity_difference
                    )

                    cost_matrix[
                        track_row,
                        detection_column
                    ] = cost

            rows, columns = (
                linear_sum_assignment(
                    cost_matrix
                )
            )

            for row, column in zip(
                rows,
                columns
            ):

                track_id = track_ids[row]

                detection = detections[
                    column
                ]

                predicted_position = (
                    active_tracks[
                        track_id
                    ].predict(
                        frame_index
                    )
                )

                detection_position = np.asarray(
                    [
                        detection[
                            "lateral_position_m"
                        ],
                        detection[
                            "forward_position_m"
                        ]
                    ]
                )

                position_distance = (
                    distance_between(
                        predicted_position,
                        detection_position
                    )
                )

                if (
                    position_distance
                    <= MAX_ASSOCIATION_DISTANCE_M
                ):

                    active_tracks[
                        track_id
                    ].update(
                        detection,
                        frame_index
                    )

                    matched_track_ids.add(
                        track_id
                    )

                    matched_detection_indices.add(
                        column
                    )

                    assignment = (
                        detection.copy()
                    )

                    assignment[
                        "track_id"
                    ] = track_id

                    assignment[
                        "track_hits"
                    ] = active_tracks[
                        track_id
                    ].hits

                    assignments_by_frame[
                        frame_index
                    ].append(
                        assignment
                    )

        # ----------------------------------------------------
        # CREATE NEW TRACKS
        # ----------------------------------------------------

        for detection_index, detection in enumerate(
            detections
        ):

            if (
                detection_index
                in matched_detection_indices
            ):
                continue

            new_track = Track(
                next_track_id,
                detection,
                frame_index
            )

            active_tracks[
                next_track_id
            ] = new_track

            assignment = detection.copy()

            assignment[
                "track_id"
            ] = next_track_id

            assignment[
                "track_hits"
            ] = 1

            assignments_by_frame[
                frame_index
            ].append(
                assignment
            )

            matched_track_ids.add(
                next_track_id
            )

            next_track_id += 1

        # ----------------------------------------------------
        # INCREASE MISSED COUNTERS
        # ----------------------------------------------------

        tracks_to_remove = []

        for track_id, track in active_tracks.items():

            if (
                track_id
                not in matched_track_ids
            ):

                track.missed_frames += 1

            if (
                track.missed_frames
                > MAX_MISSED_FRAMES
            ):

                tracks_to_remove.append(
                    track_id
                )

        for track_id in tracks_to_remove:

            finished_tracks[
                track_id
            ] = active_tracks.pop(
                track_id
            )

        print(
            f"Frame {frame_index + 1:02d}/"
            f"{NUMBER_OF_FRAMES}: "
            f"detections={len(detections)}, "
            f"active_tracks="
            f"{len(active_tracks)}"
        )

    finished_tracks.update(
        active_tracks
    )

    return (
        assignments_by_frame,
        finished_tracks
    )


# ============================================================
# SAVE TRACKING CSV
# ============================================================

def save_tracking_csv(
    assignments_by_frame
):

    fields = [
        "frame_index",
        "time_s",
        "track_id",
        "track_hits",
        "object_id_in_frame",
        "class_name",
        "selected_points",
        "quality",
        "range_m",
        "relative_radial_velocity_mps",
        "lateral_position_m",
        "forward_position_m"
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for frame_index in range(
            NUMBER_OF_FRAMES
        ):

            time_s = (
                frame_index
                * FRAME_PERIOD_SECONDS
            )

            for assignment in assignments_by_frame[
                frame_index
            ]:

                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "time_s": time_s,
                        "track_id": assignment[
                            "track_id"
                        ],
                        "track_hits": assignment[
                            "track_hits"
                        ],
                        "object_id_in_frame": assignment[
                            "object_id_in_frame"
                        ],
                        "class_name": assignment[
                            "class_name"
                        ],
                        "selected_points": assignment[
                            "selected_points"
                        ],
                        "quality": assignment[
                            "quality"
                        ],
                        "range_m": assignment[
                            "range_m"
                        ],
                        "relative_radial_velocity_mps": assignment[
                            "relative_radial_velocity_mps"
                        ],
                        "lateral_position_m": assignment[
                            "lateral_position_m"
                        ],
                        "forward_position_m": assignment[
                            "forward_position_m"
                        ]
                    }
                )


# ============================================================
# BEV VIDEO
# ============================================================

def bev_to_pixel(
    lateral,
    forward
):

    x_ratio = (
        (
            lateral
            - LATERAL_MIN
        )
        / (
            LATERAL_MAX
            - LATERAL_MIN
        )
    )

    y_ratio = (
        (
            forward
            - FORWARD_MIN
        )
        / (
            FORWARD_MAX
            - FORWARD_MIN
        )
    )

    pixel_x = int(
        x_ratio
        * (VIDEO_WIDTH - 100)
        + 50
    )

    pixel_y = int(
        (
            1.0 - y_ratio
        )
        * (VIDEO_HEIGHT - 100)
        + 50
    )

    return pixel_x, pixel_y


def track_color(
    track_id
):

    rng = np.random.default_rng(
        track_id
    )

    color = rng.integers(
        60,
        240,
        size=3
    )

    return tuple(
        int(value)
        for value in color
    )


def create_tracking_video(
    assignments_by_frame,
    tracks
):

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        VIDEO_FPS,
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create tracking video."
        )

    history_until_frame = {}

    for frame_index in range(
        NUMBER_OF_FRAMES
    ):

        image = np.full(
            (
                VIDEO_HEIGHT,
                VIDEO_WIDTH,
                3
            ),
            245,
            dtype=np.uint8
        )

        cv2.rectangle(
            image,
            (0, 0),
            (VIDEO_WIDTH, 50),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            image,
            (
                f"scene-0061 temporal tracking | "
                f"frame {frame_index:03d} | "
                f"time="
                f"{frame_index * FRAME_PERIOD_SECONDS:.1f}s"
            ),
            (25, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # شبکه
        for forward in range(
            0,
            91,
            10
        ):

            x1, y = bev_to_pixel(
                LATERAL_MIN,
                forward
            )

            x2, _ = bev_to_pixel(
                LATERAL_MAX,
                forward
            )

            cv2.line(
                image,
                (x1, y),
                (x2, y),
                (210, 210, 210),
                1
            )

            cv2.putText(
                image,
                f"{forward}m",
                (5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (80, 80, 80),
                1
            )

        for lateral in range(
            -30,
            31,
            10
        ):

            x, y1 = bev_to_pixel(
                lateral,
                0
            )

            _, y2 = bev_to_pixel(
                lateral,
                90
            )

            cv2.line(
                image,
                (x, y1),
                (x, y2),
                (210, 210, 210),
                1
            )

        # Ego vehicle
        ego_x, ego_y = bev_to_pixel(
            0,
            0
        )

        triangle = np.asarray(
            [
                [ego_x, ego_y - 22],
                [ego_x - 13, ego_y],
                [ego_x + 13, ego_y]
            ],
            dtype=np.int32
        )

        cv2.fillPoly(
            image,
            [triangle],
            (0, 0, 0)
        )

        # افزودن مشاهدات فریم جاری به تاریخچه
        for assignment in assignments_by_frame[
            frame_index
        ]:

            track_id = assignment[
                "track_id"
            ]

            history_until_frame.setdefault(
                track_id,
                []
            )

            history_until_frame[
                track_id
            ].append(
                (
                    assignment[
                        "lateral_position_m"
                    ],
                    assignment[
                        "forward_position_m"
                    ]
                )
            )

        # رسم مسیرها
        for track_id, history in history_until_frame.items():

            if len(history) == 0:
                continue

            color = track_color(
                track_id
            )

            recent_history = history[
                -MAX_TRAIL_LENGTH:
            ]

            pixel_history = [
                bev_to_pixel(
                    lateral,
                    forward
                )
                for lateral, forward
                in recent_history
            ]

            if len(pixel_history) >= 2:

                cv2.polylines(
                    image,
                    [
                        np.asarray(
                            pixel_history,
                            dtype=np.int32
                        )
                    ],
                    False,
                    color,
                    2
                )

        # رسم تشخیص‌های فعلی
        for assignment in assignments_by_frame[
            frame_index
        ]:

            track_id = assignment[
                "track_id"
            ]

            color = track_color(
                track_id
            )

            center = bev_to_pixel(
                assignment[
                    "lateral_position_m"
                ],
                assignment[
                    "forward_position_m"
                ]
            )

            cv2.circle(
                image,
                center,
                9,
                color,
                -1
            )

            cv2.circle(
                image,
                center,
                9,
                (0, 0, 0),
                1
            )

            label = (
                f"T{track_id} "
                f"R={assignment['range_m']:.1f}m "
                f"V={assignment['relative_radial_velocity_mps']:.1f}"
            )

            cv2.putText(
                image,
                label,
                (
                    center[0] + 10,
                    center[1] - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        writer.write(
            image
        )

    writer.release()


# ============================================================
# SUMMARY PLOT
# ============================================================

def create_summary_plot(
    tracks
):

    valid_tracks = [
        track
        for track in tracks.values()
        if track.hits >= MIN_TRACK_HITS
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6)
    )

    for track in valid_tracks:

        history = np.asarray(
            track.history,
            dtype=float
        )

        frames = history[:, 0]

        time = (
            frames
            * FRAME_PERIOD_SECONDS
        )

        lateral = history[:, 1]
        forward = history[:, 2]
        ranges = history[:, 3]
        velocities = history[:, 4]

        label = (
            f"Track {track.track_id} "
            f"(hits={track.hits})"
        )

        axes[0].plot(
            lateral,
            forward,
            marker="o",
            markersize=3,
            label=label
        )

        axes[1].plot(
            time,
            ranges,
            marker="o",
            markersize=3,
            label=label
        )

        axes[2].plot(
            time,
            velocities,
            marker="o",
            markersize=3,
            label=label
        )

    axes[0].scatter(
        0,
        0,
        marker="^",
        s=180,
        color="black",
        label="Ego vehicle"
    )

    axes[0].set_title(
        "Tracked trajectories in BEV"
    )

    axes[0].set_xlabel(
        "Lateral position (m)"
    )

    axes[0].set_ylabel(
        "Forward position (m)"
    )

    axes[0].set_xlim(
        LATERAL_MIN,
        LATERAL_MAX
    )

    axes[0].set_ylim(
        FORWARD_MIN,
        FORWARD_MAX
    )

    axes[1].set_title(
        "Tracked range"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].set_ylabel(
        "Range (m)"
    )

    axes[2].set_title(
        "Tracked radial velocity"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].set_ylabel(
        "Relative radial velocity (m/s)"
    )

    for axis in axes:

        axis.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

    if len(valid_tracks) <= 10:

        axes[0].legend(
            fontsize=7
        )

    figure.suptitle(
        "scene-0061: Temporal tracking of radar-associated vehicles",
        fontsize=16
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PLOT,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "STEP 5: TEMPORAL TRACKING OF "
        "RADAR-ASSOCIATED VEHICLES"
    )
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detections_by_frame = (
        read_detections()
    )

    input_detection_count = sum(
        len(detections)
        for detections
        in detections_by_frame.values()
    )

    print(
        f"Input associated detections: "
        f"{input_detection_count}"
    )

    assignments_by_frame, tracks = (
        run_tracking(
            detections_by_frame
        )
    )

    valid_tracks = [
        track
        for track in tracks.values()
        if track.hits >= MIN_TRACK_HITS
    ]

    save_tracking_csv(
        assignments_by_frame
    )

    create_tracking_video(
        assignments_by_frame,
        tracks
    )

    create_summary_plot(
        tracks
    )

    print("\n" + "=" * 72)
    print("TRACKING FINISHED")
    print("=" * 72)

    print(
        f"Total generated tracks: "
        f"{len(tracks)}"
    )

    print(
        f"Tracks with at least "
        f"{MIN_TRACK_HITS} observations: "
        f"{len(valid_tracks)}"
    )

    if len(valid_tracks) > 0:

        longest_track = max(
            valid_tracks,
            key=lambda track: track.hits
        )

        print(
            f"Longest track: "
            f"T{longest_track.track_id}"
        )

        print(
            f"Longest track observations: "
            f"{longest_track.hits}"
        )

    print(
        f"\nTracking CSV saved at:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        f"\nTracking video saved at:\n"
        f"{OUTPUT_VIDEO}"
    )

    print(
        f"\nTracking plot saved at:\n"
        f"{OUTPUT_PLOT}"
    )

    print(
        "\nImportant: Track IDs are estimated "
        "without ground-truth labels."
    )


if __name__ == "__main__":
    main()
