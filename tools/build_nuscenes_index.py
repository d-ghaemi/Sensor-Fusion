from pathlib import Path
import json
import re
import shutil

import cv2
import numpy as np


# ============================================================
# تنظیمات اصلی
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"

REPORT_ROOT = PROJECT_ROOT / "nuscenes_report"

# سرعت ویدئو
VIDEO_FPS = 5.0

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

SCENARIOS = [
    {
        "key": "0061",
        "title": "سناریوی اول — scene-0061",
        "run_prefix": "nuscenes_scene_0061",
        "description": (
            "کامیون پارک‌شده، عملیات عمرانی، تقاطع، "
            "گردش به چپ و دنبال‌کردن ون"
        ),
    },
    {
        "key": "0103",
        "title": "سناریوی دوم — scene-0103",
        "run_prefix": "nuscenes_scene_0103",
        "description": (
            "عابران پیاده، خودروی در حال گردش، "
            "ردیف دوچرخه‌ها و دوچرخه‌سوار"
        ),
    },
    {
        "key": "0553",
        "title": "سناریوی سوم — scene-0553",
        "run_prefix": "nuscenes_scene_0553",
        "description": (
            "توقف در تقاطع، گذرگاه عابر، دوچرخه، "
            "خودروها و عابران پیاده"
        ),
    },
]


# ============================================================
# مرتب‌سازی تصاویر
# ============================================================

def natural_key(path):

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_image_files(folder):

    if not folder.is_dir():
        return []

    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(
        files,
        key=natural_key,
    )


# ============================================================
# پیدا کردن پوشه خروجی YOLO
# ============================================================

def find_run_folder(prefix):

    exact_folder = RUNS_ROOT / prefix

    if (
        exact_folder.is_dir()
        and get_image_files(exact_folder)
    ):
        return exact_folder

    candidates = [
        folder
        for folder in RUNS_ROOT.glob(prefix + "*")
        if folder.is_dir()
        and get_image_files(folder)
    ]

    if not candidates:

        raise FileNotFoundError(
            "خروجی سناریو پیدا نشد:\n"
            + prefix
            + "\n\nمسیر بررسی‌شده:\n"
            + str(RUNS_ROOT)
        )

    # در صورت وجود چند خروجی، جدیدترین انتخاب می‌شود
    return max(
        candidates,
        key=lambda folder: folder.stat().st_mtime,
    )


# ============================================================
# خواندن تصاویر
# ============================================================

def read_image(path):

    # این روش با مسیرهای Unicode ویندوز نیز سازگارتر است
    binary_data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        binary_data,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise ValueError(
            "تصویر قابل خواندن نیست:\n"
            + str(path)
        )

    return image


# ============================================================
# ساخت ویدئو
# ============================================================

def make_video(frames, output_path, fps):

    first_frame = read_image(frames[0])

    height, width = first_frame.shape[:2]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        raise RuntimeError(
            "ویدئو ساخته نشد:\n"
            + str(output_path)
        )

    try:

        for frame_path in frames:

            frame = read_image(frame_path)

            if frame.shape[:2] != (height, width):

                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )

            writer.write(frame)

    finally:

        writer.release()

    return width, height


# ============================================================
# کپی فریم‌ها در پوشه گزارش
# ============================================================

def copy_frames(frames, destination):

    if destination.exists():

        shutil.rmtree(destination)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied_names = []

    for index, source_path in enumerate(frames):

        extension = source_path.suffix.lower()

        new_name = "frame_{:03d}{}".format(
            index,
            extension,
        )

        destination_path = (
            destination / new_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        copied_names.append(new_name)

    return copied_names


# ============================================================
# ساخت HTML هر سناریو
# ============================================================

def make_scene_html(record):

    key = record["key"]

    thumbnails = []

    for index, filename in enumerate(
        record["frames"]
    ):

        thumbnail = (
            '<button class="thumb" '
            'data-scene="{}" '
            'data-index="{}">'
            '<img loading="lazy" '
            'src="assets/{}/frames/{}" '
            'alt="فریم {}">'
            '<span>فریم {:03d}</span>'
            '</button>'
        ).format(
            key,
            index,
            key,
            filename,
            index,
            index,
        )

        thumbnails.append(thumbnail)

    scene_template = """
    <section class="scene" id="scene-__KEY__">

        <div class="scene-head">

            <div>

                <p class="eyebrow">
                    nuScenes-mini / CAM_FRONT
                </p>

                <h2>__TITLE__</h2>

                <p>__DESCRIPTION__</p>

            </div>

            <div class="stat">

                <strong>__COUNT__</strong>

                <span>
                    فریم پردازش‌شده
                </span>

            </div>

        </div>

        <video
            controls
            preload="metadata"
            poster="assets/__KEY__/frames/__POSTER__"
        >

            <source
                src="assets/__KEY__/scene___KEY__.mp4"
                type="video/mp4"
            >

            مرورگر شما از پخش ویدئوی MP4
            پشتیبانی نمی‌کند.

        </video>

        <div class="legend">

            <span>
                <i class="yellow"></i>
                کادر زرد: اشیای تشخیص‌داده‌شده
            </span>

            <span>
                <i class="green"></i>
                سبز: ناحیه قابل رانندگی
            </span>

            <span>
                <i class="red"></i>
                قرمز: خطوط جاده
            </span>

        </div>

        <details>

            <summary>
                نمایش گالری تمام فریم‌ها
            </summary>

            <div class="gallery">
                __GALLERY__
            </div>

        </details>

    </section>
    """

    scene_html = scene_template.replace(
        "__KEY__",
        key,
    )

    scene_html = scene_html.replace(
        "__TITLE__",
        record["title"],
    )

    scene_html = scene_html.replace(
        "__DESCRIPTION__",
        record["description"],
    )

    scene_html = scene_html.replace(
        "__COUNT__",
        str(record["count"]),
    )

    scene_html = scene_html.replace(
        "__POSTER__",
        record["frames"][0],
    )

    scene_html = scene_html.replace(
        "__GALLERY__",
        "\n".join(thumbnails),
    )

    return scene_html


# ============================================================
# ساخت فایل اصلی HTML
# ============================================================

def build_html(records):

    navigation_items = []

    scene_sections = []

    frame_map = {}

    for record in records:

        navigation_items.append(
            '<a href="#scene-{}">{}</a>'.format(
                record["key"],
                record["title"],
            )
        )

        scene_sections.append(
            make_scene_html(record)
        )

        frame_map[record["key"]] = [
            "assets/{}/frames/{}".format(
                record["key"],
                filename,
            )
            for filename in record["frames"]
        ]

    # این قالب f-string نیست و مشکل آکولاد ندارد
    html_template = r"""
<!DOCTYPE html>

<html lang="fa" dir="rtl">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        خروجی YOLOPv2 روی nuScenes-mini
    </title>

    <style>

        :root {
            --background: #07111f;
            --panel: #101e31;
            --panel-light: #162944;
            --text: #edf5ff;
            --muted: #a8bad0;
            --accent: #56b7ff;
            --border: #29425f;
        }

        * {
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            margin: 0;
            color: var(--text);
            font-family: Tahoma, "Segoe UI", sans-serif;
            line-height: 1.9;

            background:
                linear-gradient(
                    145deg,
                    #07111f,
                    #0b1930 55%,
                    #081320
                );
        }

        header,
        main {
            width: min(1400px, 92%);
            margin: auto;
        }

        header {
            padding: 60px 0 40px;
        }

        h1 {
            margin: 5px 0 10px;
            font-size: clamp(2rem, 5vw, 4rem);
            line-height: 1.3;
        }

        h2 {
            margin: 4px 0;
            font-size: clamp(1.5rem, 3vw, 2.2rem);
        }

        p {
            color: var(--muted);
        }

        .eyebrow {
            margin: 0;
            color: var(--accent);
            font-weight: bold;
        }

        nav {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 25px;
        }

        nav a {
            padding: 8px 18px;
            color: var(--text);
            text-decoration: none;
            background: #132641;
            border: 1px solid var(--border);
            border-radius: 30px;
        }

        nav a:hover {
            border-color: var(--accent);
        }

        main {
            padding-bottom: 70px;
        }

        .scene {
            padding: 32px;
            margin-bottom: 35px;
            background: rgba(16, 30, 49, 0.95);
            border: 1px solid var(--border);
            border-radius: 24px;

            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.4);
        }

        .scene-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 25px;
            margin-bottom: 20px;
        }

        .stat {
            min-width: 155px;
            padding: 15px;
            text-align: center;
            background: var(--panel-light);
            border-radius: 18px;
        }

        .stat strong {
            display: block;
            color: var(--accent);
            font-size: 2.4rem;
            line-height: 1.2;
        }

        .stat span {
            color: var(--muted);
            font-size: 0.85rem;
        }

        video {
            display: block;
            width: 100%;
            max-height: 75vh;
            background: black;
            border-radius: 16px;

            box-shadow:
                0 10px 35px rgba(0, 0, 0, 0.6);
        }

        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 22px;
            padding: 17px 3px;
            color: var(--muted);
        }

        .legend i {
            display: inline-block;
            width: 14px;
            height: 14px;
            margin-left: 6px;
            border-radius: 50%;
        }

        .yellow {
            background: #ffeb00;
        }

        .green {
            background: #31df53;
        }

        .red {
            background: #e43737;
        }

        details {
            padding-top: 14px;
            border-top: 1px solid var(--border);
        }

        summary {
            color: var(--accent);
            font-weight: bold;
            cursor: pointer;
        }

        .gallery {
            display: grid;

            grid-template-columns:
                repeat(
                    auto-fill,
                    minmax(190px, 1fr)
                );

            gap: 12px;
            margin-top: 18px;
        }

        .thumb {
            padding: 0;
            overflow: hidden;
            color: var(--text);
            text-align: right;
            cursor: pointer;
            background: #081526;
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        .thumb:hover {
            border-color: var(--accent);
            transform: translateY(-2px);
        }

        .thumb img {
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
        }

        .thumb span {
            display: block;
            padding: 7px 10px;
        }

        dialog {
            width: min(1400px, 96vw);
            padding: 14px;
            color: white;
            background: var(--background);
            border: 1px solid var(--border);
            border-radius: 18px;
        }

        dialog::backdrop {
            background: rgba(0, 0, 0, 0.87);
        }

        #modal-image {
            display: block;
            width: 100%;
            max-height: 82vh;
            object-fit: contain;
            background: black;
        }

        .controls {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            padding-top: 10px;
        }

        .controls button {
            padding: 8px 16px;
            color: white;
            cursor: pointer;
            background: #132641;
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        footer {
            padding: 10px 20px 50px;
            color: var(--muted);
            text-align: center;
        }

        @media (max-width: 700px) {

            .scene {
                padding: 18px;
            }

            .scene-head {
                display: block;
            }

            .stat {
                margin-top: 15px;
            }
        }

    </style>

</head>

<body>

    <header>

        <p class="eyebrow">
            YOLOPv2 × nuScenes-mini
        </p>

        <h1>
            داشبورد نتایج سه سناریوی رانندگی
        </h1>

        <p>
            نمایش ویدئویی و فریم‌به‌فریم تشخیص اشیا،
            ناحیه قابل رانندگی و خطوط جاده با استفاده
            از وزن‌های ازپیش‌آموزش‌دیده و بدون آموزش مجدد.
        </p>

        <nav>
            __NAVIGATION__
        </nav>

    </header>

    <main>
        __SECTIONS__
    </main>

    <footer>
        گزارش خروجی YOLOPv2 روی تصاویر CAM_FRONT
        مجموعه‌داده nuScenes-mini
    </footer>

    <dialog id="viewer">

        <img
            id="modal-image"
            alt="فریم انتخاب‌شده"
        >

        <div class="controls">

            <button id="next-frame">
                فریم بعدی
            </button>

            <span id="counter"></span>

            <button id="previous-frame">
                فریم قبلی
            </button>

            <button id="close-viewer">
                بستن
            </button>

        </div>

    </dialog>

    <script>

        const frames = __FRAME_MAP__;

        const viewer =
            document.getElementById("viewer");

        const modalImage =
            document.getElementById("modal-image");

        const counter =
            document.getElementById("counter");

        let currentScene = "";
        let currentIndex = 0;

        function showFrame(scene, index) {

            currentScene = scene;

            const numberOfFrames =
                frames[scene].length;

            currentIndex =
                (index + numberOfFrames)
                % numberOfFrames;

            modalImage.src =
                frames[scene][currentIndex];

            counter.textContent =
                "فریم "
                + (currentIndex + 1)
                + " از "
                + numberOfFrames;

            if (!viewer.open) {
                viewer.showModal();
            }
        }

        document
            .querySelectorAll(".thumb")
            .forEach(function(button) {

                button.addEventListener(
                    "click",
                    function() {

                        showFrame(
                            button.dataset.scene,
                            Number(
                                button.dataset.index
                            )
                        );
                    }
                );
            });

        document
            .getElementById("next-frame")
            .onclick = function() {

                showFrame(
                    currentScene,
                    currentIndex + 1
                );
            };

        document
            .getElementById("previous-frame")
            .onclick = function() {

                showFrame(
                    currentScene,
                    currentIndex - 1
                );
            };

        document
            .getElementById("close-viewer")
            .onclick = function() {

                viewer.close();
            };

        document.addEventListener(
            "keydown",
            function(event) {

                if (!viewer.open) {
                    return;
                }

                if (event.key === "ArrowLeft") {

                    showFrame(
                        currentScene,
                        currentIndex + 1
                    );
                }

                if (event.key === "ArrowRight") {

                    showFrame(
                        currentScene,
                        currentIndex - 1
                    );
                }
            }
        );

    </script>

</body>

</html>
"""

    frame_map_json = json.dumps(
        frame_map,
        ensure_ascii=False,
    )

    html_output = html_template.replace(
        "__NAVIGATION__",
        "\n".join(navigation_items),
    )

    html_output = html_output.replace(
        "__SECTIONS__",
        "\n".join(scene_sections),
    )

    html_output = html_output.replace(
        "__FRAME_MAP__",
        frame_map_json,
    )

    return html_output


# ============================================================
# تابع اصلی
# ============================================================

def main():

    print("=" * 68)

    print(
        "BUILDING NUSCENES YOLOPv2 DASHBOARD"
    )

    print("=" * 68)

    if not RUNS_ROOT.is_dir():

        raise FileNotFoundError(
            "پوشه خروجی YOLO پیدا نشد:\n"
            + str(RUNS_ROOT)
        )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for scenario in SCENARIOS:

        print()

        print(
            "Processing:",
            scenario["title"],
        )

        run_folder = find_run_folder(
            scenario["run_prefix"]
        )

        frames = get_image_files(
            run_folder
        )

        print(
            "Output folder:",
            run_folder,
        )

        print(
            "Frames:",
            len(frames),
        )

        asset_folder = (
            REPORT_ROOT
            / "assets"
            / scenario["key"]
        )

        copied_names = copy_frames(
            frames,
            asset_folder / "frames",
        )

        video_path = (
            asset_folder
            / (
                "scene_"
                + scenario["key"]
                + ".mp4"
            )
        )

        width, height = make_video(
            frames,
            video_path,
            VIDEO_FPS,
        )

        print(
            "Video:",
            video_path.name,
        )

        print(
            "Resolution:",
            width,
            "x",
            height,
        )

        records.append(
            {
                **scenario,
                "frames": copied_names,
                "count": len(frames),
            }
        )

    index_path = (
        REPORT_ROOT
        / "index.html"
    )

    index_path.write_text(
        build_html(records),
        encoding="utf-8",
    )

    metadata = {
        "model": "YOLOPv2 pretrained weights",
        "dataset": "nuScenes-mini CAM_FRONT",
        "video_fps": VIDEO_FPS,
        "total_frames": sum(
            record["count"]
            for record in records
        ),
        "scenarios": [
            {
                "key": record["key"],
                "title": record["title"],
                "description": record["description"],
                "count": record["count"],
            }
            for record in records
        ],
    }

    metadata_path = (
        REPORT_ROOT
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print("=" * 68)

    print(
        "DASHBOARD FINISHED"
    )

    print("=" * 68)

    print(
        "Total frames:",
        metadata["total_frames"],
    )

    print(
        "\nOpen this file in Chrome or Edge:"
    )

    print(index_path)


if __name__ == "__main__":

    main()
