from pathlib import Path
import os
import subprocess

import imageio_ffmpeg


PROJECT_ROOT = Path(__file__).resolve().parent

REPORT_ROOT = PROJECT_ROOT / "nuscenes_report"

VIDEO_FILES = [
    REPORT_ROOT / "assets" / "0061" / "scene_0061.mp4",
    REPORT_ROOT / "assets" / "0103" / "scene_0103.mp4",
    REPORT_ROOT / "assets" / "0553" / "scene_0553.mp4",
]


def convert_to_h264(input_path):

    if not input_path.is_file():

        print("File not found:")
        print(input_path)

        return False

    temporary_path = input_path.with_name(
        input_path.stem + "_h264.mp4"
    )

    ffmpeg_path = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

    command = [
        ffmpeg_path,

        "-y",

        "-i",
        str(input_path),

        # خروجی بدون صدا
        "-an",

        # کدک سازگار با Chrome و Edge
        "-c:v",
        "libx264",

        # کیفیت مناسب
        "-crf",
        "20",

        # سرعت تبدیل
        "-preset",
        "medium",

        # سازگاری بیشتر با مرورگرها
        "-pix_fmt",
        "yuv420p",

        # امکان شروع سریع ویدئو در مرورگر
        "-movflags",
        "+faststart",

        str(temporary_path),
    ]

    print()
    print("=" * 65)
    print("Converting:")
    print(input_path.name)
    print("=" * 65)

    try:

        subprocess.run(
            command,
            check=True,
        )

    except subprocess.CalledProcessError:

        print("Video conversion failed:")
        print(input_path)

        return False

    # جایگزینی فایل قدیمی با نسخه H.264
    os.replace(
        temporary_path,
        input_path,
    )

    print("Converted successfully:")
    print(input_path)

    return True


def main():

    print("=" * 65)
    print("CONVERTING VIDEOS TO H.264")
    print("=" * 65)

    successful = 0

    for video_path in VIDEO_FILES:

        if convert_to_h264(video_path):

            successful += 1

    print()
    print("=" * 65)
    print("CONVERSION FINISHED")
    print("=" * 65)

    print(
        "Converted videos:",
        successful,
        "/",
        len(VIDEO_FILES),
    )

    if successful == len(VIDEO_FILES):

        print()
        print(
            "All videos are now compatible "
            "with Chrome and Edge."
        )

        print()
        print("Open:")
        print(
            REPORT_ROOT / "index.html"
        )


if __name__ == "__main__":

    main()
