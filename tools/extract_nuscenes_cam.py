from pathlib import Path
import csv
import shutil

from nuscenes.nuscenes import NuScenes


# =========================================================
# SETTINGS
# =========================================================

# مسیر اصلی دیتاست nuScenes-mini
NUSCENES_ROOT = Path(r"D:\nuscene")

# پوشه‌ای که تصاویر استخراج‌شده در آن ذخیره می‌شوند
OUTPUT_ROOT = Path("data/nuscenes_cam_front")

# شماره سناریوهایی که می‌خواهیم بررسی کنیم
SCENE_INDICES = [0, 1, 2]

# دوربین مورد استفاده
CAMERA_CHANNEL = "CAM_FRONT"


# =========================================================
# CHECK DATASET
# =========================================================

if not NUSCENES_ROOT.exists():
    raise FileNotFoundError(
        f"nuScenes root was not found:\n{NUSCENES_ROOT}"
    )

metadata_folder = NUSCENES_ROOT / "v1.0-mini"

if not metadata_folder.exists():
    raise FileNotFoundError(
        f"v1.0-mini metadata folder was not found:\n"
        f"{metadata_folder}"
    )

camera_folder = NUSCENES_ROOT / "samples" / CAMERA_CHANNEL

if not camera_folder.exists():
    raise FileNotFoundError(
        f"{CAMERA_CHANNEL} folder was not found:\n"
        f"{camera_folder}"
    )


# =========================================================
# LOAD NUSCENES
# =========================================================

print("=" * 70)
print("LOADING NUSCENES-MINI")
print("=" * 70)

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(NUSCENES_ROOT),
    verbose=True
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

summary_rows = []


# =========================================================
# EXTRACT CAM_FRONT IMAGES
# =========================================================

for scene_index in SCENE_INDICES:

    if scene_index < 0 or scene_index >= len(nusc.scene):
        print(f"Scene index {scene_index} is invalid.")
        continue

    scene = nusc.scene[scene_index]

    scene_name = scene["name"]
    description = scene["description"]
    number_of_samples = scene["nbr_samples"]

    output_folder_name = f"{scene_index:02d}_{scene_name}"
    scene_output_folder = OUTPUT_ROOT / output_folder_name

    scene_output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("-" * 70)
    print(f"Scene index       : {scene_index}")
    print(f"Scene name        : {scene_name}")
    print(f"Description       : {description}")
    print(f"Number of samples : {number_of_samples}")
    print(f"Output folder     : {scene_output_folder.resolve()}")
    print("-" * 70)

    sample_token = scene["first_sample_token"]
    frame_number = 0
    copied_images = 0
    missing_images = 0

    while sample_token:

        sample = nusc.get(
            "sample",
            sample_token
        )

        if CAMERA_CHANNEL not in sample["data"]:
            print(
                f"Frame {frame_number:03d}: "
                f"{CAMERA_CHANNEL} token was not found."
            )

            missing_images += 1
            sample_token = sample["next"]
            frame_number += 1
            continue

        camera_token = sample["data"][CAMERA_CHANNEL]

        sample_data = nusc.get(
            "sample_data",
            camera_token
        )

        source_image = (
            NUSCENES_ROOT /
            sample_data["filename"]
        )

        output_image_name = (
            f"{scene_index:02d}_"
            f"{scene_name}_"
            f"frame_{frame_number:03d}.jpg"
        )

        output_image = (
            scene_output_folder /
            output_image_name
        )

        if source_image.exists():

            shutil.copy2(
                source_image,
                output_image
            )

            print(
                f"Frame {frame_number:03d}: "
                f"copied -> {output_image_name}"
            )

            copied_images += 1

        else:

            print(
                f"Frame {frame_number:03d}: "
                f"image file was not found:\n"
                f"{source_image}"
            )

            missing_images += 1

        sample_token = sample["next"]
        frame_number += 1

    summary_rows.append(
        {
            "scene_index": scene_index,
            "scene_name": scene_name,
            "description": description,
            "expected_samples": number_of_samples,
            "copied_images": copied_images,
            "missing_images": missing_images,
            "output_folder": str(
                scene_output_folder.resolve()
            )
        }
    )

    print()
    print(
        f"Scene finished: copied={copied_images}, "
        f"missing={missing_images}"
    )


# =========================================================
# SAVE SUMMARY
# =========================================================

summary_file = OUTPUT_ROOT / "scene_summary.csv"

with open(
    summary_file,
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "scene_index",
        "scene_name",
        "description",
        "expected_samples",
        "copied_images",
        "missing_images",
        "output_folder"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(summary_rows)


print()
print("=" * 70)
print("EXTRACTION FINISHED")
print("=" * 70)
print(f"Images saved at : {OUTPUT_ROOT.resolve()}")
print(f"Summary saved at: {summary_file.resolve()}")
