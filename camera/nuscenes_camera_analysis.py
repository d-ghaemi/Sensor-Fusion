import csv
import json
import shutil
import webbrowser
from collections import Counter
from pathlib import Path


DATA_ROOT = Path(r"D:\nuscene")
VERSION_FOLDER = DATA_ROOT / "v1.0-mini"

SCENE_INDEX = 0

OUTPUT_FOLDER = DATA_ROOT / "scenario_output"
FRAMES_FOLDER = OUTPUT_FOLDER / "frames"


def load_json(filename):
    file_path = VERSION_FOLDER / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_dictionary(records):
    return {record["token"]: record for record in records}


def get_general_category(category_name):
    name = category_name.lower()

    if "vehicle.car" in name:
        return "Car"
    if "vehicle.truck" in name:
        return "Truck"
    if "vehicle.bus" in name:
        return "Bus"
    if "vehicle.trailer" in name:
        return "Trailer"
    if "vehicle.construction" in name:
        return "Construction Vehicle"
    if "vehicle.motorcycle" in name:
        return "Motorcycle"
    if "vehicle.bicycle" in name:
        return "Bicycle"
    if "human.pedestrian" in name:
        return "Pedestrian"
    if "movable_object.barrier" in name:
        return "Barrier"
    if "movable_object.trafficcone" in name:
        return "Traffic Cone"
    if "animal" in name:
        return "Animal"

    return "Other"


def main():
    print("=" * 60)
    print("nuScenes Scenario Project")
    print("=" * 60)

    scenes = load_json("scene.json")
    samples = load_json("sample.json")
    sample_data = load_json("sample_data.json")
    annotations = load_json("sample_annotation.json")
    instances = load_json("instance.json")
    categories = load_json("category.json")
    logs = load_json("log.json")

    sample_dictionary = create_dictionary(samples)
    instance_dictionary = create_dictionary(instances)
    category_dictionary = create_dictionary(categories)
    log_dictionary = create_dictionary(logs)

    annotations_by_sample = {}

    for annotation in annotations:
        sample_token = annotation["sample_token"]

        if sample_token not in annotations_by_sample:
            annotations_by_sample[sample_token] = []

        annotations_by_sample[sample_token].append(annotation)

    front_camera_by_sample = {}

    for record in sample_data:
        filename = record.get("filename", "").replace("\\", "/")

        if "CAM_FRONT" not in filename:
            continue

        if not record.get("is_key_frame", False):
            continue

        sample_token = record.get("sample_token")

        if sample_token:
            front_camera_by_sample[sample_token] = record

    print("Number of CAM_FRONT frames found:", len(front_camera_by_sample))

    if len(front_camera_by_sample) == 0:
        print("\nNo CAM_FRONT images were found.")
        print("First 20 filenames in sample_data.json:")

        for record in sample_data[:20]:
            print(record.get("filename"))

        return

    if SCENE_INDEX >= len(scenes):
        raise ValueError("SCENE_INDEX is too large.")

    selected_scene = scenes[SCENE_INDEX]

    scene_name = selected_scene.get("name", "Unknown")
    description = selected_scene.get("description", "No description")
    log_token = selected_scene.get("log_token")
    location = log_dictionary.get(log_token, {}).get("location", "Unknown")

    print("\nSelected scenario:")
    print("Scene:", scene_name)
    print("Location:", location)
    print("Description:", description)
    print("Samples:", selected_scene.get("nbr_samples"))

    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)

    FRAMES_FOLDER.mkdir(parents=True, exist_ok=True)

    current_sample_token = selected_scene["first_sample_token"]

    frame_results = []
    total_objects = Counter()
    frame_number = 1

    while current_sample_token:
        sample = sample_dictionary.get(current_sample_token)

        if sample is None:
            break

        object_counter = Counter()

        for annotation in annotations_by_sample.get(current_sample_token, []):
            instance = instance_dictionary.get(
                annotation.get("instance_token")
            )

            if instance is None:
                continue

            category = category_dictionary.get(
                instance.get("category_token")
            )

            if category is None:
                continue

            category_name = category.get("name", "")
            general_category = get_general_category(category_name)

            object_counter[general_category] += 1

        total_objects.update(object_counter)

        camera_data = front_camera_by_sample.get(current_sample_token)

        if camera_data is None:
            print(
                f"Frame {frame_number:03d}: CAM_FRONT not found"
            )

            current_sample_token = sample.get("next", "")
            frame_number += 1
            continue

        source_filename = camera_data.get("filename", "")
        source_image = DATA_ROOT / Path(source_filename)

        if not source_image.exists():
            print(
                f"Frame {frame_number:03d}: Image file not found:"
            )
            print(source_image)

            current_sample_token = sample.get("next", "")
            frame_number += 1
            continue

        output_filename = f"frame_{frame_number:03d}{source_image.suffix}"
        destination_image = FRAMES_FOLDER / output_filename

        shutil.copy2(source_image, destination_image)

        result = {
            "frame": frame_number,
            "image": output_filename,
            "Car": object_counter.get("Car", 0),
            "Truck": object_counter.get("Truck", 0),
            "Bus": object_counter.get("Bus", 0),
            "Motorcycle": object_counter.get("Motorcycle", 0),
            "Bicycle": object_counter.get("Bicycle", 0),
            "Pedestrian": object_counter.get("Pedestrian", 0),
            "Barrier": object_counter.get("Barrier", 0),
            "Traffic Cone": object_counter.get("Traffic Cone", 0),
            "Other": object_counter.get("Other", 0),
            "Total": sum(object_counter.values())
        }

        frame_results.append(result)

        print(
            f"Frame {frame_number:03d}: "
            f"{result['Total']} objects"
        )

        current_sample_token = sample.get("next", "")
        frame_number += 1

    if len(frame_results) == 0:
        print("\nNo valid frames were copied.")
        return

    csv_path = OUTPUT_FOLDER / "scenario_statistics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "frame",
            "Car",
            "Truck",
            "Bus",
            "Motorcycle",
            "Bicycle",
            "Pedestrian",
            "Barrier",
            "Traffic Cone",
            "Other",
            "Total",
            "image"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_results)

    frames_json = json.dumps(frame_results)

    table_rows = ""

    for result in frame_results:
        table_rows += f"""
        <tr>
            <td>{result["frame"]}</td>
            <td>{result["Car"]}</td>
            <td>{result["Truck"]}</td>
            <td>{result["Bus"]}</td>
            <td>{result["Motorcycle"]}</td>
            <td>{result["Bicycle"]}</td>
            <td>{result["Pedestrian"]}</td>
            <td>{result["Barrier"]}</td>
            <td>{result["Traffic Cone"]}</td>
            <td>{result["Total"]}</td>
        </tr>
        """

    summary_cards = ""

    for category, count in total_objects.most_common():
        summary_cards += f"""
        <div class="card">
            <strong>{category}</strong>
            <span>{count}</span>
        </div>
        """

    html_content = f"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>nuScenes Scenario Project</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #eef2f7;
            margin: 0;
        }}

        .container {{
            max-width: 1150px;
            margin: 25px auto;
            padding: 20px;
        }}

        .section {{
            background: white;
            padding: 22px;
            margin-bottom: 20px;
            border-radius: 12px;
        }}

        .header {{
            background: #172554;
            color: white;
        }}

        img {{
            width: 100%;
            max-width: 1000px;
            border-radius: 10px;
        }}

        button {{
            padding: 10px 18px;
            margin: 8px;
            border: none;
            border-radius: 6px;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }}

        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }}

        .card {{
            background: #eff6ff;
            padding: 15px;
            border-radius: 8px;
        }}

        .card strong {{
            display: block;
            margin-bottom: 8px;
        }}

        .card span {{
            font-size: 22px;
            font-weight: bold;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            border: 1px solid #ccc;
            padding: 9px;
            text-align: center;
        }}

        th {{
            background: #172554;
            color: white;
        }}
    </style>
</head>

<body>

<div class="container">

    <div class="section header">
        <h1>nuScenes Scenario Analysis</h1>
        <p><strong>Scene:</strong> {scene_name}</p>
        <p><strong>Location:</strong> {location}</p>
        <p><strong>Description:</strong> {description}</p>
    </div>

    <div class="section" style="text-align:center">
        <h2>CAM_FRONT Playback</h2>

        <img id="frameImage" src="">

        <p id="frameInfo"></p>

        <button onclick="previousFrame()">Previous</button>
        <button onclick="togglePlay()" id="playButton">Play</button>
        <button onclick="nextFrame()">Next</button>
    </div>

    <div class="section">
        <h2>Total Object Statistics</h2>

        <div class="cards">
            {summary_cards}
        </div>
    </div>

    <div class="section">
        <h2>Frame Statistics</h2>

        <table>
            <tr>
                <th>Frame</th>
                <th>Car</th>
                <th>Truck</th>
                <th>Bus</th>
                <th>Motorcycle</th>
                <th>Bicycle</th>
                <th>Pedestrian</th>
                <th>Barrier</th>
                <th>Traffic Cone</th>
                <th>Total</th>
            </tr>

            {table_rows}
        </table>
    </div>

</div>

<script>
    const frames = {frames_json};

    let currentIndex = 0;
    let timer = null;

    function showFrame() {{
        const frame = frames[currentIndex];

        document.getElementById("frameImage").src =
            "frames/" + frame.image;

        document.getElementById("frameInfo").innerHTML =
            "Frame " + frame.frame +
            " | Total objects: " + frame.Total +
            " | Cars: " + frame.Car +
            " | Pedestrians: " + frame.Pedestrian;
    }}

    function nextFrame() {{
        currentIndex = (currentIndex + 1) % frames.length;
        showFrame();
    }}

    function previousFrame() {{
        currentIndex =
            (currentIndex - 1 + frames.length) % frames.length;

        showFrame();
    }}

    function togglePlay() {{
        const button = document.getElementById("playButton");

        if (timer === null) {{
            timer = setInterval(nextFrame, 500);
            button.innerText = "Pause";
        }} else {{
            clearInterval(timer);
            timer = null;
            button.innerText = "Play";
        }}
    }}

    showFrame();
</script>

</body>
</html>
"""

    html_path = OUTPUT_FOLDER / "index.html"

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print("\nProject completed successfully.")
    print("Output folder:")
    print(OUTPUT_FOLDER)
    print("\nOpen this file:")
    print(html_path)

    webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:")
        print(error)
