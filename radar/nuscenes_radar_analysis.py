import csv
import json
import math
import struct
import webbrowser
from pathlib import Path


DATA_ROOT = Path(r"D:\nuscene")
VERSION_FOLDER = DATA_ROOT / "v1.0-mini"

SCENE_INDEX = 0

RADAR_CHANNEL = "RADAR_FRONT"

OUTPUT_FOLDER = DATA_ROOT / "radar_output"

MAX_DISPLAY_DISTANCE = 100.0


def load_json(filename):
    file_path = VERSION_FOLDER / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_dictionary(records):
    return {
        record["token"]: record
        for record in records
    }


def parse_pcd_header(file):


    header = {}

    while True:
        line = file.readline()

        if not line:
            raise ValueError(
                "PCD header is incomplete."
            )

        decoded_line = line.decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not decoded_line:
            continue

        if decoded_line.startswith("#"):
            continue

        parts = decoded_line.split()

        key = parts[0].upper()
        values = parts[1:]

        header[key] = values

        if key == "DATA":
            break

    return header


def get_struct_character(type_name, size):


    type_name = type_name.upper()

    mapping = {
        ("F", 4): "f",
        ("F", 8): "d",

        ("I", 1): "b",
        ("I", 2): "h",
        ("I", 4): "i",
        ("I", 8): "q",

        ("U", 1): "B",
        ("U", 2): "H",
        ("U", 4): "I",
        ("U", 8): "Q"
    }

    key = (type_name, size)

    if key not in mapping:
        raise ValueError(
            f"Unsupported PCD field type: "
            f"TYPE={type_name}, SIZE={size}"
        )

    return mapping[key]


def read_pcd_file(file_path):
  

    points = []

    with open(file_path, "rb") as file:
        header = parse_pcd_header(file)

        fields = header.get("FIELDS", [])
        sizes = [
            int(value)
            for value in header.get("SIZE", [])
        ]

        types = header.get("TYPE", [])

        counts = [
            int(value)
            for value in header.get(
                "COUNT",
                ["1"] * len(fields)
            )
        ]

        data_format = header.get(
            "DATA",
            ["binary"]
        )[0].lower()

        point_count = int(
            header.get(
                "POINTS",
                header.get("WIDTH", ["0"])
            )[0]
        )

        if not fields:
            raise ValueError(
                f"No FIELDS found in PCD file:\n{file_path}"
            )

        if not (
            len(fields)
            == len(sizes)
            == len(types)
            == len(counts)
        ):
            raise ValueError(
                "PCD FIELDS, SIZE, TYPE and COUNT "
                "do not have the same length."
            )

        expanded_fields = []
        struct_format = "<"

        for field, size, type_name, count in zip(
            fields,
            sizes,
            types,
            counts
        ):
            character = get_struct_character(
                type_name,
                size
            )

            for count_index in range(count):
                if count == 1:
                    expanded_name = field
                else:
                    expanded_name = (
                        f"{field}_{count_index}"
                    )

                expanded_fields.append(expanded_name)
                struct_format += character

        if data_format == "binary":
            record_struct = struct.Struct(struct_format)
            record_size = record_struct.size

            for _ in range(point_count):
                raw_data = file.read(record_size)

                if len(raw_data) != record_size:
                    break

                values = record_struct.unpack(raw_data)

                point = dict(
                    zip(expanded_fields, values)
                )

                points.append(point)

        elif data_format == "ascii":
            for raw_line in file:
                decoded_line = raw_line.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if not decoded_line:
                    continue

                values = decoded_line.split()

                if len(values) < len(expanded_fields):
                    continue

                point = {}

                for field, value in zip(
                    expanded_fields,
                    values
                ):
                    try:
                        point[field] = float(value)
                    except ValueError:
                        point[field] = value

                points.append(point)

        elif data_format == "binary_compressed":
            raise ValueError(
                "binary_compressed PCD is not supported "
                "by this version of the code."
            )

        else:
            raise ValueError(
                f"Unknown PCD data format: {data_format}"
            )

    return points



def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_point_information(point):

    x = safe_float(point.get("x", 0.0))
    y = safe_float(point.get("y", 0.0))
    z = safe_float(point.get("z", 0.0))

    vx = safe_float(point.get("vx", 0.0))
    vy = safe_float(point.get("vy", 0.0))

    vx_comp = safe_float(
        point.get("vx_comp", vx)
    )

    vy_comp = safe_float(
        point.get("vy_comp", vy)
    )

    rcs = safe_float(point.get("rcs", 0.0))

    distance = math.sqrt(
        x ** 2 + y ** 2 + z ** 2
    )

    planar_distance = math.sqrt(
        x ** 2 + y ** 2
    )

    raw_speed = math.sqrt(
        vx ** 2 + vy ** 2
    )

    compensated_speed = math.sqrt(
        vx_comp ** 2 + vy_comp ** 2
    )

    return {
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vx_comp": vx_comp,
        "vy_comp": vy_comp,
        "rcs": rcs,
        "distance": distance,
        "planar_distance": planar_distance,
        "raw_speed": raw_speed,
        "compensated_speed": compensated_speed,
        "dyn_prop": int(
            safe_float(point.get("dyn_prop", 0))
        ),
        "invalid_state": int(
            safe_float(point.get("invalid_state", 0))
        ),
        "ambig_state": int(
            safe_float(point.get("ambig_state", 0))
        )
    }


def analyse_radar_frame(points):


    processed_points = []

    for point in points:
        information = calculate_point_information(
            point
        )

        if (
            information["planar_distance"]
            <= MAX_DISPLAY_DISTANCE
        ):
            processed_points.append(information)

    if not processed_points:
        return {
            "points": [],
            "point_count": 0,
            "mean_distance": 0.0,
            "max_distance": 0.0,
            "mean_speed": 0.0,
            "max_speed": 0.0,
            "mean_rcs": 0.0,
            "moving_points": 0,
            "stationary_points": 0
        }

    distances = [
        point["planar_distance"]
        for point in processed_points
    ]

    speeds = [
        point["compensated_speed"]
        for point in processed_points
    ]

    rcs_values = [
        point["rcs"]
        for point in processed_points
    ]

    moving_points = sum(
        1
        for speed in speeds
        if speed > 0.5
    )

    stationary_points = (
        len(processed_points) - moving_points
    )

    return {
        "points": processed_points,
        "point_count": len(processed_points),
        "mean_distance": (
            sum(distances) / len(distances)
        ),
        "max_distance": max(distances),
        "mean_speed": (
            sum(speeds) / len(speeds)
        ),
        "max_speed": max(speeds),
        "mean_rcs": (
            sum(rcs_values) / len(rcs_values)
        ),
        "moving_points": moving_points,
        "stationary_points": stationary_points
    }


def main():
    print("=" * 65)
    print("nuScenes Radar Scenario Analysis")
    print("=" * 65)

    print("\nLoading metadata...")

    scenes = load_json("scene.json")
    samples = load_json("sample.json")
    sample_data = load_json("sample_data.json")
    logs = load_json("log.json")

    sample_dictionary = create_dictionary(samples)
    log_dictionary = create_dictionary(logs)

    if not 0 <= SCENE_INDEX < len(scenes):
        raise ValueError(
            f"SCENE_INDEX must be between "
            f"0 and {len(scenes) - 1}."
        )

    selected_scene = scenes[SCENE_INDEX]

    scene_name = selected_scene.get(
        "name",
        "Unknown"
    )

    scene_description = selected_scene.get(
        "description",
        "No description"
    )

    log_token = selected_scene.get("log_token")

    location = log_dictionary.get(
        log_token,
        {}
    ).get(
        "location",
        "Unknown"
    )

    print("\nSelected scenario")
    print("-----------------")
    print("Scene index:", SCENE_INDEX)
    print("Scene name:", scene_name)
    print("Location:", location)
    print("Description:", scene_description)
    print(
        "Number of samples:",
        selected_scene.get("nbr_samples")
    )


    radar_by_sample = {}

    for record in sample_data:
        filename = record.get(
            "filename",
            ""
        ).replace("\\", "/")

        if RADAR_CHANNEL not in filename:
            continue

        if not record.get("is_key_frame", False):
            continue

        sample_token = record.get("sample_token")

        if sample_token:
            radar_by_sample[sample_token] = record

    print(
        f"\nNumber of {RADAR_CHANNEL} key frames found:",
        len(radar_by_sample)
    )

    if len(radar_by_sample) == 0:
        print(
            f"\nNo {RADAR_CHANNEL} files were found."
        )

        print(
            "\nSome radar-related paths "
            "from sample_data.json:"
        )

        shown = 0

        for record in sample_data:
            filename = record.get(
                "filename",
                ""
            )

            if "RADAR" in filename.upper():
                print(filename)
                shown += 1

            if shown >= 20:
                break

        return

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    current_sample_token = selected_scene[
        "first_sample_token"
    ]

    frame_number = 1
    frame_results = []

    total_points = 0
    total_moving_points = 0
    total_stationary_points = 0

    while current_sample_token:
        sample = sample_dictionary.get(
            current_sample_token
        )

        if sample is None:
            print(
                "Sample metadata was not found:",
                current_sample_token
            )
            break

        radar_record = radar_by_sample.get(
            current_sample_token
        )

        if radar_record is None:
            print(
                f"Frame {frame_number:03d}: "
                f"{RADAR_CHANNEL} not found"
            )

            current_sample_token = sample.get(
                "next",
                ""
            )

            frame_number += 1
            continue

        radar_filename = radar_record.get(
            "filename",
            ""
        )

        radar_path = DATA_ROOT / Path(
            radar_filename
        )

        if not radar_path.exists():
            print(
                f"Frame {frame_number:03d}: "
                "radar file does not exist"
            )

            print(radar_path)

            current_sample_token = sample.get(
                "next",
                ""
            )

            frame_number += 1
            continue

        try:
            raw_points = read_pcd_file(
                radar_path
            )

            analysis = analyse_radar_frame(
                raw_points
            )

        except Exception as error:
            print(
                f"Frame {frame_number:03d}: "
                f"could not read radar file"
            )

            print("Error:", error)

            current_sample_token = sample.get(
                "next",
                ""
            )

            frame_number += 1
            continue

        result = {
            "frame": frame_number,
            "timestamp": sample.get(
                "timestamp",
                0
            ),
            "radar_file": radar_filename,
            "point_count": analysis[
                "point_count"
            ],
            "moving_points": analysis[
                "moving_points"
            ],
            "stationary_points": analysis[
                "stationary_points"
            ],
            "mean_distance": round(
                analysis["mean_distance"],
                3
            ),
            "max_distance": round(
                analysis["max_distance"],
                3
            ),
            "mean_speed": round(
                analysis["mean_speed"],
                3
            ),
            "max_speed": round(
                analysis["max_speed"],
                3
            ),
            "mean_rcs": round(
                analysis["mean_rcs"],
                3
            ),
            "points": []
        }

        for point in analysis["points"]:
            result["points"].append({
                "x": round(point["x"], 3),
                "y": round(point["y"], 3),
                "speed": round(
                    point["compensated_speed"],
                    3
                ),
                "rcs": round(
                    point["rcs"],
                    3
                ),
                "distance": round(
                    point["planar_distance"],
                    3
                )
            })

        frame_results.append(result)

        total_points += result["point_count"]
        total_moving_points += result[
            "moving_points"
        ]
        total_stationary_points += result[
            "stationary_points"
        ]

        print(
            f"Frame {frame_number:03d}: "
            f"{result['point_count']} radar points | "
            f"moving: {result['moving_points']} | "
            f"stationary: "
            f"{result['stationary_points']}"
        )

        current_sample_token = sample.get(
            "next",
            ""
        )

        frame_number += 1

    if not frame_results:
        print(
            "\nNo valid radar frames were processed."
        )
        return

    csv_path = (
        OUTPUT_FOLDER
        / "radar_statistics.csv"
    )

    csv_fields = [
        "frame",
        "timestamp",
        "point_count",
        "moving_points",
        "stationary_points",
        "mean_distance",
        "max_distance",
        "mean_speed",
        "max_speed",
        "mean_rcs",
        "radar_file"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(frame_results)

    table_rows = ""

    for result in frame_results:
        table_rows += f"""
        <tr>
            <td>{result["frame"]}</td>
            <td>{result["point_count"]}</td>
            <td>{result["moving_points"]}</td>
            <td>{result["stationary_points"]}</td>
            <td>{result["mean_distance"]:.2f}</td>
            <td>{result["max_distance"]:.2f}</td>
            <td>{result["mean_speed"]:.2f}</td>
            <td>{result["max_speed"]:.2f}</td>
            <td>{result["mean_rcs"]:.2f}</td>
        </tr>
        """

    html_frames = []

    for result in frame_results:
        html_frames.append({
            "frame": result["frame"],
            "point_count": result["point_count"],
            "moving_points": result["moving_points"],
            "stationary_points": (
                result["stationary_points"]
            ),
            "mean_distance": result[
                "mean_distance"
            ],
            "max_distance": result[
                "max_distance"
            ],
            "mean_speed": result[
                "mean_speed"
            ],
            "max_speed": result[
                "max_speed"
            ],
            "mean_rcs": result["mean_rcs"],
            "points": result["points"]
        })

    frames_json = json.dumps(
        html_frames,
        ensure_ascii=False
    )

    average_points_per_frame = (
        total_points / len(frame_results)
    )

    html_content = f"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>nuScenes Radar Analysis</title>

    <style>
        body {{
            margin: 0;
            background: #eef2f7;
            font-family: Arial, sans-serif;
            color: #1f2937;
        }}

        .container {{
            max-width: 1200px;
            margin: 25px auto;
            padding: 20px;
        }}

        .section {{
            background: white;
            margin-bottom: 20px;
            padding: 22px;
            border-radius: 12px;
            box-shadow:
                0 3px 12px rgba(0, 0, 0, 0.08);
        }}

        .header {{
            background: #172554;
            color: white;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(170px, 1fr));
            gap: 14px;
        }}

        .summary-card {{
            background: #eff6ff;
            border-left: 5px solid #2563eb;
            padding: 16px;
            border-radius: 8px;
        }}

        .summary-card strong {{
            display: block;
            margin-bottom: 8px;
        }}

        .summary-card span {{
            font-size: 24px;
            font-weight: bold;
        }}

        .radar-container {{
            text-align: center;
        }}

        canvas {{
            width: 100%;
            max-width: 900px;
            background: #07111f;
            border-radius: 12px;
            border: 1px solid #334155;
        }}

        button {{
            padding: 11px 20px;
            margin: 7px;
            border: none;
            border-radius: 7px;
            background: #2563eb;
            color: white;
            font-size: 15px;
            cursor: pointer;
        }}

        button:hover {{
            background: #1d4ed8;
        }}

        .legend {{
            margin: 15px auto;
            line-height: 1.8;
        }}

        .legend-item {{
            display: inline-block;
            margin: 0 15px;
        }}

        .dot {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }}

        .slow {{
            background: #38bdf8;
        }}

        .medium {{
            background: #facc15;
        }}

        .fast {{
            background: #ef4444;
        }}

        .frame-information {{
            margin-top: 15px;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            line-height: 1.8;
        }}

        .table-container {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            border: 1px solid #d1d5db;
            padding: 9px;
            text-align: center;
        }}

        th {{
            background: #172554;
            color: white;
        }}

        tr:nth-child(even) {{
            background: #f8fafc;
        }}

        .description {{
            line-height: 1.7;
        }}
    </style>
</head>

<body>

<div class="container">

    <div class="section header">
        <h1>nuScenes Radar Scenario Analysis</h1>

        <p>
            <strong>Scene:</strong>
            {scene_name}
        </p>

        <p>
            <strong>Location:</strong>
            {location}
        </p>

        <p>
            <strong>Radar:</strong>
            {RADAR_CHANNEL}
        </p>

        <p>
            <strong>Description:</strong>
            {scene_description}
        </p>
    </div>

    <div class="section description">
        <h2>Radar Coordinate System</h2>

        <p>
            In the radar sensor coordinate system,
            the horizontal axis represents lateral position
            and the vertical axis represents forward distance.
            The vehicle is located at the bottom center
            of the plot.
        </p>

        <p>
            Point color represents compensated radar velocity:
            blue points are nearly stationary, yellow points
            have medium velocity, and red points have higher
            velocity.
        </p>
    </div>

    <div class="section">
        <h2>Scenario Summary</h2>

        <div class="summary-grid">

            <div class="summary-card">
                <strong>Processed frames</strong>
                <span>{len(frame_results)}</span>
            </div>

            <div class="summary-card">
                <strong>Total radar points</strong>
                <span>{total_points}</span>
            </div>

            <div class="summary-card">
                <strong>Average points/frame</strong>
                <span>
                    {average_points_per_frame:.1f}
                </span>
            </div>

            <div class="summary-card">
                <strong>Moving detections</strong>
                <span>{total_moving_points}</span>
            </div>

            <div class="summary-card">
                <strong>Stationary detections</strong>
                <span>{total_stationary_points}</span>
            </div>

        </div>
    </div>

    <div class="section radar-container">

        <h2>RADAR_FRONT Playback</h2>

        <canvas
            id="radarCanvas"
            width="900"
            height="650"
        ></canvas>

        <div class="legend">
            <span class="legend-item">
                <span class="dot slow"></span>
                Speed ≤ 0.5 m/s
            </span>

            <span class="legend-item">
                <span class="dot medium"></span>
                0.5–3 m/s
            </span>

            <span class="legend-item">
                <span class="dot fast"></span>
                Speed &gt; 3 m/s
            </span>
        </div>

        <div>
            <button onclick="previousFrame()">
                Previous
            </button>

            <button
                id="playButton"
                onclick="togglePlayback()"
            >
                Play
            </button>

            <button onclick="nextFrame()">
                Next
            </button>
        </div>

        <div
            id="frameInformation"
            class="frame-information"
        ></div>

    </div>

    <div class="section">

        <h2>Frame-by-Frame Radar Statistics</h2>

        <div class="table-container">
            <table>

                <thead>
                    <tr>
                        <th>Frame</th>
                        <th>Points</th>
                        <th>Moving</th>
                        <th>Stationary</th>
                        <th>Mean distance (m)</th>
                        <th>Max distance (m)</th>
                        <th>Mean speed (m/s)</th>
                        <th>Max speed (m/s)</th>
                        <th>Mean RCS (dB)</th>
                    </tr>
                </thead>

                <tbody>
                    {table_rows}
                </tbody>

            </table>
        </div>

    </div>

</div>

<script>

    const frames = {frames_json};

    const canvas =
        document.getElementById("radarCanvas");

    const context =
        canvas.getContext("2d");

    let currentFrameIndex = 0;
    let playbackTimer = null;

    const maximumDistance =
        {MAX_DISPLAY_DISTANCE};

    const lateralLimit = 50;

    function mapX(lateralPosition) {{
        return (
            canvas.width / 2
            - lateralPosition
            / lateralLimit
            * canvas.width / 2
        );
    }}

    function mapY(forwardPosition) {{
        return (
            canvas.height
            - 40
            - forwardPosition
            / maximumDistance
            * (canvas.height - 80)
        );
    }}

    function drawGrid() {{

        context.fillStyle = "#07111f";

        context.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        context.strokeStyle = "#334155";
        context.lineWidth = 1;

        context.fillStyle = "#94a3b8";
        context.font = "13px Arial";

        for (
            let distance = 0;
            distance <= maximumDistance;
            distance += 20
        ) {{
            const y = mapY(distance);

            context.beginPath();

            context.moveTo(40, y);
            context.lineTo(canvas.width - 40, y);

            context.stroke();

            context.fillText(
                distance + " m",
                5,
                y + 4
            );
        }}

        for (
            let lateral = -40;
            lateral <= 40;
            lateral += 10
        ) {{
            const x = mapX(lateral);

            context.beginPath();

            context.moveTo(x, 20);
            context.lineTo(x, canvas.height - 40);

            context.stroke();

            context.fillText(
                lateral + " m",
                x - 15,
                canvas.height - 15
            );
        }}

        // محور مرکزی خودرو
        context.strokeStyle = "#64748b";
        context.lineWidth = 2;

        context.beginPath();

        context.moveTo(
            canvas.width / 2,
            20
        );

        context.lineTo(
            canvas.width / 2,
            canvas.height - 40
        );

        context.stroke();

        // نمایش خودروی ego
        context.fillStyle = "#ffffff";

        context.fillRect(
            canvas.width / 2 - 12,
            canvas.height - 38,
            24,
            30
        );

        context.fillStyle = "#cbd5e1";
        context.fillText(
            "Ego vehicle",
            canvas.width / 2 - 35,
            canvas.height - 45
        );
    }}

    function getPointColor(speed) {{

        if (speed <= 0.5) {{
            return "#38bdf8";
        }}

        if (speed <= 3.0) {{
            return "#facc15";
        }}

        return "#ef4444";
    }}

    function drawRadarFrame() {{

        if (frames.length === 0) {{
            return;
        }}

        drawGrid();

        const frame =
            frames[currentFrameIndex];

        for (const point of frame.points) {{

            const canvasX = mapX(point.y);
            const canvasY = mapY(point.x);

            if (
                canvasX < 0
                || canvasX > canvas.width
                || canvasY < 0
                || canvasY > canvas.height
            ) {{
                continue;
            }}

            context.fillStyle =
                getPointColor(point.speed);

            let radius =
                3 + Math.max(
                    0,
                    Math.min(4, point.rcs / 10)
                );

            context.beginPath();

            context.arc(
                canvasX,
                canvasY,
                radius,
                0,
                2 * Math.PI
            );

            context.fill();
        }}

        document.getElementById(
            "frameInformation"
        ).innerHTML = `
            <strong>Frame:</strong>
            ${{frame.frame}} of ${{frames.length}}
            <br>

            <strong>Radar points:</strong>
            ${{frame.point_count}}
            |
            <strong>Moving:</strong>
            ${{frame.moving_points}}
            |
            <strong>Stationary:</strong>
            ${{frame.stationary_points}}
            <br>

            <strong>Mean distance:</strong>
            ${{frame.mean_distance.toFixed(2)}} m
            |
            <strong>Maximum distance:</strong>
            ${{frame.max_distance.toFixed(2)}} m
            <br>

            <strong>Mean compensated speed:</strong>
            ${{frame.mean_speed.toFixed(2)}} m/s
            |
            <strong>Maximum compensated speed:</strong>
            ${{frame.max_speed.toFixed(2)}} m/s
            <br>

            <strong>Mean RCS:</strong>
            ${{frame.mean_rcs.toFixed(2)}} dB
        `;
    }}

    function nextFrame() {{

        currentFrameIndex =
            (currentFrameIndex + 1)
            % frames.length;

        drawRadarFrame();
    }}

    function previousFrame() {{

        currentFrameIndex =
            (
                currentFrameIndex
                - 1
                + frames.length
            )
            % frames.length;

        drawRadarFrame();
    }}

    function togglePlayback() {{

        const button =
            document.getElementById(
                "playButton"
            );

        if (playbackTimer === null) {{

            playbackTimer = setInterval(
                nextFrame,
                500
            );

            button.innerText = "Pause";

        }} else {{

            clearInterval(playbackTimer);

            playbackTimer = null;
            button.innerText = "Play";
        }}
    }}

    drawRadarFrame();

</script>

</body>
</html>
"""

    html_path = (
        OUTPUT_FOLDER
        / "index.html"
    )

    with open(
        html_path,
        "w",
        encoding="utf-8"
    ) as html_file:
        html_file.write(html_content)

    report_path = (
        OUTPUT_FOLDER
        / "radar_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(
            "nuScenes Radar Scenario Analysis\n"
        )

        report_file.write(
            "=" * 50 + "\n\n"
        )

        report_file.write(
            f"Scene: {scene_name}\n"
        )

        report_file.write(
            f"Location: {location}\n"
        )

        report_file.write(
            f"Radar channel: {RADAR_CHANNEL}\n"
        )

        report_file.write(
            f"Description: {scene_description}\n"
        )

        report_file.write(
            f"Processed frames: "
            f"{len(frame_results)}\n"
        )

        report_file.write(
            f"Total radar points: "
            f"{total_points}\n"
        )

        report_file.write(
            f"Average points per frame: "
            f"{average_points_per_frame:.2f}\n"
        )

        report_file.write(
            f"Moving detections: "
            f"{total_moving_points}\n"
        )

        report_file.write(
            f"Stationary detections: "
            f"{total_stationary_points}\n"
        )

    print("\n" + "=" * 65)
    print("Radar project completed successfully.")
    print("=" * 65)

    print("\nGenerated files:")
    print("HTML report:", html_path)
    print("CSV data:", csv_path)
    print("Text report:", report_path)

    print("\nOpen this file:")
    print(html_path)

    webbrowser.open(
        html_path.resolve().as_uri()
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nProject execution failed.")
        print("Error:", error)

        input("\nPress Enter to close...")
