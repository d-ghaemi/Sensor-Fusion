from pathlib import Path
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# تنظیمات
# =========================================================
FILE_PATH = Path(__file__).resolve().parent / "adc_data.bin"

NUM_ADC_SAMPLES = 256
NUM_RX = 4
CHIRPS_PER_FRAME = 128
EXPECTED_FRAMES = 100


# =========================================================
# 1. بررسی فایل
# =========================================================
if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"فایل پیدا نشد:\n{FILE_PATH}"
    )


# =========================================================
# 2. خواندن فایل به صورت int16
# =========================================================
raw = np.fromfile(
    FILE_PATH,
    dtype=np.int16,
)

print("=" * 60)
print("File:", FILE_PATH.name)
print(
    f"File size: "
    f"{FILE_PATH.stat().st_size / 1024**2:.3f} MiB"
)
print(f"Number of int16 values: {raw.size:,}")
print(f"Minimum value: {raw.min()}")
print(f"Maximum value: {raw.max()}")
print("=" * 60)


# =========================================================
# 3. بررسی اشباع
# =========================================================
positive_saturation = np.count_nonzero(raw == 32767)
negative_saturation = np.count_nonzero(raw == -32768)

print("Positive saturation samples:", positive_saturation)
print("Negative saturation samples:", negative_saturation)

if positive_saturation or negative_saturation:
    print("WARNING: Possible ADC saturation.")
else:
    print("No exact int16 saturation was detected.")


# =========================================================
# 4. بازسازی داده مختلط I/Q
#
# ترتیب فرضی متداول DCA1000:
# I0, I1, Q0, Q1, I2, I3, Q2, Q3, ...
# =========================================================
if raw.size % 4 != 0:
    raise ValueError(
        "تعداد داده‌های int16 مضرب 4 نیست."
    )

complex_data = np.empty(
    raw.size // 2,
    dtype=np.complex64,
)

complex_data[0::2] = (
    raw[0::4].astype(np.float32)
    + 1j * raw[2::4].astype(np.float32)
)

complex_data[1::2] = (
    raw[1::4].astype(np.float32)
    + 1j * raw[3::4].astype(np.float32)
)

print(f"Complex samples: {complex_data.size:,}")
print(
    f"Mean of real part: "
    f"{complex_data.real.mean():.3f}"
)
print(
    f"Mean of imaginary part: "
    f"{complex_data.imag.mean():.3f}"
)


# =========================================================
# 5. محاسبه تعداد Chirp
# =========================================================
complex_values_per_chirp = (
    NUM_ADC_SAMPLES * NUM_RX
)

remainder = (
    complex_data.size
    % complex_values_per_chirp
)

if remainder != 0:
    raise ValueError(
        "داده به Chirpهای کامل تقسیم نمی‌شود.\n"
        f"Remainder = {remainder}"
    )

total_chirps = (
    complex_data.size
    // complex_values_per_chirp
)

print(
    "Complex values per chirp:",
    complex_values_per_chirp,
)
print(
    f"Total chirps calculated from file: "
    f"{total_chirps:,}"
)


# =========================================================
# 6. محاسبه تعداد Frame
# =========================================================
if total_chirps % CHIRPS_PER_FRAME != 0:
    raise ValueError(
        "تعداد Chirpها به Frameهای کامل تقسیم نمی‌شود."
    )

num_frames = (
    total_chirps
    // CHIRPS_PER_FRAME
)

print("Calculated number of frames:", num_frames)

if num_frames != EXPECTED_FRAMES:
    print(
        "WARNING: Calculated frame count differs "
        "from EXPECTED_FRAMES."
    )


# =========================================================
# 7. تشکیل آرایه ADC
#
# قالب اولیه:
# [chirp, RX, ADC sample]
#
# قالب نهایی:
# [chirp, ADC sample, RX]
# =========================================================
adc_data = complex_data.reshape(
    total_chirps,
    NUM_RX,
    NUM_ADC_SAMPLES,
)

adc_data = adc_data.transpose(0, 2, 1)

print("Corrected ADC array shape:", adc_data.shape)


# =========================================================
# 8. آمار RXها
# =========================================================
print("\nRX channel statistics:")

for rx in range(NUM_RX):
    channel = adc_data[:, :, rx]

    channel_mean = np.mean(channel)

    channel_rms = np.sqrt(
        np.mean(np.abs(channel) ** 2)
    )

    maximum = np.max(np.abs(channel))

    print(
        f"RX{rx + 1}: "
        f"mean={channel_mean:.3f}, "
        f"RMS={channel_rms:.3f}, "
        f"max_abs={maximum:.3f}"
    )


# =========================================================
# 9. هم‌بستگی کانال‌ها
# =========================================================
chirp_id = 0

rx_signals = adc_data[
    chirp_id,
    :,
    :,
]

correlation_i = np.corrcoef(
    rx_signals.real.T
)

correlation_q = np.corrcoef(
    rx_signals.imag.T
)

print("\nCorrelation matrix of I signals:")
print(np.round(correlation_i, 3))

print("\nCorrelation matrix of Q signals:")
print(np.round(correlation_q, 3))


# =========================================================
# 10. رسم چهار RX
# =========================================================
fig, axes = plt.subplots(
    NUM_RX,
    1,
    figsize=(11, 9),
    sharex=True,
)

for rx in range(NUM_RX):
    signal = adc_data[
        chirp_id,
        :,
        rx,
    ]

    axes[rx].plot(
        signal.real,
        label="I",
        linewidth=1,
    )

    axes[rx].plot(
        signal.imag,
        label="Q",
        linewidth=1,
    )

    axes[rx].set_title(f"RX{rx + 1}")
    axes[rx].set_ylabel("ADC")
    axes[rx].grid(True, alpha=0.3)
    axes[rx].legend(loc="upper right")

axes[-1].set_xlabel("ADC sample")

fig.suptitle(
    f"Corrected raw I/Q data - Chirp {chirp_id}",
    fontsize=14,
)

plt.tight_layout()

output_path = (
    FILE_PATH.parent
    / "step1_corrected_raw_iq.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print("\nCorrected plot saved at:")
print(output_path)
print("\nSTEP 1B FINISHED.")
# =========================================================
# STEP 2: RANGE FFT
# =========================================================

print("\n" + "=" * 60)
print("STEP 2: RANGE FFT")
print("=" * 60)


# =========================================================
# پارامترهای رادار
# =========================================================
SAMPLE_RATE = 10e6              # Hz
FREQUENCY_SLOPE = 29.9817e12    # Hz/s
SPEED_OF_LIGHT = 299_792_458    # m/s

NUM_RANGE_FFT = 256
FRAME_ID = 0


# =========================================================
# 1. انتخاب یک Frame
#
# adc_data:
# [all chirps, ADC samples, RX]
# =========================================================
first_chirp = FRAME_ID * CHIRPS_PER_FRAME
last_chirp = first_chirp + CHIRPS_PER_FRAME

frame_data = adc_data[
    first_chirp:last_chirp,
    :,
    :,
]

print("Selected frame shape:", frame_data.shape)

# =========================================================
# 2. حذف DC از هر Chirp و هر RX
# =========================================================
frame_dc_removed = (
    frame_data
    - np.mean(
        frame_data,
        axis=1,
        keepdims=True,
    )
)

print(
    "Mean magnitude before DC removal:",
    np.abs(np.mean(frame_data)),
)

print(
    "Mean magnitude after DC removal:",
    np.abs(np.mean(frame_dc_removed)),
)

# =========================================================
# 3. اعمال پنجره Hann روی ADC samples
# =========================================================
range_window = np.hanning(
    NUM_ADC_SAMPLES
).astype(np.float32)

windowed_frame = (
    frame_dc_removed
    * range_window[None, :, None]
)

# =========================================================
# 4. اجرای Range FFT
#
# axis=1 همان محور ADC samples است.
# =========================================================
range_fft = np.fft.fft(
    windowed_frame,
    n=NUM_RANGE_FFT,
    axis=1,
)

print("Range FFT shape:", range_fft.shape)

# =========================================================
# 5. محاسبه محور فاصله
# =========================================================
range_bin_spacing = (
    SPEED_OF_LIGHT
    * SAMPLE_RATE
    / (
        2
        * FREQUENCY_SLOPE
        * NUM_RANGE_FFT
    )
)

range_axis = (
    np.arange(NUM_RANGE_FFT)
    * range_bin_spacing
)

print(
    f"Range-bin spacing: "
    f"{range_bin_spacing:.4f} m"
)

print(
    f"Maximum displayed range: "
    f"{range_axis[-1]:.2f} m"
)
# =========================================================
# 6. محاسبه توان Range FFT
# =========================================================
range_power = np.abs(range_fft) ** 2
range_profile = np.mean(
    range_power,
    axis=(0, 2),
)

print("Range profile shape:", range_profile.shape)
range_profile_db = 10 * np.log10(
    range_profile + 1e-12
)

range_profile_normalized_db = (
    range_profile_db
    - np.max(range_profile_db)
)
# =========================================================
# 7. پیدا کردن قله‌های مستقل Range Profile
# =========================================================
MIN_RANGE = 0.4
MAX_RANGE = 10.0

valid_mask = (
    (range_axis >= MIN_RANGE)
    & (range_axis <= MAX_RANGE)
)

valid_indices = np.where(valid_mask)[0]
valid_profile_db = range_profile_normalized_db[valid_mask]

peak_local_indices, peak_properties = find_peaks(
    valid_profile_db,
    prominence=4.0,
    distance=2,
    height=-35,
)

peak_bins = valid_indices[peak_local_indices]

# مرتب‌سازی قله‌ها بر اساس توان
peak_bins = peak_bins[
    np.argsort(
        range_profile_normalized_db[peak_bins]
    )[::-1]
]

print("\nIndependent range peaks:")

for number, range_bin in enumerate(
    peak_bins,
    start=1,
):
    print(
        f"{number:2d}. "
        f"bin={range_bin:3d}, "
        f"range={range_axis[range_bin]:7.3f} m, "
        f"level="
        f"{range_profile_normalized_db[range_bin]:7.2f} dB"
    )

    # =========================================================
# 8. رسم Range Profile از صفر تا 50 متر
# =========================================================
fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 8),
)

axes[0].plot(
    range_axis,
    range_profile_normalized_db,
    linewidth=1.2,
    color="navy",
)

axes[0].set_title(
    f"Range profile - Frame {FRAME_ID}"
)

axes[0].set_xlabel("Range (m)")
axes[0].set_ylabel("Normalized power (dB)")
axes[0].set_xlim(
    0,
    range_axis[-1],
)
axes[0].set_ylim(-80, 5)
axes[0].grid(True, alpha=0.3)


# نمایش بزرگ‌نمایی فاصله صفر تا 10 متر
axes[1].plot(
    range_axis,
    range_profile_normalized_db,
    linewidth=1.2,
    color="darkred",
)

axes[1].set_title(
    "Range profile - Zoomed view"
)

axes[1].set_xlabel("Range (m)")
axes[1].set_ylabel("Normalized power (dB)")
axes[1].set_xlim(0, 10)
axes[1].set_ylim(-80, 5)
axes[1].grid(True, alpha=0.3)


# مشخص کردن قله‌های قوی روی نمودار
for range_bin in peak_bins:
    if range_axis[range_bin] <= 10:
        axes[1].plot(
            range_axis[range_bin],
            range_profile_normalized_db[range_bin],
            marker="o",
            color="black",
            markersize=4,
        )

        axes[1].annotate(
            f"{range_axis[range_bin]:.2f} m",
            (
                range_axis[range_bin],
                range_profile_normalized_db[range_bin],
            ),
            textcoords="offset points",
            xytext=(3, 5),
            fontsize=8,
        )

plt.tight_layout()

range_plot_path = (
    FILE_PATH.parent
    / "step2_range_profile.png"
)

plt.savefig(
    range_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print("\nRange profile saved at:")
print(range_plot_path)

print("\nSTEP 2 FINISHED.")
# =========================================================
# STEP 3: DOPPLER FFT
# =========================================================

print("\n" + "=" * 60)
print("STEP 3: DOPPLER FFT")
print("=" * 60)


# =========================================================
# پارامترهای Doppler
# =========================================================
START_FREQUENCY = 77e9
IDLE_TIME = 100e-6
RAMP_END_TIME = 60e-6

CHIRP_PERIOD = IDLE_TIME + RAMP_END_TIME

NUM_DOPPLER_FFT = CHIRPS_PER_FRAME

wavelength = (
    SPEED_OF_LIGHT
    / START_FREQUENCY
)

print(
    f"Wavelength: "
    f"{wavelength * 1000:.4f} mm"
)

print(
    f"Chirp period: "
    f"{CHIRP_PERIOD * 1e6:.2f} us"
)
# =========================================================
# STEP 3: DOPPLER FFT
# =========================================================

print("\n" + "=" * 60)
print("STEP 3: DOPPLER FFT")
print("=" * 60)


# =========================================================
# 1. پارامترهای واقعی استخراج‌شده از Log
# =========================================================
START_FREQUENCY = 77e9       # Hz
IDLE_TIME = 100e-6           # s
RAMP_END_TIME = 60e-6        # s

CHIRP_PERIOD = IDLE_TIME + RAMP_END_TIME

NUM_DOPPLER_FFT = CHIRPS_PER_FRAME

wavelength = (
    SPEED_OF_LIGHT
    / START_FREQUENCY
)

print(
    f"Wavelength: "
    f"{wavelength * 1000:.4f} mm"
)

print(
    f"Chirp period: "
    f"{CHIRP_PERIOD * 1e6:.2f} us"
)


# =========================================================
# 2. بررسی ابعاد Range FFT
#
# range_fft:
# [chirp, range bin, RX]
# =========================================================
print(
    "Input Range FFT shape:",
    range_fft.shape,
)

if range_fft.shape != (
    CHIRPS_PER_FRAME,
    NUM_RANGE_FFT,
    NUM_RX,
):
    raise ValueError(
        "ابعاد range_fft با ابعاد مورد انتظار "
        "مطابقت ندارد."
    )


# =========================================================
# 3. پنجره Hann روی محور Chirp
# =========================================================
doppler_window = np.hanning(
    CHIRPS_PER_FRAME
).astype(np.float32)

windowed_range_fft = (
    range_fft
    * doppler_window[:, None, None]
)


# =========================================================
# 4. اجرای Doppler FFT
#
# axis=0 همان محور Chirp است.
# =========================================================
range_doppler_cube = np.fft.fft(
    windowed_range_fft,
    n=NUM_DOPPLER_FFT,
    axis=0,
)

range_doppler_cube = np.fft.fftshift(
    range_doppler_cube,
    axes=0,
)

print(
    "Range-Doppler cube shape:",
    range_doppler_cube.shape,
)


# =========================================================
# 5. محاسبه توان و ترکیب چهار RX
# =========================================================
rd_power = np.sum(
    np.abs(range_doppler_cube) ** 2,
    axis=2,
)

rd_map_db = 10 * np.log10(
    rd_power + 1e-12
)

rd_map_normalized_db = (
    rd_map_db
    - np.max(rd_map_db)
)

print(
    "Range-Doppler power shape:",
    rd_power.shape,
)


# =========================================================
# 6. ساخت محور سرعت
# =========================================================
doppler_frequency_axis = np.fft.fftshift(
    np.fft.fftfreq(
        NUM_DOPPLER_FFT,
        d=CHIRP_PERIOD,
    )
)

velocity_axis = (
    doppler_frequency_axis
    * wavelength
    / 2
)

velocity_bin_spacing = (
    velocity_axis[1]
    - velocity_axis[0]
)

print(
    f"Velocity-bin spacing: "
    f"{velocity_bin_spacing:.4f} m/s"
)

print(
    f"Velocity range: "
    f"{velocity_axis[0]:.3f} to "
    f"{velocity_axis[-1]:.3f} m/s"
)


# =========================================================
# 7. پیدا کردن سرعت هدف نزدیک 3 متر
# =========================================================
EXPECTED_TARGET_RANGE = 3.0

target_range_bin = np.argmin(
    np.abs(
        range_axis
        - EXPECTED_TARGET_RANGE
    )
)

target_doppler_bin = np.argmax(
    rd_power[:, target_range_bin]
)

measured_target_range = range_axis[
    target_range_bin
]

measured_target_velocity = velocity_axis[
    target_doppler_bin
]

target_power_db = (
    rd_map_normalized_db[
        target_doppler_bin,
        target_range_bin,
    ]
)

print("\nTarget near 3 m:")
print(
    f"Range bin: "
    f"{target_range_bin}"
)

print(
    f"Measured range: "
    f"{measured_target_range:.3f} m"
)

print(
    f"Doppler bin: "
    f"{target_doppler_bin}"
)

print(
    f"Measured radial velocity: "
    f"{measured_target_velocity:.4f} m/s"
)

print(
    f"Normalized power: "
    f"{target_power_db:.2f} dB"
)


# =========================================================
# 8. پیدا کردن قوی‌ترین سلول در محدوده 0.4 تا 10 متر
# =========================================================
MIN_PROCESSING_RANGE = 0.4
MAX_PROCESSING_RANGE = 10.0

min_range_bin = np.searchsorted(
    range_axis,
    MIN_PROCESSING_RANGE,
)

max_range_bin = np.searchsorted(
    range_axis,
    MAX_PROCESSING_RANGE,
)

roi_power = rd_power[
    :,
    min_range_bin:max_range_bin,
]

roi_peak = np.unravel_index(
    np.argmax(roi_power),
    roi_power.shape,
)

strongest_doppler_bin = roi_peak[0]

strongest_range_bin = (
    roi_peak[1]
    + min_range_bin
)

strongest_range = range_axis[
    strongest_range_bin
]

strongest_velocity = velocity_axis[
    strongest_doppler_bin
]

print("\nStrongest Range-Doppler cell:")

print(
    f"Range: "
    f"{strongest_range:.3f} m"
)

print(
    f"Velocity: "
    f"{strongest_velocity:.4f} m/s"
)

print(
    f"Level: "
    f"{rd_map_normalized_db[strongest_doppler_bin, strongest_range_bin]:.2f} dB"
)


# =========================================================
# 9. رسم Range-Doppler Map
# =========================================================
rd_plot = rd_map_normalized_db[
    :,
    :max_range_bin,
]

plt.figure(figsize=(11, 7))

plt.imshow(
    rd_plot.T,
    origin="lower",
    aspect="auto",
    extent=[
        velocity_axis[0],
        velocity_axis[-1],
        range_axis[0],
        range_axis[max_range_bin - 1],
    ],
    cmap="jet",
    vmin=-50,
    vmax=0,
)

plt.colorbar(
    label="Normalized power (dB)"
)

plt.xlabel(
    "Radial velocity (m/s)"
)

plt.ylabel(
    "Range (m)"
)

plt.title(
    f"Range-Doppler map - Frame {FRAME_ID}"
)

plt.scatter(
    measured_target_velocity,
    measured_target_range,
    color="white",
    edgecolor="black",
    marker="o",
    s=80,
    label="Target near 3 m",
)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

doppler_plot_path = (
    FILE_PATH.parent
    / "step3_range_doppler_map.png"
)

plt.savefig(
    doppler_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print("\nRange-Doppler map saved at:")
print(doppler_plot_path)

print("\nSTEP 3 FINISHED.")
# =========================================================
# STEP 4: 2D CA-CFAR
# =========================================================

from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d

print("\n" + "=" * 60)
print("STEP 4: 2D CA-CFAR")
print("=" * 60)


# =========================================================
# 1. پارامترهای CFAR
# =========================================================
TRAIN_DOPPLER = 8
TRAIN_RANGE = 6

GUARD_DOPPLER = 1
GUARD_RANGE = 2

P_FALSE_ALARM = 1e-4

MIN_CFAR_RANGE = 0.4
MAX_CFAR_RANGE = 10.0

MIN_POWER_DB = -35.0


# =========================================================
# 2. ساخت Kernel سلول‌های Training
# =========================================================
half_doppler = (
    TRAIN_DOPPLER
    + GUARD_DOPPLER
)

half_range = (
    TRAIN_RANGE
    + GUARD_RANGE
)

kernel_height = (
    2 * half_doppler + 1
)

kernel_width = (
    2 * half_range + 1
)

cfar_kernel = np.ones(
    (kernel_height, kernel_width),
    dtype=np.float64,
)


# حذف CUT و Guard Cellها از Kernel
center_d = half_doppler
center_r = half_range

cfar_kernel[
    center_d - GUARD_DOPPLER:
    center_d + GUARD_DOPPLER + 1,

    center_r - GUARD_RANGE:
    center_r + GUARD_RANGE + 1,
] = 0.0


number_of_training_cells = int(
    np.sum(cfar_kernel)
)

cfar_kernel /= number_of_training_cells

print(
    "Number of CFAR training cells:",
    number_of_training_cells,
)


# =========================================================
# 3. تخمین نویز محلی
# =========================================================
noise_estimate = convolve2d(
    rd_power,
    cfar_kernel,
    mode="same",
    boundary="symm",
)


# =========================================================
# 4. محاسبه ضریب آستانه CFAR
# =========================================================
cfar_alpha = (
    number_of_training_cells
    * (
        P_FALSE_ALARM
        ** (
            -1.0
            / number_of_training_cells
        )
        - 1.0
    )
)

cfar_threshold = (
    cfar_alpha
    * noise_estimate
)

print(
    f"CFAR alpha: {cfar_alpha:.4f}"
)


# =========================================================
# 5. تشخیص اولیه سلول‌های عبورکرده از CFAR
# =========================================================
cfar_detection_mask = (
    rd_power
    > cfar_threshold
)


# =========================================================
# 6. حذف لبه‌های نامعتبر
# =========================================================
cfar_detection_mask[
    :half_doppler,
    :
] = False

cfar_detection_mask[
    -half_doppler:,
    :
] = False

cfar_detection_mask[
    :,
    :half_range
] = False

cfar_detection_mask[
    :,
    -half_range:
] = False


# =========================================================
# 7. محدود کردن فاصله پردازش
# =========================================================
range_roi_mask = (
    (range_axis >= MIN_CFAR_RANGE)
    & (range_axis <= MAX_CFAR_RANGE)
)

cfar_detection_mask &= (
    range_roi_mask[None, :]
)


# =========================================================
# 8. حذف Detectionهای بسیار ضعیف
# =========================================================
cfar_detection_mask &= (
    rd_map_normalized_db
    >= MIN_POWER_DB
)


# =========================================================
# 9. Peak Grouping
#
# فقط نقاطی نگه داشته می‌شوند که در همسایگی
# 3×3 بیشینه محلی باشند.
# =========================================================
local_maximum = (
    rd_power
    == maximum_filter(
        rd_power,
        size=(3, 3),
        mode="nearest",
    )
)

final_detection_mask = (
    cfar_detection_mask
    & local_maximum
)


# =========================================================
# 10. استخراج Detectionها
# =========================================================
detected_doppler_bins, detected_range_bins = (
    np.where(final_detection_mask)
)

print(
    "Number of detections after CFAR and grouping:",
    detected_range_bins.size,
)


# =========================================================
# 11. محاسبه اطلاعات Detectionها
# =========================================================
detections = []

for doppler_bin, range_bin in zip(
    detected_doppler_bins,
    detected_range_bins,
):
    detected_range = range_axis[
        range_bin
    ]

    detected_velocity = velocity_axis[
        doppler_bin
    ]

    detected_power_db = (
        rd_map_normalized_db[
            doppler_bin,
            range_bin,
        ]
    )

    detected_snr_db = 10 * np.log10(
        (
            rd_power[
                doppler_bin,
                range_bin,
            ]
            + 1e-12
        )
        / (
            noise_estimate[
                doppler_bin,
                range_bin,
            ]
            + 1e-12
        )
    )

    detections.append(
        {
            "doppler_bin": int(doppler_bin),
            "range_bin": int(range_bin),
            "range_m": float(detected_range),
            "velocity_mps": float(
                detected_velocity
            ),
            "power_db": float(
                detected_power_db
            ),
            "snr_db": float(
                detected_snr_db
            ),
        }
    )


# مرتب‌سازی بر اساس قدرت
detections.sort(
    key=lambda item: item["power_db"],
    reverse=True,
)


# =========================================================
# 12. چاپ Detectionها
# =========================================================
print("\nCFAR detections:")

for number, detection in enumerate(
    detections[:20],
    start=1,
):
    print(
        f"{number:2d}. "
        f"R={detection['range_m']:6.3f} m, "
        f"v={detection['velocity_mps']:7.3f} m/s, "
        f"power={detection['power_db']:7.2f} dB, "
        f"SNR={detection['snr_db']:6.2f} dB"
    )


# =========================================================
# 13. پیدا کردن Detection نزدیک هدف واقعی
# =========================================================
if detections:
    target_detection = min(
        detections,
        key=lambda item: (
            abs(
                item["range_m"]
                - EXPECTED_TARGET_RANGE
            )
            / range_bin_spacing
            +
            abs(
                item["velocity_mps"]
            )
            / abs(
                velocity_bin_spacing
            )
        ),
    )

    print("\nDetection nearest to the real target:")

    print(
        f"Range: "
        f"{target_detection['range_m']:.3f} m"
    )

    print(
        f"Velocity: "
        f"{target_detection['velocity_mps']:.4f} m/s"
    )

    print(
        f"SNR: "
        f"{target_detection['snr_db']:.2f} dB"
    )


# =========================================================
# 14. ذخیره Detectionها در CSV
# =========================================================
csv_path = (
    FILE_PATH.parent
    / "step4_cfar_detections.csv"
)

if detections:
    detection_array = np.array(
        [
            [
                item["range_m"],
                item["velocity_mps"],
                item["power_db"],
                item["snr_db"],
                item["range_bin"],
                item["doppler_bin"],
            ]
            for item in detections
        ],
        dtype=float,
    )

    np.savetxt(
        csv_path,
        detection_array,
        delimiter=",",
        header=(
            "range_m,velocity_mps,"
            "power_db,snr_db,"
            "range_bin,doppler_bin"
        ),
        comments="",
        fmt="%.6f",
    )

    print("\nCFAR detections saved at:")
    print(csv_path)


# =========================================================
# 15. رسم نتیجه CFAR روی Range-Doppler Map
# =========================================================
plt.figure(figsize=(11, 7))

plt.imshow(
    rd_plot.T,
    origin="lower",
    aspect="auto",
    extent=[
        velocity_axis[0],
        velocity_axis[-1],
        range_axis[0],
        range_axis[max_range_bin - 1],
    ],
    cmap="jet",
    vmin=-50,
    vmax=0,
)

plt.colorbar(
    label="Normalized power (dB)"
)

if detected_range_bins.size > 0:
    plt.scatter(
        velocity_axis[
            detected_doppler_bins
        ],
        range_axis[
            detected_range_bins
        ],
        marker="x",
        color="red",
        s=65,
        linewidths=1.8,
        label="CFAR detections",
    )

plt.scatter(
    measured_target_velocity,
    measured_target_range,
    facecolors="none",
    edgecolors="white",
    s=100,
    linewidths=1.8,
    label="Real target near 3 m",
)

plt.xlabel(
    "Radial velocity (m/s)"
)

plt.ylabel(
    "Range (m)"
)

plt.title(
    f"2D CA-CFAR detections - Frame {FRAME_ID}"
)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

cfar_plot_path = (
    FILE_PATH.parent
    / "step4_cfar_detections.png"
)

plt.savefig(
    cfar_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print("\nCFAR plot saved at:")
print(cfar_plot_path)

print("\nSTEP 4 FINISHED.")
# =========================================================
# STEP 5: AZIMUTH FFT AND POINT CLOUD
# =========================================================

print("\n" + "=" * 60)
print("STEP 5: AZIMUTH FFT AND POINT CLOUD")
print("=" * 60)


# =========================================================
# 1. پارامترهای تخمین زاویه
# =========================================================
NUM_ANGLE_FFT = 64

# فاصله RXهای مجاور در AWR1843 تقریباً lambda/2 است.
ANTENNA_SPACING = wavelength / 2

# اگر در آزمایش جهت چپ و راست برعکس نمایش داده شد،
# مقدار آن را از 1 به -1 تغییر بده.
ANGLE_SIGN = 1

print(
    f"Number of physical RX antennas: "
    f"{NUM_RX}"
)

print(
    f"Angle FFT size: "
    f"{NUM_ANGLE_FFT}"
)

print(
    f"Assumed RX spacing: "
    f"{ANTENNA_SPACING * 1000:.4f} mm"
)


# =========================================================
# 2. محور زاویه
#
# برای فاصله آنتنی lambda/2:
#
# sin(theta) = 2*k/N
# =========================================================
angle_bins = (
    np.arange(NUM_ANGLE_FFT)
    - NUM_ANGLE_FFT / 2
)

spatial_frequency = (
    2.0
    * angle_bins
    / NUM_ANGLE_FFT
)

spatial_frequency = np.clip(
    spatial_frequency,
    -1.0,
    1.0,
)

angle_axis_rad = np.arcsin(
    spatial_frequency
)

angle_axis_deg = (
    ANGLE_SIGN
    * np.degrees(angle_axis_rad)
)


# =========================================================
# 3. پنجره فضایی آنتن‌ها
# =========================================================
angle_window = np.hamming(
    NUM_RX
).astype(np.float32)


# =========================================================
# 4. اجرای Angle FFT برای Detectionهای CFAR
# =========================================================
point_cloud = []

for detection in detections:

    doppler_bin = detection[
        "doppler_bin"
    ]

    range_bin = detection[
        "range_bin"
    ]

    # مقدار مختلط چهار RX در سلول Range-Doppler
    antenna_vector = range_doppler_cube[
        doppler_bin,
        range_bin,
        :
    ]

    # حذف فاز مشترک نسبت به RX1
    reference_phase = np.angle(
        antenna_vector[0]
    )

    calibrated_vector = (
        antenna_vector
        * np.exp(-1j * reference_phase)
    )

    # اعمال پنجره فضایی
    windowed_antenna_vector = (
        calibrated_vector
        * angle_window
    )

    # اجرای FFT روی محور آنتن
    angle_fft = np.fft.fft(
        windowed_antenna_vector,
        n=NUM_ANGLE_FFT,
    )

    angle_fft = np.fft.fftshift(
        angle_fft
    )

    angle_power = (
        np.abs(angle_fft) ** 2
    )

    angle_peak_bin = int(
        np.argmax(angle_power)
    )

    azimuth_deg = float(
        angle_axis_deg[
            angle_peak_bin
        ]
    )

    azimuth_rad = np.radians(
        azimuth_deg
    )

    target_range = detection[
        "range_m"
    ]

    # قرارداد مختصات:
    # x: جهت افقی
    # y: جهت روبه‌جلو
    # z: ارتفاع
    x_coordinate = (
        target_range
        * np.sin(azimuth_rad)
    )

    y_coordinate = (
        target_range
        * np.cos(azimuth_rad)
    )

    z_coordinate = 0.0

    angle_power_db = (
        10
        * np.log10(
            angle_power + 1e-12
        )
    )

    angle_power_normalized_db = (
        angle_power_db
        - np.max(angle_power_db)
    )

    point_cloud.append(
        {
            "x_m": float(x_coordinate),
            "y_m": float(y_coordinate),
            "z_m": float(z_coordinate),
            "range_m": float(target_range),
            "velocity_mps": float(
                detection["velocity_mps"]
            ),
            "azimuth_deg": float(
                azimuth_deg
            ),
            "power_db": float(
                detection["power_db"]
            ),
            "snr_db": float(
                detection["snr_db"]
            ),
            "range_bin": int(
                range_bin
            ),
            "doppler_bin": int(
                doppler_bin
            ),
            "angle_bin": int(
                angle_peak_bin
            ),
            "angle_spectrum_db":
                angle_power_normalized_db,
        }
    )


# =========================================================
# 5. چاپ Point Cloud
# =========================================================
print(
    f"Number of point-cloud points: "
    f"{len(point_cloud)}"
)

print("\nEstimated point cloud:")

for number, point in enumerate(
    point_cloud,
    start=1,
):
    print(
        f"{number:2d}. "
        f"x={point['x_m']:7.3f} m, "
        f"y={point['y_m']:7.3f} m, "
        f"z={point['z_m']:6.3f} m, "
        f"R={point['range_m']:6.3f} m, "
        f"az={point['azimuth_deg']:7.2f} deg, "
        f"v={point['velocity_mps']:7.3f} m/s, "
        f"SNR={point['snr_db']:6.2f} dB"
    )


# =========================================================
# 6. پیدا کردن نقطه متناظر با هدف سه‌متری
# =========================================================
if point_cloud:

    real_target_point = min(
        point_cloud,
        key=lambda point: (
            abs(
                point["range_m"]
                - EXPECTED_TARGET_RANGE
            )
            +
            abs(
                point["velocity_mps"]
            )
        ),
    )

    print("\nPoint nearest to real target:")

    print(
        f"Range: "
        f"{real_target_point['range_m']:.3f} m"
    )

    print(
        f"Azimuth: "
        f"{real_target_point['azimuth_deg']:.2f} deg"
    )

    print(
        f"x: "
        f"{real_target_point['x_m']:.3f} m"
    )

    print(
        f"y: "
        f"{real_target_point['y_m']:.3f} m"
    )

    print(
        f"z: "
        f"{real_target_point['z_m']:.3f} m"
    )

    print(
        f"Velocity: "
        f"{real_target_point['velocity_mps']:.4f} m/s"
    )

    print(
        f"SNR: "
        f"{real_target_point['snr_db']:.2f} dB"
    )


# =========================================================
# 7. ذخیره Point Cloud در CSV
# =========================================================
point_cloud_csv_path = (
    FILE_PATH.parent
    / "step5_point_cloud.csv"
)

if point_cloud:

    point_cloud_array = np.array(
        [
            [
                point["x_m"],
                point["y_m"],
                point["z_m"],
                point["range_m"],
                point["velocity_mps"],
                point["azimuth_deg"],
                point["power_db"],
                point["snr_db"],
                point["range_bin"],
                point["doppler_bin"],
                point["angle_bin"],
            ]
            for point in point_cloud
        ],
        dtype=float,
    )

    np.savetxt(
        point_cloud_csv_path,
        point_cloud_array,
        delimiter=",",
        header=(
            "x_m,y_m,z_m,range_m,"
            "velocity_mps,azimuth_deg,"
            "power_db,snr_db,"
            "range_bin,doppler_bin,angle_bin"
        ),
        comments="",
        fmt="%.6f",
    )

    print("\nPoint cloud saved at:")
    print(point_cloud_csv_path)


# =========================================================
# 8. رسم Angle Spectrum هدف سه‌متری
# =========================================================
if point_cloud:

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        angle_axis_deg,
        real_target_point[
            "angle_spectrum_db"
        ],
        color="darkblue",
        linewidth=1.5,
    )

    plt.axvline(
        real_target_point[
            "azimuth_deg"
        ],
        color="red",
        linestyle="--",
        label=(
            f"Estimated angle = "
            f"{real_target_point['azimuth_deg']:.2f} deg"
        ),
    )

    plt.xlabel(
        "Azimuth angle (degree)"
    )

    plt.ylabel(
        "Normalized angle power (dB)"
    )

    plt.title(
        "Azimuth spectrum of target near 3 m"
    )

    plt.xlim(-90, 90)
    plt.ylim(-40, 5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    angle_plot_path = (
        FILE_PATH.parent
        / "step5_target_angle_spectrum.png"
    )

    plt.savefig(
        angle_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print("\nAngle spectrum saved at:")
    print(angle_plot_path)


# =========================================================
# 9. رسم Point Cloud دوبعدی و سه‌بعدی
# =========================================================
if point_cloud:

    x_values = np.array(
        [
            point["x_m"]
            for point in point_cloud
        ]
    )

    y_values = np.array(
        [
            point["y_m"]
            for point in point_cloud
        ]
    )

    z_values = np.array(
        [
            point["z_m"]
            for point in point_cloud
        ]
    )

    snr_values = np.array(
        [
            point["snr_db"]
            for point in point_cloud
        ]
    )

    fig = plt.figure(
        figsize=(14, 6)
    )


    # -----------------------------------------------------
    # نمای بالا
    # -----------------------------------------------------
    ax1 = fig.add_subplot(
        1,
        2,
        1,
    )

    scatter_2d = ax1.scatter(
        x_values,
        y_values,
        c=snr_values,
        cmap="turbo",
        s=90,
        edgecolors="black",
    )

    ax1.scatter(
        0,
        0,
        marker="^",
        color="black",
        s=120,
        label="Radar",
    )

    ax1.set_xlabel(
        "Lateral position x (m)"
    )

    ax1.set_ylabel(
        "Forward position y (m)"
    )

    ax1.set_title(
        "Radar point cloud - Top view"
    )

    ax1.grid(
        True,
        alpha=0.3,
    )

    ax1.axis("equal")
    ax1.legend()

    fig.colorbar(
        scatter_2d,
        ax=ax1,
        label="SNR (dB)",
    )


    # -----------------------------------------------------
    # نمای سه‌بعدی با z=0
    # -----------------------------------------------------
    ax2 = fig.add_subplot(
        1,
        2,
        2,
        projection="3d",
    )

    scatter_3d = ax2.scatter(
        x_values,
        y_values,
        z_values,
        c=snr_values,
        cmap="turbo",
        s=90,
        edgecolors="black",
    )

    ax2.scatter(
        [0],
        [0],
        [0],
        marker="^",
        color="black",
        s=120,
        label="Radar",
    )

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_zlabel("z (m)")

    ax2.set_title(
        "Radar point cloud - z=0"
    )

    ax2.set_zlim(
        -0.5,
        0.5,
    )

    ax2.legend()

    fig.colorbar(
        scatter_3d,
        ax=ax2,
        label="SNR (dB)",
        shrink=0.75,
    )

    plt.tight_layout()

    point_cloud_plot_path = (
        FILE_PATH.parent
        / "step5_azimuth_point_cloud.png"
    )

    plt.savefig(
        point_cloud_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print("\nPoint-cloud plot saved at:")
    print(point_cloud_plot_path)


print("\nSTEP 5 FINISHED.")
# =========================================================
# STEP 6: MULTI-FRAME POINT CLOUD AND TRACKING
# =========================================================

print("\n" + "=" * 60)
print("STEP 6: MULTI-FRAME POINT CLOUD AND TRACKING")
print("=" * 60)


# =========================================================
# 1. تنظیمات زمانی و Tracking
# =========================================================

# مقدار Frame Periodicity از Log:
# 8,000,000 unit × 5 ns = 40 ms
FRAME_PERIOD = 40e-3

# هدف واقعی در فاصله حدود 3 متر قرار داشته است.
TRACK_MIN_RANGE = 2.4
TRACK_MAX_RANGE = 3.5

# بیشترین جابه‌جایی قابل قبول میان دو Frame
TRACK_GATE_M = 0.75

print(
    f"Number of frames: {num_frames}"
)

print(
    f"Frame period: "
    f"{FRAME_PERIOD * 1000:.2f} ms"
)

print(
    f"Frame rate: "
    f"{1 / FRAME_PERIOD:.2f} Hz"
)


# =========================================================
# 2. تابع پردازش یک Frame
# =========================================================
def process_one_frame(frame_id):

    # -----------------------------------------------------
    # انتخاب Chirpهای Frame
    # -----------------------------------------------------
    first_chirp = (
        frame_id
        * CHIRPS_PER_FRAME
    )

    last_chirp = (
        first_chirp
        + CHIRPS_PER_FRAME
    )

    current_frame = adc_data[
        first_chirp:last_chirp,
        :,
        :,
    ]


    # -----------------------------------------------------
    # حذف DC
    # -----------------------------------------------------
    current_frame = (
        current_frame
        - np.mean(
            current_frame,
            axis=1,
            keepdims=True,
        )
    )


    # -----------------------------------------------------
    # Range FFT
    # -----------------------------------------------------
    current_range_fft = np.fft.fft(
        current_frame
        * range_window[None, :, None],
        n=NUM_RANGE_FFT,
        axis=1,
    )


    # -----------------------------------------------------
    # Doppler FFT
    # -----------------------------------------------------
    current_rd_cube = np.fft.fft(
        current_range_fft
        * doppler_window[:, None, None],
        n=NUM_DOPPLER_FFT,
        axis=0,
    )

    current_rd_cube = np.fft.fftshift(
        current_rd_cube,
        axes=0,
    )


    # -----------------------------------------------------
    # محاسبه توان
    # -----------------------------------------------------
    current_rd_power = np.sum(
        np.abs(current_rd_cube) ** 2,
        axis=2,
    )

    current_rd_db = 10 * np.log10(
        current_rd_power + 1e-12
    )

    current_rd_normalized_db = (
        current_rd_db
        - np.max(current_rd_db)
    )


    # -----------------------------------------------------
    # تخمین نویز CFAR
    # -----------------------------------------------------
    current_noise = convolve2d(
        current_rd_power,
        cfar_kernel,
        mode="same",
        boundary="symm",
    )

    current_threshold = (
        cfar_alpha
        * current_noise
    )


    # -----------------------------------------------------
    # CFAR Detection
    # -----------------------------------------------------
    current_detection_mask = (
        current_rd_power
        > current_threshold
    )


    # حذف لبه‌های CFAR
    current_detection_mask[
        :half_doppler,
        :
    ] = False

    current_detection_mask[
        -half_doppler:,
        :
    ] = False

    current_detection_mask[
        :,
        :half_range
    ] = False

    current_detection_mask[
        :,
        -half_range:
    ] = False


    # محدودکردن بازه فاصله
    current_detection_mask &= (
        range_roi_mask[None, :]
    )


    # حذف نقاط بسیار ضعیف
    current_detection_mask &= (
        current_rd_normalized_db
        >= MIN_POWER_DB
    )


    # -----------------------------------------------------
    # Peak Grouping
    # -----------------------------------------------------
    current_local_maximum = (
        current_rd_power
        == maximum_filter(
            current_rd_power,
            size=(3, 3),
            mode="nearest",
        )
    )

    current_detection_mask &= (
        current_local_maximum
    )


    # -----------------------------------------------------
    # استخراج Range Bin و Doppler Bin
    # -----------------------------------------------------
    current_doppler_bins, current_range_bins = (
        np.where(
            current_detection_mask
        )
    )


    # -----------------------------------------------------
    # تخمین زاویه و ساخت Point Cloud
    # -----------------------------------------------------
    frame_points = []

    for doppler_bin, range_bin in zip(
        current_doppler_bins,
        current_range_bins,
    ):

        antenna_vector = current_rd_cube[
            doppler_bin,
            range_bin,
            :
        ]

        reference_phase = np.angle(
            antenna_vector[0]
        )

        antenna_vector = (
            antenna_vector
            * np.exp(
                -1j * reference_phase
            )
        )

        angle_fft = np.fft.fft(
            antenna_vector
            * angle_window,
            n=NUM_ANGLE_FFT,
        )

        angle_fft = np.fft.fftshift(
            angle_fft
        )

        angle_power = (
            np.abs(angle_fft) ** 2
        )

        angle_bin = int(
            np.argmax(angle_power)
        )

        azimuth_deg = float(
            angle_axis_deg[
                angle_bin
            ]
        )

        azimuth_rad = np.radians(
            azimuth_deg
        )

        detected_range = float(
            range_axis[
                range_bin
            ]
        )

        detected_velocity = float(
            velocity_axis[
                doppler_bin
            ]
        )

        power_db = float(
            current_rd_normalized_db[
                doppler_bin,
                range_bin,
            ]
        )

        snr_db = float(
            10
            * np.log10(
                (
                    current_rd_power[
                        doppler_bin,
                        range_bin,
                    ]
                    + 1e-12
                )
                /
                (
                    current_noise[
                        doppler_bin,
                        range_bin,
                    ]
                    + 1e-12
                )
            )
        )

        x_position = (
            detected_range
            * np.sin(azimuth_rad)
        )

        y_position = (
            detected_range
            * np.cos(azimuth_rad)
        )

        frame_points.append(
            {
                "frame": int(frame_id),
                "time_s": float(
                    frame_id
                    * FRAME_PERIOD
                ),
                "x_m": float(x_position),
                "y_m": float(y_position),
                "z_m": 0.0,
                "range_m": detected_range,
                "velocity_mps":
                    detected_velocity,
                "azimuth_deg":
                    azimuth_deg,
                "power_db": power_db,
                "snr_db": snr_db,
                "range_bin":
                    int(range_bin),
                "doppler_bin":
                    int(doppler_bin),
                "angle_bin":
                    int(angle_bin),
            }
        )

    return frame_points


# =========================================================
# 3. پردازش تمام Frameها
# =========================================================
all_frame_points = []
points_by_frame = []

for frame_id in range(num_frames):

    frame_points = process_one_frame(
        frame_id
    )

    points_by_frame.append(
        frame_points
    )

    all_frame_points.extend(
        frame_points
    )

    if (
        frame_id % 10 == 0
        or frame_id == num_frames - 1
    ):
        print(
            f"Processed frame "
            f"{frame_id + 1}/{num_frames}, "
            f"detections="
            f"{len(frame_points)}"
        )


print(
    "\nTotal point-cloud points:",
    len(all_frame_points)
)


# =========================================================
# 4. ذخیره Point Cloud تمام Frameها
# =========================================================
all_points_csv_path = (
    FILE_PATH.parent
    / "step6_all_frames_point_cloud.csv"
)

if all_frame_points:

    all_points_array = np.array(
        [
            [
                point["frame"],
                point["time_s"],
                point["x_m"],
                point["y_m"],
                point["z_m"],
                point["range_m"],
                point["velocity_mps"],
                point["azimuth_deg"],
                point["power_db"],
                point["snr_db"],
                point["range_bin"],
                point["doppler_bin"],
                point["angle_bin"],
            ]
            for point in all_frame_points
        ],
        dtype=float,
    )

    np.savetxt(
        all_points_csv_path,
        all_points_array,
        delimiter=",",
        header=(
            "frame,time_s,x_m,y_m,z_m,"
            "range_m,velocity_mps,"
            "azimuth_deg,power_db,snr_db,"
            "range_bin,doppler_bin,angle_bin"
        ),
        comments="",
        fmt="%.6f",
    )

    print(
        "\nAll-frame point cloud saved at:"
    )

    print(
        all_points_csv_path
    )


# =========================================================
# 5. ردیابی هدف سه‌متری
# =========================================================
target_track = []
previous_target = None

for frame_id, frame_points in enumerate(
    points_by_frame
):

    # فقط Detectionهای نزدیک فاصله هدف
    candidates = [
        point
        for point in frame_points
        if (
            TRACK_MIN_RANGE
            <= point["range_m"]
            <= TRACK_MAX_RANGE
        )
    ]


    # اگر هدف در Frame قبلی وجود داشته،
    # از Distance Gate استفاده می‌شود.
    if previous_target is not None:

        gated_candidates = []

        for point in candidates:

            displacement = np.sqrt(
                (
                    point["x_m"]
                    - previous_target["x_m"]
                ) ** 2
                +
                (
                    point["y_m"]
                    - previous_target["y_m"]
                ) ** 2
            )

            if displacement <= TRACK_GATE_M:
                gated_candidates.append(
                    point
                )

        if gated_candidates:
            candidates = gated_candidates


    if candidates:

        # انتخاب نقطه دارای بیشترین SNR
        selected_target = max(
            candidates,
            key=lambda point:
                point["snr_db"],
        )

        selected_target = (
            selected_target.copy()
        )

        selected_target[
            "detected"
        ] = 1

        target_track.append(
            selected_target
        )

        previous_target = (
            selected_target
        )

    else:

        # در صورت گم‌شدن Detection
        target_track.append(
            {
                "frame": frame_id,
                "time_s":
                    frame_id * FRAME_PERIOD,
                "x_m": np.nan,
                "y_m": np.nan,
                "z_m": np.nan,
                "range_m": np.nan,
                "velocity_mps": np.nan,
                "azimuth_deg": np.nan,
                "power_db": np.nan,
                "snr_db": np.nan,
                "detected": 0,
            }
        )


# =========================================================
# 6. آمار Tracking
# =========================================================
valid_track = [
    point
    for point in target_track
    if point["detected"] == 1
]

detection_rate = (
    100
    * len(valid_track)
    / num_frames
)

print("\nTracking results:")

print(
    f"Detected frames: "
    f"{len(valid_track)}/{num_frames}"
)

print(
    f"Detection rate: "
    f"{detection_rate:.2f}%"
)


if valid_track:

    track_ranges = np.array(
        [
            point["range_m"]
            for point in valid_track
        ]
    )

    track_angles = np.array(
        [
            point["azimuth_deg"]
            for point in valid_track
        ]
    )

    track_velocities = np.array(
        [
            point["velocity_mps"]
            for point in valid_track
        ]
    )

    track_x = np.array(
        [
            point["x_m"]
            for point in valid_track
        ]
    )

    track_y = np.array(
        [
            point["y_m"]
            for point in valid_track
        ]
    )

    track_snr = np.array(
        [
            point["snr_db"]
            for point in valid_track
        ]
    )

    print(
        f"Mean range: "
        f"{np.mean(track_ranges):.3f} m"
    )

    print(
        f"Range standard deviation: "
        f"{np.std(track_ranges):.4f} m"
    )

    print(
        f"Mean azimuth: "
        f"{np.mean(track_angles):.2f} deg"
    )

    print(
        f"Azimuth standard deviation: "
        f"{np.std(track_angles):.2f} deg"
    )

    print(
        f"Mean radial velocity: "
        f"{np.mean(track_velocities):.4f} m/s"
    )

    print(
        f"Mean x: "
        f"{np.mean(track_x):.3f} m"
    )

    print(
        f"Mean y: "
        f"{np.mean(track_y):.3f} m"
    )

    print(
        f"Mean SNR: "
        f"{np.mean(track_snr):.2f} dB"
    )


# =========================================================
# 7. ذخیره Track هدف
# =========================================================
track_csv_path = (
    FILE_PATH.parent
    / "step6_target_track.csv"
)

track_array = np.array(
    [
        [
            point["frame"],
            point["time_s"],
            point["x_m"],
            point["y_m"],
            point["z_m"],
            point["range_m"],
            point["velocity_mps"],
            point["azimuth_deg"],
            point["power_db"],
            point["snr_db"],
            point["detected"],
        ]
        for point in target_track
    ],
    dtype=float,
)

np.savetxt(
    track_csv_path,
    track_array,
    delimiter=",",
    header=(
        "frame,time_s,x_m,y_m,z_m,"
        "range_m,velocity_mps,"
        "azimuth_deg,power_db,snr_db,"
        "detected"
    ),
    comments="",
    fmt="%.6f",
)

print("\nTarget track saved at:")
print(track_csv_path)


# =========================================================
# 8. رسم نتایج Tracking
# =========================================================
track_time = np.array(
    [
        point["time_s"]
        for point in target_track
    ]
)

track_range_plot = np.array(
    [
        point["range_m"]
        for point in target_track
    ]
)

track_angle_plot = np.array(
    [
        point["azimuth_deg"]
        for point in target_track
    ]
)

track_velocity_plot = np.array(
    [
        point["velocity_mps"]
        for point in target_track
    ]
)

track_snr_plot = np.array(
    [
        point["snr_db"]
        for point in target_track
    ]
)

track_x_plot = np.array(
    [
        point["x_m"]
        for point in target_track
    ]
)

track_y_plot = np.array(
    [
        point["y_m"]
        for point in target_track
    ]
)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)


# Range versus time
axes[0, 0].plot(
    track_time,
    track_range_plot,
    marker="o",
    markersize=3,
    linewidth=1,
)

axes[0, 0].axhline(
    3.0,
    color="red",
    linestyle="--",
    label="Real range = 3 m",
)

axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Range (m)")
axes[0, 0].set_title("Tracked range")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()


# Azimuth versus time
axes[0, 1].plot(
    track_time,
    track_angle_plot,
    marker="o",
    markersize=3,
    linewidth=1,
    color="darkorange",
)

axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Azimuth (degree)")
axes[0, 1].set_title("Tracked azimuth")
axes[0, 1].grid(True, alpha=0.3)


# Velocity and SNR
axes[1, 0].plot(
    track_time,
    track_velocity_plot,
    label="Velocity (m/s)",
    linewidth=1.2,
)

axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Velocity (m/s)")
axes[1, 0].set_title("Tracked radial velocity")
axes[1, 0].grid(True, alpha=0.3)


# Trajectory in x-y plane
scatter_track = axes[1, 1].scatter(
    track_x_plot,
    track_y_plot,
    c=track_time,
    cmap="viridis",
    s=35,
)

axes[1, 1].scatter(
    0,
    0,
    marker="^",
    color="black",
    s=100,
    label="Radar",
)

axes[1, 1].set_xlabel("x (m)")
axes[1, 1].set_ylabel("y (m)")
axes[1, 1].set_title("Target trajectory")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axis("equal")
axes[1, 1].legend()

fig.colorbar(
    scatter_track,
    ax=axes[1, 1],
    label="Time (s)",
)

plt.tight_layout()

tracking_plot_path = (
    FILE_PATH.parent
    / "step6_tracking_results.png"
)

plt.savefig(
    tracking_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print("\nTracking plot saved at:")
print(tracking_plot_path)

print("\nSTEP 6 FINISHED.")
