"""
Doppler Effect Demonstration v6 - Raspberry Pi + USB Webcam + LED + Speaker
------------------------------------------------------------------------------
Detects hand distance continuously via webcam (motion-based, per-session
skin-color calibration - works across different people's skin tones).

NEW IN THIS VERSION:
  1. STARTUP BANNER - a clear title/info screen when the script launches.
  2. AUTO-RETRY CALIBRATION - if calibration fails validation once, it
     automatically retries before falling back to generic thresholds.
  3. TWO-HAND MODE - your LEFT hand (as seen by the camera) controls
     pitch/blink rate as before. Your RIGHT hand, if visible, controls
     VOLUME - closer right hand = louder, farther = quieter. This adds
     amplitude change alongside frequency change, closer to a full
     acoustic Doppler simulation. Falls back to single-hand behavior
     automatically if only one hand is visible.

CARRIED FORWARD:
  - Motion-based hand detection + per-session skin-color calibration
  - OpenCV 3.x/4.x compatibility
  - ROI as % of actual camera resolution, camera warm-up frames
  - Exponential smoothing on distance readings
  - 16-step resolution, live distance meter (terminal bar)
  - Auto-generated session graph (PNG) on exit
  - Smooth PWM LED fade instead of hard blink
  - Live Doppler physics overlay (real vs exaggerated-demo frequency shift)
  - Spoken "approaching"/"receding" voice announcements

WIRING:
  LED anode -> 1k ohm resistor -> GPIO18 (Pi Pin 12)
  LED cathode -> GND (Pi Pin 14)
  USB webcam -> any USB port
  Speaker/earphones -> Pi's 3.5mm audio jack

SETUP:
  sudo apt update
  sudo apt install python3-opencv python3-matplotlib espeak alsa-utils
  python3 -c "import numpy" || sudo apt install python3-numpy
  sudo raspi-config -> System Options -> Audio -> Headphones
  Check your volume mixer name with: amixer scontrols
    (this script assumes "PCM" - change VOLUME_MIXER_NAME below if
    your system uses "Master" or something else instead)

Requires: gpiozero, opencv (cv2), numpy, matplotlib
"""

from gpiozero import PWMLED
import cv2
import numpy as np
import wave
import os
import time
import datetime

import matplotlib
matplotlib.use("Agg")  # headless - no display attached over SSH
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# GPIO PIN SETUP
# ---------------------------------------------------------------------
LED_PIN = 18
led = PWMLED(LED_PIN)

# ---------------------------------------------------------------------
# CAMERA CONFIGURATION
# ---------------------------------------------------------------------
CAMERA_INDEX = 0
REQUESTED_WIDTH = 320
REQUESTED_HEIGHT = 240
ROI_MARGIN_FRACTION = 0.10
NUM_FRAMES_TO_WARM_UP = 10

# ---------------------------------------------------------------------
# SKIN COLOR RANGE - determined per session by calibration (see below).
# ---------------------------------------------------------------------
FALLBACK_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
FALLBACK_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)
current_ycrcb_lower = FALLBACK_YCRCB_LOWER.copy()
current_ycrcb_upper = FALLBACK_YCRCB_UPPER.copy()

# ---------------------------------------------------------------------
# CALIBRATION CONFIGURATION
# ---------------------------------------------------------------------
CALIBRATION_COUNTDOWN_SEC = 3
CALIBRATION_SAMPLE_FRAMES = 20
MOTION_DIFF_THRESHOLD = 25
MIN_MOTION_BLOB_AREA = 400
COLOR_RANGE_MARGIN_CR = 8
COLOR_RANGE_MARGIN_CB = 8
BACKGROUND_SAFETY_MARGIN = 1.4
MAX_CALIBRATION_ATTEMPTS = 2  # auto-retry once before falling back

FALLBACK_MIN_HAND_AREA = 800
FALLBACK_MAX_HAND_AREA = 15000

# ---------------------------------------------------------------------
# BLINK / TONE / SMOOTHING CONFIGURATION
# ---------------------------------------------------------------------
NEAR_BLINK_INTERVAL = 0.08
FAR_BLINK_INTERVAL = 0.6
NEAR_FREQ_HZ = 1000
FAR_FREQ_HZ = 300
TONE_DURATION = 0.12
NUM_TONE_STEPS = 16
SMOOTHING_ALPHA = 0.35

# ---------------------------------------------------------------------
# TWO-HAND MODE CONFIGURATION
# ---------------------------------------------------------------------
TWO_HAND_MODE_ENABLED = True
MIN_SECOND_HAND_AREA = 400       # ignore small noise blobs as a "second hand"
VOLUME_FAR_PERCENT = 20          # quietest (right hand far / not visible)
VOLUME_NEAR_PERCENT = 100        # loudest (right hand close)
VOLUME_CHANGE_THRESHOLD = 4      # only re-set system volume if change exceeds this %
VOLUME_MIXER_NAME = "PCM"        # check with 'amixer scontrols'; try "Master" if this fails

# ---------------------------------------------------------------------
# GRAPH / LOGGING CONFIGURATION
# ---------------------------------------------------------------------
GRAPH_OUTPUT_DIR = "/home/pi"
DISTANCE_BAR_WIDTH = 24

# ---------------------------------------------------------------------
# PHYSICS OVERLAY CONFIGURATION
# ---------------------------------------------------------------------
SPEED_OF_SOUND_MPS = 343.0
PSEUDO_DISTANCE_FAR_CM = 50.0
PSEUDO_DISTANCE_NEAR_CM = 5.0
MIN_SPEED_FOR_DIRECTION_MPS = 0.02

# ---------------------------------------------------------------------
# VOICE ANNOUNCEMENT CONFIGURATION
# ---------------------------------------------------------------------
SPEECH_ENABLED = True
SPEECH_COOLDOWN_SEC = 2.5

TONE_DIR = "/tmp"
tone_files = []


def print_startup_banner():
    print("=" * 60)
    print("   DOPPLER EFFECT DEMONSTRATION - Camera Version".center(60))
    print("=" * 60)
    print("  Simulates the Doppler effect using hand distance,")
    print("  detected via webcam, to control LED pulse rate and")
    print("  audio pitch (and volume, in two-hand mode).")
    print("")
    print("  Components: Raspberry Pi, USB webcam, LED, resistor")
    print("  Run date : {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 60)
    print("")


def generate_tone_file(filename, frequency, duration, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = volume * np.sin(2 * np.pi * frequency * t)
    audio = (waveform * 32767).astype(np.int16)
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


def generate_all_tones():
    print("Generating {} tone steps...".format(NUM_TONE_STEPS))
    for i in range(NUM_TONE_STEPS):
        fraction = i / float(NUM_TONE_STEPS - 1)
        freq = FAR_FREQ_HZ + fraction * (NEAR_FREQ_HZ - FAR_FREQ_HZ)
        filename = os.path.join(TONE_DIR, "tone_{}.wav".format(i))
        generate_tone_file(filename, freq, TONE_DURATION)
        tone_files.append(filename)


def play_tone_file(filename):
    os.system("aplay -q {}".format(filename))


def set_system_volume(percent):
    percent = max(0, min(100, int(round(percent))))
    os.system("amixer -q sset {} {}% 2>/dev/null".format(VOLUME_MIXER_NAME, percent))


def find_contours_compat(mask):
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 3:
        _, contours, _ = result
    else:
        contours, _ = result
    return contours


def get_roi_coords(frame_width, frame_height):
    x1 = int(frame_width * ROI_MARGIN_FRACTION)
    y1 = int(frame_height * ROI_MARGIN_FRACTION)
    x2 = int(frame_width * (1 - ROI_MARGIN_FRACTION))
    y2 = int(frame_height * (1 - ROI_MARGIN_FRACTION))
    return x1, y1, x2, y2


def get_skin_mask(roi_frame):
    ycrcb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, current_ycrcb_lower, current_ycrcb_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def get_hand_area_from_roi(roi_frame):
    mask = get_skin_mask(roi_frame)
    contours = find_contours_compat(mask)
    if not contours:
        return 0
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest)


def get_hand_area(frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    return get_hand_area_from_roi(roi)


def contour_centroid_x(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return 0
    return m["m10"] / m["m00"]


def get_up_to_two_hands(frame, roi_coords):
    """
    Find up to two skin-colored blobs in the ROI, sorted LEFT to RIGHT
    (as seen by the camera). Returns a list of areas: [left_area] if
    one hand visible, [left_area, right_area] if two. Small blobs
    below MIN_SECOND_HAND_AREA are ignored so noise doesn't get
    mistaken for a second hand.
    """
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    mask = get_skin_mask(roi)
    contours = find_contours_compat(mask)
    if not contours:
        return []

    valid = [c for c in contours if cv2.contourArea(c) >= MIN_SECOND_HAND_AREA]
    if not valid:
        return []

    valid.sort(key=cv2.contourArea, reverse=True)
    valid = valid[:2]
    valid.sort(key=contour_centroid_x)  # left-to-right

    return [cv2.contourArea(c) for c in valid]


def get_largest_motion_contour(roi_frame, background_gray_roi):
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background_gray_roi)
    _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = find_contours_compat(thresh)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_MOTION_BLOB_AREA:
        return None, None
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return largest, mask


def countdown(seconds):
    for i in range(seconds, 0, -1):
        print("  {}...".format(i))
        time.sleep(1)


def attempt_calibration(cap, roi_coords):
    """One calibration attempt. Returns (min_area, max_area, success_bool)."""
    global current_ycrcb_lower, current_ycrcb_upper
    x1, y1, x2, y2 = roi_coords

    print("Step 1: Move your hand OUT of the camera's view completely.")
    countdown(CALIBRATION_COUNTDOWN_SEC)
    print("Capturing background...")

    background_frames = []
    for _ in range(CALIBRATION_SAMPLE_FRAMES):
        ret, frame = cap.read()
        if ret:
            background_frames.append(frame[y1:y2, x1:x2])
        time.sleep(0.03)

    if not background_frames:
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA, False

    background_gray_stack = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in background_frames]
    )
    background_gray_avg = np.mean(background_gray_stack, axis=0).astype(np.uint8)

    print("")
    print("Step 2: Hold your hand as CLOSE to the camera as you will during the demo.")
    print("(Keep it inside the center of the frame)")
    countdown(CALIBRATION_COUNTDOWN_SEC)
    print("Measuring your hand...")

    near_areas = []
    sampled_cr = []
    sampled_cb = []

    for _ in range(CALIBRATION_SAMPLE_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        roi_frame = frame[y1:y2, x1:x2]
        contour, mask = get_largest_motion_contour(roi_frame, background_gray_avg)
        if contour is None:
            time.sleep(0.03)
            continue
        near_areas.append(cv2.contourArea(contour))
        ycrcb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)
        hand_pixels = ycrcb[mask == 255]
        if len(hand_pixels) > 0:
            sampled_cr.extend(hand_pixels[:, 1].tolist())
            sampled_cb.extend(hand_pixels[:, 2].tolist())
        time.sleep(0.03)

    if not near_areas or not sampled_cr:
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA, False

    cr_low = max(0, int(np.percentile(sampled_cr, 5)) - COLOR_RANGE_MARGIN_CR)
    cr_high = min(255, int(np.percentile(sampled_cr, 95)) + COLOR_RANGE_MARGIN_CR)
    cb_low = max(0, int(np.percentile(sampled_cb, 5)) - COLOR_RANGE_MARGIN_CB)
    cb_high = min(255, int(np.percentile(sampled_cb, 95)) + COLOR_RANGE_MARGIN_CB)

    current_ycrcb_lower = np.array([0, cr_low, cb_low], dtype=np.uint8)
    current_ycrcb_upper = np.array([255, cr_high, cb_high], dtype=np.uint8)

    print("Personalized skin-color range set: Cr[{},{}] Cb[{},{}]".format(
        cr_low, cr_high, cb_low, cb_high))

    near_level = float(np.percentile(near_areas, 90))
    background_color_areas = [get_hand_area_from_roi(f) for f in background_frames]
    background_level = max(background_color_areas) if background_color_areas else 0

    min_area = background_level * BACKGROUND_SAFETY_MARGIN
    max_area = near_level

    if max_area <= min_area * 1.2:
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA, False

    return min_area, max_area, True


def run_calibration(cap, roi_coords):
    """Runs calibration, automatically retrying once if the first attempt fails."""
    print("")
    print("=== CALIBRATION ===")

    for attempt in range(1, MAX_CALIBRATION_ATTEMPTS + 1):
        if attempt > 1:
            print("")
            print("Calibration attempt {} didn't succeed - retrying automatically...".format(attempt - 1))
            print("(Tip: better/more even lighting helps a lot)")
            print("")

        min_area, max_area, success = attempt_calibration(cap, roi_coords)

        if success:
            print("")
            print("=== CALIBRATION COMPLETE (attempt {}) ===".format(attempt))
            print("MIN_HAND_AREA = {:.0f}   MAX_HAND_AREA = {:.0f}".format(min_area, max_area))
            print("")
            return min_area, max_area

    print("")
    print("WARNING: Calibration failed after {} attempts.".format(MAX_CALIBRATION_ATTEMPTS))
    print("Using generic fallback thresholds - detection may be less accurate.")
    print("")
    return min_area, max_area


def map_area_to_step(area, min_area, max_area):
    if area < min_area:
        return 0
    clamped = min(area, max_area)
    span = float(max_area - min_area)
    fraction = (clamped - min_area) / span if span > 0 else 0
    step = int(round(fraction * (NUM_TONE_STEPS - 1)))
    return max(0, min(step, NUM_TONE_STEPS - 1))


def step_to_blink_interval(step):
    fraction = step / float(NUM_TONE_STEPS - 1)
    return FAR_BLINK_INTERVAL - fraction * (FAR_BLINK_INTERVAL - NEAR_BLINK_INTERVAL)


def make_distance_bar(fraction, width=DISTANCE_BAR_WIDTH):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    bar = "|" * filled + "-" * (width - filled)
    return "[{}] {:3.0f}%".format(bar, fraction * 100)


def fraction_to_pseudo_distance_cm(fraction):
    fraction = max(0.0, min(1.0, fraction))
    return PSEUDO_DISTANCE_FAR_CM - fraction * (PSEUDO_DISTANCE_FAR_CM - PSEUDO_DISTANCE_NEAR_CM)


def compute_real_doppler_freq(base_freq, velocity_mps):
    if velocity_mps > 0:
        return base_freq * SPEED_OF_SOUND_MPS / (SPEED_OF_SOUND_MPS - velocity_mps)
    elif velocity_mps < 0:
        return base_freq * SPEED_OF_SOUND_MPS / (SPEED_OF_SOUND_MPS + abs(velocity_mps))
    return base_freq


def speak_async(text):
    if SPEECH_ENABLED:
        os.system("espeak '{}' 2>/dev/null &".format(text))


def generate_session_graph(timestamps, areas, intervals):
    if len(timestamps) < 2:
        print("Not enough data collected to generate a graph.")
        return

    t0 = timestamps[0]
    rel_times = [t - t0 for t in timestamps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(rel_times, areas, color="#2e7d32")
    ax1.set_ylabel("Detected hand area (pixels)")
    ax1.set_title("Doppler Effect Demo - Session Recording")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rel_times, intervals, color="#1565c0")
    ax2.set_ylabel("Blink interval (seconds)")
    ax2.set_xlabel("Time (seconds since start)")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(GRAPH_OUTPUT_DIR, "doppler_session_{}.png".format(timestamp_str))
    try:
        plt.savefig(filename, dpi=120)
        print("Session graph saved to: {}".format(filename))
    except Exception as e:
        print("Could not save graph ({}). Trying /tmp instead...".format(e))
        fallback = os.path.join("/tmp", "doppler_session_{}.png".format(timestamp_str))
        plt.savefig(fallback, dpi=120)
        print("Session graph saved to: {}".format(fallback))


def run_demo():
    print_startup_banner()
    generate_all_tones()

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Check 'ls /dev/video*' and CAMERA_INDEX.")
        return

    print("Warming up camera ({} frames)...".format(NUM_FRAMES_TO_WARM_UP))
    for _ in range(NUM_FRAMES_TO_WARM_UP):
        cap.read()
        time.sleep(0.05)

    ret, sample_frame = cap.read()
    if not ret:
        print("ERROR: Could not read a frame from the camera.")
        cap.release()
        return
    actual_height, actual_width = sample_frame.shape[:2]
    print("Actual camera resolution: {}x{}".format(actual_width, actual_height))

    roi_coords = get_roi_coords(actual_width, actual_height)
    print("Detection zone (ROI): {}".format(roi_coords))

    min_area, max_area = run_calibration(cap, roi_coords)

    print("Doppler Effect Demo running.")
    print("LEFT hand controls pitch/blink rate. RIGHT hand (if visible) controls volume.")
    print("Press CTRL+C to stop.")
    print("")

    smoothed_area = 0.0
    smoothed_volume_area = 0.0
    last_volume_set = -1
    session_timestamps = []
    session_areas = []
    session_intervals = []

    prev_distance_cm = None
    prev_time = None
    last_spoken_direction = None
    last_spoken_time = 0.0

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame, retrying...")
                time.sleep(0.2)
                continue

            if TWO_HAND_MODE_ENABLED:
                hand_areas = get_up_to_two_hands(frame, roi_coords)
                raw_area = hand_areas[0] if len(hand_areas) >= 1 else 0
                raw_volume_area = hand_areas[1] if len(hand_areas) >= 2 else 0
                second_hand_visible = len(hand_areas) >= 2
            else:
                raw_area = get_hand_area(frame, roi_coords)
                raw_volume_area = 0
                second_hand_visible = False

            smoothed_area = (SMOOTHING_ALPHA * raw_area) + ((1 - SMOOTHING_ALPHA) * smoothed_area)
            smoothed_volume_area = (SMOOTHING_ALPHA * raw_volume_area) + ((1 - SMOOTHING_ALPHA) * smoothed_volume_area)

            step = map_area_to_step(smoothed_area, min_area, max_area)
            interval = step_to_blink_interval(step)
            detected = smoothed_area >= min_area

            distance_fraction = 0.0
            if max_area > min_area:
                distance_fraction = (smoothed_area - min_area) / float(max_area - min_area)
            distance_fraction = max(0.0, min(1.0, distance_fraction))

            # --- Volume from second (right) hand ---
            if second_hand_visible and max_area > min_area:
                volume_fraction = (smoothed_volume_area - min_area) / float(max_area - min_area)
                volume_fraction = max(0.0, min(1.0, volume_fraction))
                target_volume = VOLUME_FAR_PERCENT + volume_fraction * (VOLUME_NEAR_PERCENT - VOLUME_FAR_PERCENT)
            else:
                target_volume = VOLUME_NEAR_PERCENT  # default full volume in single-hand mode

            if abs(target_volume - last_volume_set) >= VOLUME_CHANGE_THRESHOLD:
                set_system_volume(target_volume)
                last_volume_set = target_volume

            bar = make_distance_bar(distance_fraction)

            # --- Physics overlay ---
            distance_cm = fraction_to_pseudo_distance_cm(distance_fraction)
            velocity_mps = 0.0
            if detected and prev_distance_cm is not None and prev_time is not None:
                dt = loop_start - prev_time
                if dt > 0:
                    velocity_mps = (prev_distance_cm - distance_cm) / 100.0 / dt

            real_freq = compute_real_doppler_freq(FAR_FREQ_HZ, velocity_mps)
            real_shift_hz = real_freq - FAR_FREQ_HZ
            demo_freq = FAR_FREQ_HZ + (step / float(NUM_TONE_STEPS - 1)) * (NEAR_FREQ_HZ - FAR_FREQ_HZ)

            second_hand_str = "yes (vol={:.0f}%)".format(target_volume) if second_hand_visible else "no"
            print("{}  detected={}  step={}/{}  right_hand={}".format(
                bar, detected, step, NUM_TONE_STEPS - 1, second_hand_str))
            print("  Physics: v={:+.3f} m/s | real shift={:+.5f} Hz (imperceptible) | demo tone (exaggerated)={:.0f} Hz".format(
                velocity_mps, real_shift_hz, demo_freq))

            # --- Voice announcement on direction change ---
            if detected:
                if velocity_mps > MIN_SPEED_FOR_DIRECTION_MPS:
                    direction = "approaching"
                elif velocity_mps < -MIN_SPEED_FOR_DIRECTION_MPS:
                    direction = "receding"
                else:
                    direction = None

                if (direction is not None
                        and direction != last_spoken_direction
                        and (loop_start - last_spoken_time) > SPEECH_COOLDOWN_SEC):
                    speak_async(direction)
                    last_spoken_direction = direction
                    last_spoken_time = loop_start
            else:
                last_spoken_direction = None

            prev_distance_cm = distance_cm
            prev_time = loop_start

            session_timestamps.append(loop_start)
            session_areas.append(smoothed_area)
            session_intervals.append(interval)

            led.pulse(fade_in_time=interval * 0.4, fade_out_time=interval * 0.4,
                      n=1, background=True)
            play_tone_file(tone_files[step])

            elapsed = time.time() - loop_start
            remaining = max(0, (interval * 2) - elapsed)
            time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        led.off()
        cap.release()
        generate_session_graph(session_timestamps, session_areas, session_intervals)


if __name__ == "__main__":
    run_demo()
