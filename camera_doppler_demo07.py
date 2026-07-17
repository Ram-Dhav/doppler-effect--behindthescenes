"""
Doppler Effect Demonstration v5 - Raspberry Pi + USB Webcam + LED + Speaker
------------------------------------------------------------------------------
Detects hand distance continuously via webcam (motion-based, per-session
skin-color calibration - works across different people's skin tones).

NEW IN THIS VERSION:
  1. LIVE DISTANCE METER - a text bar printed each frame showing your
     hand's relative distance at a glance, e.g. [||||||----------] 38%
  2. AUTO-GENERATED GRAPH - when you stop the demo (CTRL+C), it saves a
     PNG plot of hand-distance and blink-interval over time - useful
     evidence for your report showing the actual measured response.
  3. SMOOTH LED FADE (PWM) - the LED now fades in/out smoothly each
     pulse instead of switching hard on/off, while the PULSE RATE still
     represents "frequency" the same way blinking did before (faster
     pulses = hand closer). This keeps the Doppler-frequency meaning
     intact while looking more polished.

CARRIED FORWARD FROM EARLIER VERSIONS:
  - Motion-based hand detection + per-session skin-color calibration
    (works across different people, not just one)
  - OpenCV 3.x/4.x compatibility
  - ROI as % of actual camera resolution
  - Camera warm-up frames
  - Exponential smoothing on distance readings (less jitter)
  - 16-step resolution for gradual response

WIRING:
  LED anode -> 1k ohm resistor -> GPIO18 (Pi Pin 12)
  LED cathode -> GND (Pi Pin 14)
  USB webcam -> any USB port
  Speaker/earphones -> Pi's 3.5mm audio jack

SETUP:
  sudo apt update
  sudo apt install python3-opencv python3-matplotlib
  python3 -c "import numpy" || sudo apt install python3-numpy
  sudo raspi-config -> System Options -> Audio -> Headphones

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
led = PWMLED(LED_PIN)  # PWM-capable LED object, enables smooth fading

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
# GRAPH / LOGGING CONFIGURATION
# ---------------------------------------------------------------------
GRAPH_OUTPUT_DIR = "/home/pi"  # change if your home folder differs
DISTANCE_BAR_WIDTH = 24

TONE_DIR = "/tmp"
tone_files = []


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


def get_hand_area(frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    return get_hand_area_from_roi(roi)


def get_hand_area_from_roi(roi_frame):
    ycrcb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, current_ycrcb_lower, current_ycrcb_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours = find_contours_compat(mask)
    if not contours:
        return 0
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest)


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


def run_calibration(cap, roi_coords):
    global current_ycrcb_lower, current_ycrcb_upper
    x1, y1, x2, y2 = roi_coords

    print("")
    print("=== CALIBRATION ===")
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
        print("WARNING: Could not capture background frames. Using fallback thresholds.")
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA

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
        print("")
        print("WARNING: Could not reliably detect your hand during calibration.")
        print("Falling back to generic thresholds - detection may be less accurate.")
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA

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
        print("")
        print("WARNING: Calibration range too narrow to be reliable.")
        print("Falling back to default thresholds instead.")
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA

    print("")
    print("=== CALIBRATION COMPLETE ===")
    print("MIN_HAND_AREA = {:.0f}   MAX_HAND_AREA = {:.0f}".format(min_area, max_area))
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
    """Return a text bar like [||||||----------] 38% for quick visual feedback."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    bar = "|" * filled + "-" * (width - filled)
    return "[{}] {:3.0f}%".format(bar, fraction * 100)


def generate_session_graph(timestamps, areas, intervals):
    """Save a PNG plot of hand distance and blink interval over the session."""
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
    ax2.invert_yaxis()  # smaller interval (faster blink) drawn as "higher" on the graph
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

    print("Doppler Effect Demo (camera version) running.")
    print("Move your hand closer/farther from the camera. Press CTRL+C to stop.")
    print("(Run this script again each time a different person tries it,")
    print(" so it can recalibrate to their hand.)")
    print("")

    smoothed_area = 0.0
    session_timestamps = []
    session_areas = []
    session_intervals = []

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame, retrying...")
                time.sleep(0.2)
                continue

            raw_area = get_hand_area(frame, roi_coords)
            smoothed_area = (SMOOTHING_ALPHA * raw_area) + ((1 - SMOOTHING_ALPHA) * smoothed_area)

            step = map_area_to_step(smoothed_area, min_area, max_area)
            interval = step_to_blink_interval(step)
            detected = smoothed_area >= min_area

            distance_fraction = 0.0
            if max_area > min_area:
                distance_fraction = (smoothed_area - min_area) / float(max_area - min_area)

            bar = make_distance_bar(distance_fraction)
            print("{}  detected={}  step={}/{}".format(
                bar, detected, step, NUM_TONE_STEPS - 1))

            # Log for the end-of-session graph
            session_timestamps.append(loop_start)
            session_areas.append(smoothed_area)
            session_intervals.append(interval)

            # Smooth PWM fade pulse instead of a hard on/off blink. The
            # pulse RATE (how often it fires) still represents
            # "frequency" the same way blink rate did before.
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
