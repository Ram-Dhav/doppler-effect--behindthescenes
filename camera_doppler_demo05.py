"""
Doppler Effect Demonstration v4 - Raspberry Pi + USB Webcam + LED + Speaker
------------------------------------------------------------------------------
This version detects hand distance continuously using a webcam, and now
works reliably across DIFFERENT PEOPLE'S SKIN TONES - not just one
person's, which was a real limitation in earlier versions.

WHY EARLIER VERSIONS FAILED ACROSS DIFFERENT PEOPLE:
v3 used a FIXED skin-color range (tuned from one person's hand photo) to
find the hand in each frame. That works great for that one person, but
poorly for anyone with a noticeably different skin tone - their hand
partially or fully fails to match the color range, giving unreliable,
massively inconsistent readings between people.

THE FIX - MOTION-BASED DETECTION + PER-SESSION COLOR CALIBRATION:
Instead of assuming what "skin color" looks like in advance, this
version:
  1. Learns what the BACKGROUND looks like (Step 1 of calibration)
  2. Asks the current user to hold their hand close (Step 2) and finds
     the hand by detecting what's DIFFERENT from the background (motion/
     change detection) - this works regardless of skin tone, since it's
     not relying on color at all to find the hand initially.
  3. Samples the ACTUAL color of that detected hand region and uses it
     to set THIS PERSON'S skin-color range for the rest of the session.
This means every new person who runs the calibration gets detection
tuned to their own hand, not whoever was calibrated last.

WIRING:
  LED anode -> 1k ohm resistor -> GPIO18 (Pi Pin 12)
  LED cathode -> GND (Pi Pin 14)
  USB webcam -> any USB port
  Speaker/earphones -> Pi's 3.5mm audio jack

SETUP:
  sudo apt update
  sudo apt install python3-opencv
  python3 -c "import numpy" || sudo apt install python3-numpy
  sudo raspi-config -> System Options -> Audio -> Headphones

Requires: gpiozero, opencv (cv2), numpy
"""

from gpiozero import LED
import cv2
import numpy as np
import wave
import os
import time

# ---------------------------------------------------------------------
# GPIO PIN SETUP
# ---------------------------------------------------------------------
LED_PIN = 18
led = LED(LED_PIN)

# ---------------------------------------------------------------------
# CAMERA CONFIGURATION
# ---------------------------------------------------------------------
CAMERA_INDEX = 0
REQUESTED_WIDTH = 320
REQUESTED_HEIGHT = 240
ROI_MARGIN_FRACTION = 0.10   # ROI as a fraction of actual frame size
NUM_FRAMES_TO_WARM_UP = 10

# ---------------------------------------------------------------------
# SKIN COLOR RANGE - now determined PER SESSION by calibration below,
# not fixed in advance. These starting values are only used if
# calibration fails and falls back to a generic guess.
# ---------------------------------------------------------------------
FALLBACK_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
FALLBACK_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

# These are updated by run_calibration() and used by get_hand_area().
current_ycrcb_lower = FALLBACK_YCRCB_LOWER.copy()
current_ycrcb_upper = FALLBACK_YCRCB_UPPER.copy()

# ---------------------------------------------------------------------
# CALIBRATION CONFIGURATION
# ---------------------------------------------------------------------
CALIBRATION_COUNTDOWN_SEC = 3
CALIBRATION_SAMPLE_FRAMES = 20
MOTION_DIFF_THRESHOLD = 25      # grayscale difference to count as "moved"
MIN_MOTION_BLOB_AREA = 400      # ignore tiny noise blobs during calibration
COLOR_RANGE_MARGIN_CR = 8       # widen sampled Cr range by this much each side
COLOR_RANGE_MARGIN_CB = 8       # widen sampled Cb range by this much each side
BACKGROUND_SAFETY_MARGIN = 1.4  # MIN area threshold = background reading * this

FALLBACK_MIN_HAND_AREA = 800
FALLBACK_MAX_HAND_AREA = 15000

# ---------------------------------------------------------------------
# BLINK / TONE CONFIGURATION
# ---------------------------------------------------------------------
NEAR_BLINK_INTERVAL = 0.08
FAR_BLINK_INTERVAL = 0.6
NEAR_FREQ_HZ = 1000
FAR_FREQ_HZ = 300
TONE_DURATION = 0.12
NUM_TONE_STEPS = 8

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
    """Handles both OpenCV 3.x (3 return values) and 4.x (2 return values)."""
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
    """Find the largest blob matching the CURRENT session's skin-color range."""
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, current_ycrcb_lower, current_ycrcb_upper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = find_contours_compat(mask)
    if not contours:
        return 0

    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest)


def get_largest_motion_contour(roi_frame, background_gray_roi):
    """
    Find the largest region that's DIFFERENT from the background image,
    using plain grayscale differencing - no assumption about color at
    all, so this works regardless of skin tone.
    Returns (contour, mask) or (None, None) if nothing significant moved.
    """
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
    """
    Two-step guided calibration that works for ANY user's skin tone:
      Step 1: capture background reference frames (nothing in the ROI)
      Step 2: user holds hand close; the hand is found by MOTION
              (difference from background), then its ACTUAL color is
              sampled to set this session's skin-color range, and its
              area sets the 'near' threshold.
    Background frames are then re-checked using the new personalized
    color range to set a safe 'far/empty' threshold.
    Returns (min_area, max_area). Falls back to generic values if
    calibration doesn't produce a reliable result.
    """
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
        print("WARNING: Could not reliably detect your hand during calibration")
        print("(try better lighting, or hold your hand more centrally/closer).")
        print("Falling back to generic thresholds - detection may be less accurate.")
        return FALLBACK_MIN_HAND_AREA, FALLBACK_MAX_HAND_AREA

    # Derive this person's actual skin-color range from sampled pixels
    cr_low = max(0, int(np.percentile(sampled_cr, 5)) - COLOR_RANGE_MARGIN_CR)
    cr_high = min(255, int(np.percentile(sampled_cr, 95)) + COLOR_RANGE_MARGIN_CR)
    cb_low = max(0, int(np.percentile(sampled_cb, 5)) - COLOR_RANGE_MARGIN_CB)
    cb_high = min(255, int(np.percentile(sampled_cb, 95)) + COLOR_RANGE_MARGIN_CB)

    current_ycrcb_lower = np.array([0, cr_low, cb_low], dtype=np.uint8)
    current_ycrcb_upper = np.array([255, cr_high, cb_high], dtype=np.uint8)

    print("Personalized skin-color range set: Cr[{},{}] Cb[{},{}]".format(
        cr_low, cr_high, cb_low, cb_high))

    near_level = float(np.percentile(near_areas, 90))

    # Re-check the background using the NEW personalized color range,
    # so the 'empty frame' threshold is consistent with what we'll
    # actually be filtering by during the real demo.
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


def get_hand_area_from_roi(roi_frame):
    """Same color-matching as get_hand_area(), but takes an already-cropped ROI frame."""
    ycrcb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, current_ycrcb_lower, current_ycrcb_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours = find_contours_compat(mask)
    if not contours:
        return 0
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest)


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

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame, retrying...")
                time.sleep(0.2)
                continue

            area = get_hand_area(frame, roi_coords)
            step = map_area_to_step(area, min_area, max_area)
            interval = step_to_blink_interval(step)

            print("Hand blob area: {:.0f}  ->  step {}/{}".format(
                area, step, NUM_TONE_STEPS - 1))

            led.on()
            play_tone_file(tone_files[step])

            elapsed = time.time() - loop_start
            remaining = max(0, interval - elapsed)
            time.sleep(remaining)

            led.off()
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    finally:
        led.off()
        cap.release()


if __name__ == "__main__":
    run_demo()
