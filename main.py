
import cv2
import time
import math
import queue
import threading
import subprocess

from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


# ==================================================
# CONFIGURATION
# ==================================================

PHONE_CONFIDENCE = 0.45
PHONE_CLASS_ID = 67

EAR_THRESHOLD = 0.25
DROWSY_SECONDS = 2.0

PHONE_ALERT_SECONDS = 2.0
PHONE_GRACE_SECONDS = 0.5

VOICE_REPEAT_SECONDS = 5.0

MAR_THRESHOLD = 0.45
YAWN_SECONDS = 1.2

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH_CORNERS = [61, 291]
MOUTH_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]


# ==================================================
# VOICE ALERT SYSTEM - WINDOWS SPEECH
# ==================================================

voice_queue = queue.Queue()
queued_messages = set()
queue_lock = threading.Lock()


def speak(message):
    """Queue a warning, skipping identical pending messages."""
    with queue_lock:
        if message in queued_messages:
            print("Duplicate voice skipped:", message)
            return

        queued_messages.add(message)
        voice_queue.put(message)
        print("VOICE QUEUED:", message)


def voice_worker():
    """Play one Windows Speech message at a time."""

    while True:
        message = voice_queue.get()

        try:
            if message is None:
                break

            safe_message = message.replace("'", "''")

            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object "
                "System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Volume = 100; "
                "$speaker.Rate = 0; "
                f"$speaker.Speak('{safe_message}'); "
                "$speaker.Dispose()"
            )

            print("SPEAKING:", message)

            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )

            if result.returncode != 0:
                print("VOICE ERROR:", result.stderr)
            else:
                print("VOICE FINISHED:", message)

        except subprocess.TimeoutExpired:
            print("VOICE ERROR: Speech timed out.")

        except Exception as error:
            print("VOICE ERROR:", error)

        finally:
            with queue_lock:
                if message is not None:
                    queued_messages.discard(message)

            voice_queue.task_done()


voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)
voice_thread.start()


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def distance(point_a, point_b):
    return math.dist(point_a, point_b)


def get_point(landmarks, index, width, height):
    point = landmarks[index]
    return point.x * width, point.y * height


def calculate_ear(landmarks, eye_indices, width, height):
    points = [
        get_point(landmarks, index, width, height)
        for index in eye_indices
    ]

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (
        2.0 * horizontal
    )


def calculate_mar(landmarks, width, height):
    left_corner = get_point(
        landmarks, MOUTH_CORNERS[0], width, height
    )

    right_corner = get_point(
        landmarks, MOUTH_CORNERS[1], width, height
    )

    mouth_width = distance(left_corner, right_corner)

    if mouth_width == 0:
        return 0.0

    vertical_distances = []

    for upper_index, lower_index in MOUTH_VERTICAL_PAIRS:
        upper = get_point(
            landmarks, upper_index, width, height
        )

        lower = get_point(
            landmarks, lower_index, width, height
        )

        vertical_distances.append(distance(upper, lower))

    return sum(vertical_distances) / (
        len(vertical_distances) * mouth_width
    )


# ==================================================
# LOAD MODELS
# ==================================================

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

print("Loading face landmark model...")

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

face_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(
    face_options
)


# ==================================================
# START WEBCAM
# ==================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    landmarker.close()
    voice_queue.put(None)
    voice_thread.join(timeout=3)
    raise SystemExit(1)


# ==================================================
# STATE VARIABLES
# ==================================================

eyes_closed_start = None
drowsy_alert = False

yawn_start = None
yawn_alert = False

phone_start = None
last_phone_seen = 0.0
phone_alert = False

phone_history = deque(maxlen=5)

last_drowsy_voice = 0.0
last_phone_voice = 0.0
last_yawn_voice = 0.0

timestamp_ms = 0


# ==================================================
# MAIN LOOP
# ==================================================

try:
    print("AI Driver Safety System started.")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Cannot read webcam frame.")
            break

        current_time = time.monotonic()
        height, width = frame.shape[:2]

        # ------------------------------------------
        # FACE LANDMARK DETECTION
        # ------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = max(
            timestamp_ms + 1,
            int(time.monotonic() * 1000)
        )

        face_result = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]

            # DROWSINESS DETECTION
            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )

            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )

            average_ear = (left_ear + right_ear) / 2.0

            for index in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[index].x * width)
                y = int(landmarks[index].y * height)

                cv2.circle(
                    frame, (x, y), 2, (0, 255, 255), -1
                )

            if average_ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = current_time

                closed_duration = (
                    current_time - eyes_closed_start
                )

                drowsy_alert = (
                    closed_duration >= DROWSY_SECONDS
                )

                cv2.putText(
                    frame,
                    f"Eyes closed: {closed_duration:.1f}s",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

            else:
                eyes_closed_start = None
                drowsy_alert = False

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # YAWNING DETECTION
            mar = calculate_mar(
                landmarks, width, height
            )

            if mar >= MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = current_time

                yawn_duration = current_time - yawn_start

                yawn_alert = (
                    yawn_duration >= YAWN_SECONDS
                )

                cv2.putText(
                    frame,
                    f"Mouth open: {yawn_duration:.1f}s",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

            else:
                yawn_start = None
                yawn_alert = False

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:
            eyes_closed_start = None
            drowsy_alert = False

            yawn_start = None
            yawn_alert = False

        # ------------------------------------------
        # YOLO PHONE DETECTION
        # ------------------------------------------

        phone_found = False

        results = model.predict(
            frame,
            conf=PHONE_CONFIDENCE,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id == PHONE_CLASS_ID:
                    phone_found = True

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Cell phone {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        # Stabilize phone detection across frames
        phone_history.append(phone_found)
        stable_phone = sum(phone_history) >= 3

        if stable_phone:
            last_phone_seen = current_time

        phone_present = stable_phone or (
            phone_start is not None
            and current_time - last_phone_seen
            < PHONE_GRACE_SECONDS
        )

        if phone_present:
            if phone_start is None:
                phone_start = current_time

            phone_alert = (
                current_time - phone_start
                >= PHONE_ALERT_SECONDS
            )
        else:
            phone_start = None
            phone_alert = False

        # ------------------------------------------
        # RESET TIMERS WHEN ALERT ENDS
        # ------------------------------------------

        if not drowsy_alert:
            last_drowsy_voice = 0.0

        if not phone_alert:
            last_phone_voice = 0.0

        if not yawn_alert:
            last_yawn_voice = 0.0

        # ------------------------------------------
        # SEPARATE VOICE ALERTS
        # ------------------------------------------

        if (
            drowsy_alert
            and current_time - last_drowsy_voice
            >= VOICE_REPEAT_SECONDS
        ):
            speak(
                "Warning! Driver is feeling sleepy. "
                "Please take a break."
            )
            last_drowsy_voice = current_time

        if (
            phone_alert
            and current_time - last_phone_voice
            >= VOICE_REPEAT_SECONDS
        ):
            speak(
                "Warning! Please stop using your phone."
            )
            last_phone_voice = current_time

        if (
            yawn_alert
            and current_time - last_yawn_voice
            >= VOICE_REPEAT_SECONDS
        ):
            speak(
                "Warning! You may be yawning. "
                "Please stay alert."
            )
            last_yawn_voice = current_time

        # ------------------------------------------
        # DISPLAY ALERTS
        # ------------------------------------------

        if drowsy_alert:
            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        if phone_alert:
            cv2.putText(
                frame,
                "PHONE USE ALERT!",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        if yawn_alert:
            cv2.putText(
                frame,
                "YAWNING ALERT!",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

        cv2.imshow("AI Driver Safety System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Finish queued messages before stopping voice worker
    voice_queue.join()
    voice_queue.put(None)
    voice_thread.join(timeout=5)

    print("AI Driver Safety System stopped.")
