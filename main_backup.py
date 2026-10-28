
import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque


# ---------------- CONFIGURATION ----------------

PHONE_ALERT_SECONDS = 2.0
PHONE_GRACE_PERIOD = 0.5

DROWSY_ALERT_SECONDS = 2.0
EAR_THRESHOLD = 0.25

PHONE_CLASS_ID = 67  # YOLO class ID for cell phone

# ---------------- LOAD MODELS ----------------

print("Loading YOLO phone detection model...")
phone_model = YOLO("yolo11n.pt")

print("Loading MediaPipe face landmarker...")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ---------------- TIMER VARIABLES ----------------

phone_detected_start = None

phone_last_seen = None
phone_history = deque(maxlen=5)

eyes_closed_start = None

previous_timestamp_ms = 0

# ---------------- EAR CALCULATION ----------------

def calculate_ear(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) from six eye landmarks.
    """

    points = np.array([
        [landmarks[i].x, landmarks[i].y]
        for i in eye_indices
    ], dtype=np.float32)

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ---------------- START WEBCAM ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError(
        "Cannot open webcam. Check camera permissions or camera index."
    )

print("Driver Safety System started.")
print("Press 'q' to quit.")

try:
    start_time = time.time()

    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        current_time = time.time()

        # ==================================================
        # 1. PHONE DETECTION USING YOLO
        # ==================================================

        phone_found = False

        results = phone_model(
            frame,
            conf=0.45,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

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
                        "PHONE DETECTED",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2
                    )
        phone_history.append(phone_found)
        phone_found = sum(phone_history) >= 3
        # ==================================================
        # 2. PHONE TIMER WITH 0.5-SECOND GRACE PERIOD
        # ==================================================

        if phone_found:
            phone_last_seen = current_time

            if phone_detected_start is None:
                phone_detected_start = current_time

        elif phone_detected_start is not None:
            # Reset only after the phone has been missing
            # for longer than the grace period.
            if (
                phone_last_seen is not None
                and current_time - phone_last_seen
                > PHONE_GRACE_PERIOD
            ):
                phone_detected_start = None
                phone_last_seen = None

        if phone_detected_start is not None:
            phone_duration = (
                current_time - phone_detected_start
            )

            cv2.putText(
                frame,
                f"Phone timer: {phone_duration:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if phone_duration >= PHONE_ALERT_SECONDS:
                cv2.putText(
                    frame,
                    "PHONE USE ALERT!",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

        # ==================================================
        # 3. FACE LANDMARK DETECTION USING MEDIAPIPE
        # ==================================================

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (current_time - start_time) * 1000
        )

        # VIDEO mode requires increasing timestamps.
        timestamp_ms = max(
            timestamp_ms,
            previous_timestamp_ms + 1
        )
        previous_timestamp_ms = timestamp_ms

        face_result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ==================================================
        # 4. DROWSINESS DETECTION USING EAR
        # ==================================================

        if face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]

            # Draw face landmark points.
            height, width = frame.shape[:2]

            for point in landmarks:
                px = int(point.x * width)
                py = int(point.y * height)

                cv2.circle(
                    frame,
                    (px, py),
                    1,
                    (0, 255, 255),
                    -1
                )

            left_ear = calculate_ear(
                landmarks, LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks, RIGHT_EYE
            )

            average_ear = (left_ear + right_ear) / 2.0

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            if average_ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = current_time

                closed_duration = (
                    current_time - eyes_closed_start
                )

                cv2.putText(
                    frame,
                    f"EYES CLOSED: {closed_duration:.1f}s",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

                if closed_duration >= DROWSY_ALERT_SECONDS:
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

            else:
                eyes_closed_start = None

        else:
            # No face detected: reset the eye-closure timer.
            eyes_closed_start = None

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

        # ==================================================
        # 5. DISPLAY WEBCAM WINDOW
        # ==================================================

        cv2.imshow(
            "AI Driver Safety System",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("Driver Safety System stopped.")