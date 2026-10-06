import cv2
import math
import csv
import time
import subprocess
import os
import sys
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# ORTHOSCAN SETTINGS
# ============================================================

MODEL_PATH = "models/pose_landmarker.task"

OUTPUT_FILE = "movement_data.csv"

RECORD_SECONDS = 10


# ============================================================
# CALCULATE KNEE ANGLE
# ============================================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: Pose Landmarker model not found!")
    print()
    print("Expected location:")
    print(os.path.abspath(MODEL_PATH))
    print()
    print("Make sure pose_landmarker.task is inside:")
    print("movement/models/")
    print()

    sys.exit()


# ============================================================
# MEDIAPIPE POSE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(
    options
)


# ============================================================
# START CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print()
    print("ERROR: Camera not detected!")
    print("Check whether another application is using the camera.")
    print()

    landmarker.close()
    sys.exit()


# ============================================================
# PROGRAM INFORMATION
# ============================================================

print()
print("==========================================")
print("              ORTHOSCAN")
print("       MOVEMENT DATA RECORDER")
print("==========================================")
print()
print("Camera started successfully.")
print()
print("Controls:")
print("R  = Start 10-second recording")
print("Q  = Quit")
print()


# ============================================================
# RECORDING VARIABLES
# ============================================================

recording = False

start_time = None

movement_data = []


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Failed to read camera frame.")
        break


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Detect pose
    # --------------------------------------------------------

    result = landmarker.detect(mp_image)


    current_angle = None


    # ========================================================
    # PERSON DETECTED
    # ========================================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]


        # ----------------------------------------------------
        # LEFT LEG LANDMARKS
        #
        # 23 = Left Hip
        # 25 = Left Knee
        # 27 = Left Ankle
        # ----------------------------------------------------

        left_hip = landmarks[23]

        left_knee = landmarks[25]

        left_ankle = landmarks[27]


        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        hip = [
            left_hip.x,
            left_hip.y
        ]

        knee = [
            left_knee.x,
            left_knee.y
        ]

        ankle = [
            left_ankle.x,
            left_ankle.y
        ]


        # ----------------------------------------------------
        # Calculate knee angle
        # ----------------------------------------------------

        current_angle = calculate_angle(
            hip,
            knee,
            ankle
        )


        # ----------------------------------------------------
        # Frame dimensions
        # ----------------------------------------------------

        h, w, _ = frame.shape


        # ----------------------------------------------------
        # Convert normalized coordinates to pixels
        # ----------------------------------------------------

        hip_point = (
            int(left_hip.x * w),
            int(left_hip.y * h)
        )

        knee_point = (
            int(left_knee.x * w),
            int(left_knee.y * h)
        )

        ankle_point = (
            int(left_ankle.x * w),
            int(left_ankle.y * h)
        )


        # ----------------------------------------------------
        # Draw hip
        # ----------------------------------------------------

        cv2.circle(
            frame,
            hip_point,
            7,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # Draw knee
        # ----------------------------------------------------

        cv2.circle(
            frame,
            knee_point,
            7,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # Draw ankle
        # ----------------------------------------------------

        cv2.circle(
            frame,
            ankle_point,
            7,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # Draw hip → knee
        # ----------------------------------------------------

        cv2.line(
            frame,
            hip_point,
            knee_point,
            (255, 255, 255),
            3
        )


        # ----------------------------------------------------
        # Draw knee → ankle
        # ----------------------------------------------------

        cv2.line(
            frame,
            knee_point,
            ankle_point,
            (255, 255, 255),
            3
        )


        # ----------------------------------------------------
        # Display knee angle
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Left Knee: {current_angle:.1f} deg",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # PERSON NOT DETECTED
    # ========================================================

    else:

        cv2.putText(
            frame,
            "Person not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # RECORDING PROCESS
    # ========================================================

    if recording:

        elapsed = time.time() - start_time


        # ----------------------------------------------------
        # Recording indicator
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"RECORDING: {elapsed:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # Save valid knee angle
        # ----------------------------------------------------

        if current_angle is not None:

            movement_data.append(
                [
                    round(elapsed, 3),
                    round(current_angle, 2)
                ]
            )


        # ----------------------------------------------------
        # Stop after 10 seconds
        # ----------------------------------------------------

        if elapsed >= RECORD_SECONDS:

            recording = False


            print()
            print("==========================================")
            print("Recording completed.")
            print("==========================================")


            # =================================================
            # SAVE RAW MOVEMENT DATA
            # =================================================

            with open(
                OUTPUT_FILE,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)


                writer.writerow(
                    [
                        "time",
                        "knee_angle"
                    ]
                )


                writer.writerows(
                    movement_data
                )


            print(
                f"Data saved to: {OUTPUT_FILE}"
            )

            print(
                f"Samples recorded: {len(movement_data)}"
            )


            # =================================================
            # AUTOMATIC FEATURE EXTRACTION
            # =================================================

            print()
            print("Running feature extraction...")


            extract_script = os.path.join(
                os.path.dirname(__file__),
                "extract_features.py"
            )


            if os.path.exists(extract_script):

                subprocess.run(
                    [
                        sys.executable,
                        extract_script
                    ]
                )

            else:

                print(
                    "WARNING: extract_features.py not found."
                )


            # -------------------------------------------------
            # Clear data for next recording
            # -------------------------------------------------

            movement_data = []


    # ========================================================
    # NOT RECORDING
    # ========================================================

    else:

        cv2.putText(
            frame,
            "Press R to record",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ========================================================
    # ORTHOSCAN TITLE
    # ========================================================

    cv2.putText(
        frame,
        "ORTHOSCAN - MOVEMENT ANALYSIS",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "OrthoScan - Movement Analysis",
        frame
    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # R = START RECORDING
    # --------------------------------------------------------

    if key == ord("r") and not recording:

        movement_data = []

        recording = True

        start_time = time.time()


        print()
        print("==========================================")
        print("Recording started...")
        print("Move naturally for 10 seconds.")
        print("==========================================")


    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()


print()
print("==========================================")
print("OrthoScan stopped.")
print("==========================================")