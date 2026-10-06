from flask import Flask, render_template, Response, request, jsonify
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import pandas as pd
import joblib
import csv
import time
import subprocess
import sys
import threading

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVEMENT_DIR = os.path.join(
    BASE_DIR, "..", "movement"
)

POSE_MODEL_PATH = os.path.join(
    MOVEMENT_DIR,
    "models",
    "pose_landmarker.task"
)

AI_MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "orthoscan_movement_model.pkl"
)

FEATURES_PATH = os.path.join(
    MOVEMENT_DIR,
    "features.csv"
)

MOVEMENT_DATA_PATH = os.path.join(
    MOVEMENT_DIR,
    "movement_data.csv"
)

EXTRACT_SCRIPT = os.path.join(
    MOVEMENT_DIR,
    "extract_features.py"
)

RECORD_SECONDS = 10


# ============================================================
# GLOBAL STATE
# ============================================================

recording = False
recording_start = None
movement_data = []

recording_status = "Ready"

recording_lock = threading.Lock()


# ============================================================
# LOAD AI MODEL
# ============================================================

try:

    ai_model = joblib.load(
        AI_MODEL_PATH
    )

    print(
        "AI model loaded successfully."
    )

except Exception as e:

    ai_model = None

    print(
        "WARNING: AI model could not be loaded:",
        e
    )


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL_PATH
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    camera.release()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

if camera.isOpened():

    print(
        "Camera opened successfully."
    )

else:

    print(
        "WARNING: Camera could not be opened."
    )


# ============================================================
# KNEE ANGLE
# ============================================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(
            c[1] - b[1],
            c[0] - b[0]
        )
        -
        math.atan2(
            a[1] - b[1],
            a[0] - b[0]
        )
    )

    angle = abs(angle)

    if angle > 180:

        angle = 360 - angle

    return angle


# ============================================================
# GET FEATURES
# ============================================================

def get_features():

    try:

        if not os.path.exists(
            FEATURES_PATH
        ):

            return {
                "mean_angle": None,
                "knee_range": None,
                "variation": None,
                "available": False
            }


        data = pd.read_csv(
            FEATURES_PATH
        )


        if data.empty:

            return {
                "mean_angle": None,
                "knee_range": None,
                "variation": None,
                "available": False
            }


        return {

            "mean_angle":
                float(
                    data[
                        "mean_knee_angle"
                    ].iloc[0]
                ),

            "knee_range":
                float(
                    data[
                        "knee_range"
                    ].iloc[0]
                ),

            "variation":
                float(
                    data[
                        "knee_angle_variation"
                    ].iloc[0]
                ),

            "available": True
        }


    except Exception as e:

        print(
            "Feature read error:",
            e
        )

        return {

            "mean_angle": None,

            "knee_range": None,

            "variation": None,

            "available": False
        }


# ============================================================
# SAVE MOVEMENT DATA + EXTRACT FEATURES
# ============================================================

def save_movement_data():

    global movement_data
    global recording_status


    try:

        # Make a separate copy
        data_to_save = list(
            movement_data
        )


        print()

        print(
            "=========================================="
        )

        print(
            "Recording completed."
        )

        print(
            "=========================================="
        )

        print(
            "Samples recorded:",
            len(data_to_save)
        )


        # ----------------------------------------------------
        # CHECK DATA
        # ----------------------------------------------------

        if len(data_to_save) < 10:

            recording_status = (
                "Not enough movement data"
            )

            print(
                "ERROR: Not enough samples."
            )

            return


        # ----------------------------------------------------
        # SAVE CSV
        # ----------------------------------------------------

        with open(
            MOVEMENT_DATA_PATH,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([
                "time",
                "knee_angle"
            ])

            writer.writerows(
                data_to_save
            )


        print(
            "Movement data saved."
        )


        # ----------------------------------------------------
        # CHECK EXTRACTION SCRIPT
        # ----------------------------------------------------

        if not os.path.exists(
            EXTRACT_SCRIPT
        ):

            recording_status = (
                "Feature extraction script not found"
            )

            return


        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        recording_status = (
            "Processing recorded movement..."
        )


        print()

        print(
            "Running feature extraction..."
        )


        result = subprocess.run(

            [
                sys.executable,
                EXTRACT_SCRIPT
            ],

            cwd=MOVEMENT_DIR,

            capture_output=True,

            text=True
        )


        if result.stdout:

            print(
                result.stdout
            )


        if result.stderr:

            print(
                result.stderr
            )


        # ----------------------------------------------------
        # CHECK RESULT
        # ----------------------------------------------------

        if (

            result.returncode == 0

            and

            os.path.exists(
                FEATURES_PATH
            )

        ):

            recording_status = (
                "Assessment completed"
            )


            print()

            print(
                "=========================================="
            )

            print(
                "NEW ASSESSMENT READY"
            )

            print(
                "=========================================="
            )


        else:

            recording_status = (
                "Feature extraction failed"
            )


            print(
                "Feature extraction failed."
            )


    except Exception as e:

        recording_status = (
            "Recording error"
        )


        print(
            "Recording error:",
            e
        )


# ============================================================
# CAMERA STREAM
# ============================================================

def generate_frames():

    global recording
    global recording_start
    global movement_data


    while True:


        # ----------------------------------------------------
        # CAMERA FRAME
        # ----------------------------------------------------

        success, frame = camera.read()


        if not success:

            print(
                "Camera frame could not be read."
            )

            time.sleep(
                0.1
            )

            continue


        # ----------------------------------------------------
        # MIRROR
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(

            image_format=
                mp.ImageFormat.SRGB,

            data=rgb

        )


        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        try:

            result = landmarker.detect(
                mp_image
            )

        except Exception as e:

            print(
                "Pose detection error:",
                e
            )

            result = None


        current_angle = None


        # ====================================================
        # POSE DETECTED
        # ====================================================

        if (

            result

            and

            result.pose_landmarks

        ):


            landmarks = (
                result.pose_landmarks[0]
            )


            h, w, _ = frame.shape


            # Left leg
            hip = landmarks[23]
            knee = landmarks[25]
            ankle = landmarks[27]


            hip_point = (

                int(
                    hip.x * w
                ),

                int(
                    hip.y * h
                )

            )


            knee_point = (

                int(
                    knee.x * w
                ),

                int(
                    knee.y * h
                )

            )


            ankle_point = (

                int(
                    ankle.x * w
                ),

                int(
                    ankle.y * h
                )

            )


            # ------------------------------------------------
            # VISIBILITY
            # ------------------------------------------------

            if (

                hip.visibility > 0.4

                and

                knee.visibility > 0.4

                and

                ankle.visibility > 0.4

            ):


                current_angle = calculate_angle(

                    [
                        hip.x,
                        hip.y
                    ],

                    [
                        knee.x,
                        knee.y
                    ],

                    [
                        ankle.x,
                        ankle.y
                    ]

                )


                # --------------------------------------------
                # DRAW LANDMARKS
                # --------------------------------------------

                cv2.circle(

                    frame,

                    hip_point,

                    7,

                    (0, 255, 0),

                    -1

                )


                cv2.circle(

                    frame,

                    knee_point,

                    8,

                    (0, 255, 0),

                    -1

                )


                cv2.circle(

                    frame,

                    ankle_point,

                    7,

                    (0, 255, 0),

                    -1

                )


                # --------------------------------------------
                # DRAW LEG
                # --------------------------------------------

                cv2.line(

                    frame,

                    hip_point,

                    knee_point,

                    (255, 255, 255),

                    3

                )


                cv2.line(

                    frame,

                    knee_point,

                    ankle_point,

                    (255, 255, 255),

                    3

                )


                # --------------------------------------------
                # LIVE ANGLE
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Live Knee Angle: "
                    f"{current_angle:.1f} deg",

                    (20, 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.72,

                    (0, 255, 0),

                    2

                )


                cv2.putText(

                    frame,

                    "POSE DETECTED",

                    (20, 75),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 0),

                    2

                )


            else:


                cv2.putText(

                    frame,

                    "LOW VISIBILITY - "
                    "MOVE INTO FRAME",

                    (20, 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.62,

                    (0, 0, 255),

                    2

                )


        else:


            cv2.putText(

                frame,

                "PERSON NOT DETECTED",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2

            )


        # ====================================================
        # RECORDING
        # ====================================================

        if recording:


            elapsed = (

                time.time()

                -

                recording_start

            )


            # Save valid angle

            if current_angle is not None:

                movement_data.append(

                    [

                        round(
                            elapsed,
                            3
                        ),

                        round(
                            current_angle,
                            2
                        )

                    ]

                )


            # Remaining time

            remaining = max(

                0,

                RECORD_SECONDS
                -
                elapsed

            )


            cv2.putText(

                frame,

                f"RECORDING: "
                f"{remaining:.1f}s",

                (20, 115),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2

            )


            # ------------------------------------------------
            # COMPLETE
            # ------------------------------------------------

            if elapsed >= RECORD_SECONDS:


                recording = False


                recording_status = (
                    "Processing recorded movement..."
                )


                # Process in background

                threading.Thread(

                    target=
                        save_movement_data,

                    daemon=True

                ).start()


        # ====================================================
        # ENCODE
        # ====================================================

        success, buffer = cv2.imencode(

            ".jpg",

            frame

        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            +

            frame_bytes

            +

            b"\r\n"

        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    features = get_features()


    return render_template(

        "index.html",

        features=features,

        risk=None,

        pain="",

        stiffness=""

    )


# ============================================================
# VIDEO
# ============================================================

@app.route(
    "/video_feed"
)
def video_feed():

    return Response(

        generate_frames(),

        mimetype=(

            "multipart/x-mixed-replace; "
            "boundary=frame"

        )

    )


# ============================================================
# START RECORDING
# ============================================================

@app.route(

    "/start-recording",

    methods=["POST"]

)
def start_recording():

    global recording
    global recording_start
    global movement_data
    global recording_status


    with recording_lock:


        if recording:

            return jsonify({

                "success": False,

                "message":
                    "Recording already in progress."

            })


        # ----------------------------------------------------
        # CLEAR PREVIOUS MOVEMENT DATA
        # ----------------------------------------------------

        movement_data = []


        # ----------------------------------------------------
        # REMOVE OLD FEATURES
        # ----------------------------------------------------

        if os.path.exists(
            FEATURES_PATH
        ):

            try:

                os.remove(
                    FEATURES_PATH
                )

                print(
                    "Old features.csv removed."
                )

            except Exception as e:

                print(
                    "Could not remove "
                    "old features:",
                    e
                )


        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        recording = True

        recording_start = (
            time.time()
        )

        recording_status = (
            "Recording"
        )


    print()

    print(
        "=========================================="
    )

    print(
        "ORTHOSCAN RECORDING STARTED"
    )

    print(
        "Duration: 10 seconds"
    )

    print(
        "=========================================="
    )


    return jsonify({

        "success": True,

        "message":
            "10-second recording started."

    })


# ============================================================
# RECORDING STATUS
# ============================================================

@app.route(
    "/recording-status"
)
def recording_status_api():


    if recording:


        elapsed = (

            time.time()

            -

            recording_start

        )


        remaining = max(

            0,

            RECORD_SECONDS
            -
            elapsed

        )


        return jsonify({

            "recording": True,

            "status":
                "Recording",

            "remaining":
                round(
                    remaining,
                    1
                )

        })


    return jsonify({

        "recording": False,

        "status":
            recording_status,

        "remaining": 0

    })


# ============================================================
# FEATURES
# ============================================================

@app.route(
    "/features"
)
def features_api():

    return jsonify(
        get_features()
    )


# ============================================================
# AI + HYBRID SYMPTOM ANALYSIS
# ============================================================

@app.route(

    "/analyze",

    methods=["POST"]

)
def analyze():


    try:


        # ----------------------------------------------------
        # MODEL CHECK
        # ----------------------------------------------------

        if ai_model is None:

            return jsonify({

                "success": False,

                "message":
                    "AI model is not loaded."

            }), 500


        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

        pain = float(

            request.form[
                "pain"
            ]

        )


        stiffness = float(

            request.form[
                "stiffness"
            ]

        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not 0 <= pain <= 10:

            return jsonify({

                "success": False,

                "message":
                    "Pain score must be "
                    "between 0 and 10."

            }), 400


        if not 0 <= stiffness <= 10:

            return jsonify({

                "success": False,

                "message":
                    "Stiffness score must be "
                    "between 0 and 10."

            }), 400


        # ----------------------------------------------------
        # GET RECORDED FEATURES
        # ----------------------------------------------------

        features = get_features()


        if not features["available"]:

            return jsonify({

                "success": False,

                "message":
                    "Please complete a "
                    "10-second movement "
                    "recording first."

            }), 400


        # ----------------------------------------------------
        # CREATE AI INPUT
        # ----------------------------------------------------

        patient = pd.DataFrame([{

            "mean_knee_angle":
                features["mean_angle"],

            "knee_range":
                features["knee_range"],

            "knee_angle_variation":
                features["variation"],

            "pain_score":
                pain,

            "stiffness_score":
                stiffness

        }])


        # ----------------------------------------------------
        # RANDOM FOREST PREDICTION
        # ----------------------------------------------------

        ai_risk = ai_model.predict(

            patient

        )[0]


        ai_risk = str(
            ai_risk
        )


        # ====================================================
        # HYBRID PROTOTYPE DECISION
        # ====================================================

        if (

            pain >= 8

            and

            stiffness >= 8

        ):

            risk = "HIGH"

            decision_source = (
                "High symptom score"
            )


        elif (

            pain >= 6

            and

            stiffness >= 6

        ):

            risk = "MODERATE"

            decision_source = (
                "Moderate symptom score"
            )


        else:


            normalized_ai = (
                ai_risk.lower()
            )


            if normalized_ai == "low":

                risk = "LOW"


            elif normalized_ai in [

                "moderate",

                "medium"

            ]:

                risk = "MODERATE"


            elif normalized_ai == "high":

                risk = "HIGH"


            else:

                risk = ai_risk.upper()


            decision_source = (
                "Movement AI model"
            )


        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print()

        print(
            "=========================================="
        )

        print(
            "ORTHOSCAN AI SCREENING"
        )

        print(
            "=========================================="
        )

        print(
            f"Mean Knee Angle : "
            f"{features['mean_angle']:.2f}"
        )

        print(
            f"Knee Range      : "
            f"{features['knee_range']:.2f}"
        )

        print(
            f"Variation        : "
            f"{features['variation']:.2f}"
        )

        print(
            f"Pain             : "
            f"{pain}/10"
        )

        print(
            f"Stiffness        : "
            f"{stiffness}/10"
        )

        print(
            f"AI Prediction    : "
            f"{ai_risk}"
        )

        print(
            f"Final Risk       : "
            f"{risk}"
        )

        print(
            f"Decision Source  : "
            f"{decision_source}"
        )

        print(
            "=========================================="
        )


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "risk":
                risk,

            "ai_risk":
                ai_risk,

            "decision_source":
                decision_source,

            "features":
                features,

            "pain":
                pain,

            "stiffness":
                stiffness

        })


    except Exception as e:


        print(
            "Analysis error:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True

    )