import joblib
import pandas as pd


# ==============================
# LOAD MODEL
# ==============================

MODEL_FILE = "orthoscan_movement_model.pkl"

model = joblib.load(MODEL_FILE)


# ==============================
# LOAD CAMERA FEATURES
# ==============================

features = pd.read_csv(
    "../movement/features.csv"
)

mean_angle = features["mean_knee_angle"].iloc[0]
knee_range = features["knee_range"].iloc[0]
angle_variation = features["knee_angle_variation"].iloc[0]


# ==============================
# PATIENT INPUT
# ==============================

print("================================")
print("        ORTHOSCAN SCREENING")
print("================================")

print("\nMovement features detected:")
print(f"Mean Knee Angle : {mean_angle:.2f} deg")
print(f"Knee Range      : {knee_range:.2f} deg")
print(f"Angle Variation : {angle_variation:.2f}")


pain = float(
    input("\nEnter pain score (0-10): ")
)

stiffness = float(
    input("Enter stiffness score (0-10): ")
)


# ==============================
# CREATE MODEL INPUT
# ==============================

patient = pd.DataFrame([{

    "mean_knee_angle": mean_angle,

    "knee_range": knee_range,

    "knee_angle_variation": angle_variation,

    "pain_score": pain,

    "stiffness_score": stiffness

}])


# ==============================
# PREDICTION
# ==============================

prediction = model.predict(patient)[0]


# ==============================
# RESULT
# ==============================

print("\n================================")
print("       ORTHOSCAN RESULT")
print("================================")

print(f"\nPreliminary Risk: {prediction.upper()}")

print("\n--------------------------------")
print("Movement + symptom assessment")
print("--------------------------------")

print("\nNote:")
print("This is a prototype screening result.")
print("It is NOT a medical diagnosis.")
print("Clinical validation is required.")