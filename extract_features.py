import pandas as pd
import numpy as np

# ==============================
# LOAD DATA
# ==============================

INPUT_FILE = "movement_data.csv"
OUTPUT_FILE = "features.csv"

data = pd.read_csv(INPUT_FILE)

print("================================")
print("      ORTHOSCAN FEATURE EXTRACTION")
print("================================")

print("\nRaw samples:", len(data))


# ==============================
# REMOVE INVALID VALUES
# ==============================

data = data.dropna()

# Keep realistic pose measurements
# Very small angles are treated as tracking errors
data = data[
    (data["knee_angle"] >= 40) &
    (data["knee_angle"] <= 180)
]

print("Valid samples:", len(data))


# ==============================
# CALCULATE FEATURES
# ==============================

angles = data["knee_angle"]

min_angle = angles.min()
max_angle = angles.max()
mean_angle = angles.mean()
angle_range = max_angle - min_angle
angle_std = angles.std()


# ==============================
# MOVEMENT SCORE
# ==============================

# Larger variation means more movement
movement_variation = angle_std


# ==============================
# CREATE FEATURE TABLE
# ==============================

features = pd.DataFrame([{

    "min_knee_angle": round(min_angle, 2),

    "max_knee_angle": round(max_angle, 2),

    "mean_knee_angle": round(mean_angle, 2),

    "knee_range": round(angle_range, 2),

    "knee_angle_variation": round(
        movement_variation, 2
    ),

    "sample_count": len(data)

}])


# ==============================
# SAVE FEATURES
# ==============================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==============================
# DISPLAY RESULTS
# ==============================

print("\nMovement Features")
print("------------------------------")

for column in features.columns:

    print(
        f"{column}: {features[column].iloc[0]}"
    )

print("\nFeatures saved to:")
print(OUTPUT_FILE)