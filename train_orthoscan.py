import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


print("================================")
print("      ORTHOSCAN AI MODEL")
print("================================")


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    "../data/orthoscan_training.csv"
)


print("\nTraining samples:", len(data))

print("\nRisk distribution:")
print(data["risk"].value_counts())


# ============================================================
# FEATURES
# ============================================================

feature_columns = [

    "mean_knee_angle",

    "knee_range",

    "knee_angle_variation",

    "pain_score",

    "stiffness_score"

]


X = data[feature_columns]

y = data["risk"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.25,

    random_state=42,

    stratify=y

)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=8,

    min_samples_leaf=2,

    random_state=42,

    class_weight="balanced"

)


# ============================================================
# TRAIN
# ============================================================

model.fit(

    X_train,

    y_train

)


# ============================================================
# TEST
# ============================================================

prediction = model.predict(

    X_test

)


accuracy = accuracy_score(

    y_test,

    prediction

)


print("\n================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("================================")


print("\nClassification Report:")

print(

    classification_report(

        y_test,

        prediction,

        zero_division=0

    )

)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

for feature, importance in zip(

    feature_columns,

    model.feature_importances_

):

    print(

        f"{feature}: "
        f"{importance:.3f}"

    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    "orthoscan_movement_model.pkl"
)


joblib.dump(

    model,

    model_path

)


print("\nModel saved successfully!")

print(
    model_path
)