"""
AI Food Freshness Classification - Model Training Script
Loads food_freshness_dataset.csv, trains a Random Forest classifier,
evaluates it, and saves the trained model + encoders for later use.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ---------- 1. Load data ----------
df = pd.read_csv("food_freshness_dataset.csv")
print(f"Loaded {len(df)} rows, {df['freshness_label'].nunique()} classes")
print(df['freshness_label'].value_counts(), "\n")

# ---------- 2. Feature engineering ----------
df["ratio"] = df["days_since_purchase"] / df["typical_shelf_life_days"]

food_enc = LabelEncoder()
cat_enc = LabelEncoder()
label_enc = LabelEncoder()

df["food_item_enc"] = food_enc.fit_transform(df["food_item"])
df["category_enc"] = cat_enc.fit_transform(df["category"])
df["freshness_label_enc"] = label_enc.fit_transform(df["freshness_label"])

feature_cols = [
    "food_item_enc", "category_enc", "typical_shelf_life_days",
    "days_since_purchase", "ratio", "storage_temp_C", "humidity_percent",
    "color_score", "odor_score", "texture_score", "gas_sensor_ppm",
    "moisture_loss_percent"
]

X = df[feature_cols]
y = df["freshness_label_enc"]

# ---------- 3. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4. Train model ----------
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# ---------- 5. Evaluate ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}\n")
report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
print(report)

with open("evaluation_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.3f}\n\n")
    f.write(report)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(label_enc.classes_)))
ax.set_yticks(range(len(label_enc.classes_)))
ax.set_xticklabels(label_enc.classes_, rotation=30)
ax.set_yticklabels(label_enc.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
for i in range(len(label_enc.classes_)):
    for j in range(len(label_enc.classes_)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
fig.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:\n", importances)
importances.to_csv("feature_importance.csv")

# ---------- 6. Save model + encoders ----------
joblib.dump(model, "freshness_model.joblib")
joblib.dump(food_enc, "food_encoder.joblib")
joblib.dump(cat_enc, "category_encoder.joblib")
joblib.dump(label_enc, "label_encoder.joblib")

# Save food -> (category, shelf_life) lookup for the predict script / app
lookup = df.groupby("food_item").first()[["category", "typical_shelf_life_days"]]
lookup.to_dict(orient="index")
with open("food_lookup.json", "w") as f:
    json.dump(lookup.to_dict(orient="index"), f, indent=2)

print("\nSaved: freshness_model.joblib, encoders, food_lookup.json")
