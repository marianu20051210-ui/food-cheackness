"""
Predict freshness for a new food item using the trained model.
Usage: python3 predict.py
"""
import joblib
import json
import pandas as pd

model = joblib.load("freshness_model.joblib")
food_enc = joblib.load("food_encoder.joblib")
cat_enc = joblib.load("category_encoder.joblib")
label_enc = joblib.load("label_encoder.joblib")

with open("food_lookup.json") as f:
    food_lookup = json.load(f)


def predict_freshness(food_item, days_since_purchase, storage_temp_C=25,
                       humidity_percent=60, color_score=None, odor_score=None,
                       texture_score=None, gas_sensor_ppm=None, moisture_loss_percent=None):

    if food_item not in food_lookup:
        raise ValueError(f"Unknown food_item '{food_item}'. Known: {list(food_lookup.keys())}")

    info = food_lookup[food_item]
    category = info["category"]
    shelf_life = info["typical_shelf_life_days"]
    ratio = days_since_purchase / shelf_life

    # If sensor readings not supplied, estimate them from ratio (same relationship
    # used to build the training data) so the demo can run with minimal input.
    def est(v):
        return max(0, min(10, 10 * (1 - ratio))) if v is None else v

    color_score = est(color_score)
    odor_score = est(odor_score)
    texture_score = est(texture_score)
    gas_sensor_ppm = max(0, 50 * ratio) if gas_sensor_ppm is None else gas_sensor_ppm
    moisture_loss_percent = max(0, min(100, 60 * ratio)) if moisture_loss_percent is None else moisture_loss_percent

    row = pd.DataFrame([{
        "food_item_enc": food_enc.transform([food_item])[0],
        "category_enc": cat_enc.transform([category])[0],
        "typical_shelf_life_days": shelf_life,
        "days_since_purchase": days_since_purchase,
        "ratio": ratio,
        "storage_temp_C": storage_temp_C,
        "humidity_percent": humidity_percent,
        "color_score": color_score,
        "odor_score": odor_score,
        "texture_score": texture_score,
        "gas_sensor_ppm": gas_sensor_ppm,
        "moisture_loss_percent": moisture_loss_percent,
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    label = label_enc.inverse_transform([pred])[0]

    advice = {
        "Fresh": "Good to eat / store normally.",
        "Slightly Stale": "Use within 1-2 days, or cook/freeze it now to avoid waste.",
        "Spoiled": "Do not consume. Discard or compost.",
    }[label]

    return {
        "food_item": food_item,
        "prediction": label,
        "confidence": round(float(max(proba)), 3),
        "advice": advice,
    }


if __name__ == "__main__":
    # quick manual test - edit these values
    result = predict_freshness("Banana", days_since_purchase=5)
    print(result)

    result2 = predict_freshness("Milk", days_since_purchase=1)
    print(result2)

    result3 = predict_freshness("Chicken", days_since_purchase=4)
    print(result3)
