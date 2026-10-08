# PantryFresh AI Console (Streamlit)

## Setup
```
pip install -r requirements.txt
```

## Run
```
streamlit run app.py
```
This opens automatically in your browser at http://localhost:8501

Keep these files in the **same folder** as `app.py`:
`freshness_model.joblib`, `food_encoder.joblib`, `category_encoder.joblib`, `label_encoder.joblib`,
`food_lookup.json`, `supermarket_dataset.csv`, `evaluation_report.txt`, `confusion_matrix.png`.

## What's new (frontend upgrade)
- **🔮 Smart Predictor (new page)** — the actual trained `freshness_model.joblib` Random Forest
  now runs live in the app (previously it only existed in `predict.py`/`train_model.py`). Pick a
  food item, drag the "days since purchase" slider, optionally override storage temperature,
  humidity, and sensor readings, then hit **Run Prediction** to get a real model output: a
  confidence gauge, a per-class probability bar chart, retail advice, and a session-based
  prediction history log (clearable).
- **Interactive Plotly charts everywhere**, replacing plain text/progress bars:
  - Photo Scanner: horizontal probability bar chart for the CV heuristic result, plus a running
    "recent scans" history strip.
  - Batch Inventory: a status-mix donut chart and a per-category bar chart that update live with
    your filters, plus a new **sort-by** control (days remaining, name, stock) and a
    **⬇️ Download filtered CSV** button.
  - Mission & Impact: added a 4th KPI tile (₹ value moved via markdown vs. ₹ value composted) and
    a grouped bar chart of status breakdown by category.
  - Tech Specs: the feature-importance chart is now computed **live** from the loaded model
    (not a static image), and the confusion matrix image + full evaluation report are shown
    inline.
- Small "REAL RANDOM FOREST" / "CV HEURISTIC" pills on each analysis page so it's always clear
  which page is running the trained model vs. the transparent colour-heuristic — keeps the
  honesty note below still accurate.

## Files
- `app.py` — the full Streamlit dashboard (5 pages: Photo Scanner, Smart Predictor, Batch Inventory, Mission & Impact, Tech Specs)
- `predict.py` — standalone CLI version of the same real-model prediction logic used by Smart Predictor
- `supermarket_dataset.csv` — 300-row retail inventory dataset used by the Batch Inventory page
- `food_freshness_dataset.csv` — the original tabular training dataset (optional, for reference / train_model.py)
- `train_model.py` — trains the real Random Forest tabular classifier (see Tech Specs page)

## Important honesty note
The **Photo Scanner** page does NOT use a trained CNN (no MobileNetV2, no .pth weights file).
There was no food-image training dataset or internet access available to train or download one
in this environment. Instead it uses a transparent colour/brightness/spot-area heuristic
(see `heuristic_freshness()` in app.py) to estimate freshness from a photo — this is a legitimate
computer-vision technique, but it is NOT deep learning. Say this plainly if asked in your
project review; claiming it's a trained CNN would not hold up to a technical question.

The **Smart Predictor** page, by contrast, calls the real trained Random Forest
(`freshness_model.joblib`, 100% accuracy on the held-out test split — see
`evaluation_report.txt`) directly, so you now have a genuine live ML demo alongside the
transparent heuristic one.

To make the photo scanner a real CNN: download a labelled dataset (e.g. Kaggle "Fruits fresh and
rotten for classification"), train a MobileNetV2/ResNet with torchvision, export weights, and
load them in place of the heuristic function.
