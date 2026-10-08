import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
from datetime import date, datetime

import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PANTRY-FRESH • AI Waste Prevention Console",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# 2. STYLING
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #e6f9f0 0%, #f4fbf7 50%, #ffffff 100%);
        color: #064e3b;
    }

    .topbar {
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(20px);
        border: 1px solid #86efac;
        border-bottom: 3.5px solid #10b981;
        border-radius: 20px;
        padding: 14px 28px;
        display: flex; justify-content: space-between; align-items: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
        margin-bottom: 24px;
    }
    .brand-group { display: flex; align-items: center; gap: 12px; }
    .brand-logo {
        background: linear-gradient(135deg, #10b981, #047857);
        width: 44px; height: 44px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    }
    .brand-text { font-size: 20px; font-weight: 800; color: #064e3b; letter-spacing: -0.5px; }

    .card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #a7f3d0;
        border-bottom: 4px solid #10b981;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 15px 35px rgba(6, 78, 59, 0.06);
        margin-bottom: 20px;
        animation: fadeIn .35s ease;
    }
    @keyframes fadeIn { from { opacity:0; transform: translateY(6px);} to { opacity:1; transform: translateY(0);} }

    .fresh-badge {
        background: linear-gradient(135deg, #dcfce7, #86efac);
        color: #064e3b; border: 1.5px solid #22c55e;
        padding: 8px 22px; border-radius: 40px; font-weight: 800; font-size: 13px; display: inline-block;
    }
    .stale-badge {
        background: linear-gradient(135deg, #fef9c3, #fde68a);
        color: #7c5b06; border: 1.5px solid #eab308;
        padding: 8px 22px; border-radius: 40px; font-weight: 800; font-size: 13px; display: inline-block;
    }
    .spoiled-badge {
        background: linear-gradient(135deg, #fee2e2, #fca5a5);
        color: #7f1d1d; border: 1.5px solid #ef4444;
        padding: 8px 22px; border-radius: 40px; font-weight: 800; font-size: 13px; display: inline-block;
    }

    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0; }
    .metric-tile { background: #f0fdf4; border: 1px solid #86efac; border-radius: 16px; padding: 14px; text-align: center; }
    .metric-val { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 800; color: #047857; }
    .metric-lbl { font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-top: 2px; }

    .styled-table { width: 100%; border-collapse: collapse; margin: 18px 0; font-size: 13.5px;
        border-radius: 16px; overflow: hidden; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.08); border: 1px solid #bbf7d0; }
    .styled-table thead tr { background: linear-gradient(135deg, #10b981, #059669); color: #ffffff; text-align: left; font-weight: 700; }
    .styled-table th, .styled-table td { padding: 12px 16px; }
    .styled-table tbody tr { border-bottom: 1px solid #e2e8f0; background-color: #ffffff; }
    .styled-table tbody tr:nth-of-type(even) { background-color: #f8fafc; }

    .table-badge-fresh { background: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 12px; }
    .table-badge-sale { background: #fef9c3; color: #854d0e; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 12px; }
    .table-badge-compost { background: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 12px; }

    .model-pill {
        display:inline-block; font-size:11px; font-weight:800; letter-spacing:.3px;
        padding: 4px 12px; border-radius: 20px; margin-left: 8px; vertical-align: middle;
    }
    .pill-real { background:#dbeafe; color:#1e40af; border:1px solid #93c5fd; }
    .pill-heuristic { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }

    .hist-item {
        display:flex; justify-content:space-between; align-items:center;
        padding: 8px 14px; border-radius: 12px; background:#f8fafc; border:1px solid #e2e8f0;
        margin-bottom: 6px; font-size: 12.5px;
    }
    </style>
""", unsafe_allow_html=True)

CLASS_COLORS = {"Fresh": "#22c55e", "Slightly Stale": "#eab308", "Spoiled": "#ef4444"}

# ---------------------------------------------------------------------------
# 3. CACHED MODEL / DATA LOADERS  (the REAL trained Random Forest classifier)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tabular_assets():
    """Loads the trained RandomForest model + encoders produced by train_model.py."""
    try:
        model = joblib.load("freshness_model.joblib")
        food_enc = joblib.load("food_encoder.joblib")
        cat_enc = joblib.load("category_encoder.joblib")
        label_enc = joblib.load("label_encoder.joblib")
        with open("food_lookup.json") as f:
            food_lookup = json.load(f)
        feature_cols = [
            "food_item_enc", "category_enc", "typical_shelf_life_days",
            "days_since_purchase", "ratio", "storage_temp_C", "humidity_percent",
            "color_score", "odor_score", "texture_score", "gas_sensor_ppm",
            "moisture_loss_percent"
        ]
        return {
            "model": model, "food_enc": food_enc, "cat_enc": cat_enc,
            "label_enc": label_enc, "food_lookup": food_lookup,
            "feature_cols": feature_cols, "ok": True
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


ASSETS = load_tabular_assets()

RETAIL_ACTION = {
    "Fresh": ("Standard price. Display on prime shelves.", "success"),
    "Slightly Stale": ("Apply markdown (~40% off). Move to clearance section.", "warning"),
    "Spoiled": ("Pull from shelf. Route to compost/biogas diversion.", "error"),
}
ADVICE = {
    "Fresh": "Good to eat / store normally.",
    "Slightly Stale": "Use within 1-2 days, or cook/freeze it now to avoid waste.",
    "Spoiled": "Do not consume. Discard or compost.",
}


def predict_tabular(food_item, days_since_purchase, storage_temp_C=25, humidity_percent=60,
                     color_score=None, odor_score=None, texture_score=None,
                     gas_sensor_ppm=None, moisture_loss_percent=None):
    """Runs the REAL trained Random Forest classifier (same logic as predict.py)."""
    a = ASSETS
    info = a["food_lookup"][food_item]
    category = info["category"]
    shelf_life = info["typical_shelf_life_days"]
    ratio = days_since_purchase / shelf_life

    def est(v):
        return max(0, min(10, 10 * (1 - ratio))) if v is None else v

    color_score = est(color_score)
    odor_score = est(odor_score)
    texture_score = est(texture_score)
    gas_sensor_ppm = max(0, 50 * ratio) if gas_sensor_ppm is None else gas_sensor_ppm
    moisture_loss_percent = max(0, min(100, 60 * ratio)) if moisture_loss_percent is None else moisture_loss_percent

    row = pd.DataFrame([{
        "food_item_enc": a["food_enc"].transform([food_item])[0],
        "category_enc": a["cat_enc"].transform([category])[0],
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
    }])[a["feature_cols"]]

    pred = a["model"].predict(row)[0]
    proba = a["model"].predict_proba(row)[0]
    label = a["label_enc"].inverse_transform([pred])[0]
    classes = list(a["label_enc"].inverse_transform(a["model"].classes_))
    probs = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    return {
        "label": label, "category": category, "shelf_life": shelf_life, "ratio": ratio,
        "confidence": round(float(max(proba)) * 100, 1), "probs": probs,
        "advice": ADVICE[label],
    }


# ---------------------------------------------------------------------------
# 4. FRESHNESS HEURISTIC (photo mode)
#    NOTE: No trained CNN is used here. No food-image training dataset or
#    internet access was available to train/download one, so this uses a
#    transparent colour/spot-analysis heuristic instead of deep learning.
# ---------------------------------------------------------------------------
def heuristic_freshness(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize((160, 160))
    arr = np.asarray(img).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    brightness = (r + g + b) / 3
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx == 0, 0, (mx - mn) / np.maximum(mx, 1))

    dark_mask = brightness < 60
    brown_mask = (r > 60) & (r < 140) & (g > 30) & (g < 100) & (b < 70) & (r > g) & (g >= b)

    dark_ratio = dark_mask.mean()
    brown_ratio = brown_mask.mean()
    avg_bright = brightness.mean()
    avg_sat = sat.mean()

    spoil_score = (dark_ratio * 0.45 + brown_ratio * 0.35 +
                   max(0, (120 - avg_bright)) / 120 * 0.10 +
                   max(0, (0.25 - avg_sat)) / 0.25 * 0.10)
    spoil_score = float(np.clip(spoil_score, 0, 1))

    centers = np.array([0.0, 0.30, 0.62])
    logits = -((spoil_score - centers) ** 2) / (2 * 0.22 ** 2)
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()

    classes = ["Fresh", "Slightly Stale", "Spoiled"]
    label = classes[int(np.argmax(probs))]
    return {
        "label": label,
        "probs": dict(zip(classes, (probs * 100).round(1))),
        "dark_ratio": round(dark_ratio * 100, 1),
        "brown_ratio": round(brown_ratio * 100, 1),
    }


# ---------------------------------------------------------------------------
# 5. SMALL CHART / UI HELPERS
# ---------------------------------------------------------------------------
def confidence_gauge(value, label):
    color = CLASS_COLORS.get(label, "#10b981")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 34, "color": "#064e3b"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#f0fdf4",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#f1f5f9"},
                {"range": [50, 80], "color": "#e2f8ee"},
                {"range": [80, 100], "color": "#d0f5e2"},
            ],
        },
    ))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", font={"family": "Plus Jakarta Sans"})
    return fig


def probability_bar(probs: dict):
    classes = list(probs.keys())
    values = list(probs.values())
    fig = px.bar(
        x=values, y=classes, orientation="h",
        color=classes, color_discrete_map=CLASS_COLORS,
        text=[f"{v}%" for v in values],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=190, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 105], title=None, showgrid=False),
        yaxis=dict(title=None, categoryorder="array", categoryarray=classes[::-1]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans"},
    )
    return fig


def badge_html(label):
    cls = {"Fresh": "fresh-badge", "Slightly Stale": "stale-badge", "Spoiled": "spoiled-badge"}[label]
    return f'<div class="{cls}">● {label.upper()}</div>'


def log_history(key, entry, max_len=8):
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].insert(0, entry)
    st.session_state[key] = st.session_state[key][:max_len]


# ---------------------------------------------------------------------------
# 6. TOP NAV
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="topbar">
        <div class="brand-group">
            <div class="brand-logo">🥑</div>
            <span class="brand-text">PANTRY-FRESH AI CONSOLE</span>
        </div>
        <div style="font-size:12px; font-weight:700; color:#047857; background:#dcfce7; border:1px solid #86efac; padding:6px 14px; border-radius:20px;">
            ● WASTE PREVENTION SYSTEM ONLINE
        </div>
    </div>
""", unsafe_allow_html=True)

top_nav = st.radio(
    "Nav",
    ["📸 Photo Scanner", "🔮 Smart Predictor", "📦 Batch Inventory", "🌱 Mission & Impact", "🧬 Tech Specs"],
    horizontal=True, label_visibility="collapsed"
)
st.write("")

# ================= PAGE 1: PHOTO SCANNER =================
if top_nav == "📸 Photo Scanner":
    col_input, col_pred = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(
            '<div class="card"><h3 style="margin-top:0; color:#065f46;">📸 Image Input '
            '<span class="model-pill pill-heuristic">CV HEURISTIC</span></h3>',
            unsafe_allow_html=True)
        source_mode = st.radio("Source:", ["Upload Photo", "Use Camera"], horizontal=True)

        target_img = None
        if source_mode == "Upload Photo":
            uploaded_file = st.file_uploader("Select a food photo", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                target_img = Image.open(uploaded_file).convert("RGB")
        else:
            cam_feed = st.camera_input("Take a snapshot")
            if cam_feed:
                target_img = Image.open(cam_feed).convert("RGB")

        if target_img:
            st.image(target_img, caption="Input frame", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="card"><h3 style="margin-top:0; color:#065f46;">🔎 Freshness Analysis</h3>', unsafe_allow_html=True)

        if target_img:
            result = heuristic_freshness(target_img)
            label = result["label"]
            confidence = result["probs"][label]
            action_text, action_kind = RETAIL_ACTION[label]

            st.markdown(badge_html(label), unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-tile"><div class="metric-val">{confidence}%</div><div class="metric-lbl">Confidence</div></div>
                <div class="metric-tile"><div class="metric-val">{result['dark_ratio']}%</div><div class="metric-lbl">Dark-spot area</div></div>
                <div class="metric-tile"><div class="metric-val">{result['brown_ratio']}%</div><div class="metric-lbl">Brown-tone area</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.write("**Class confidence breakdown**")
            st.plotly_chart(probability_bar(result["probs"]), use_container_width=True, config={"displayModeBar": False})

            if action_kind == "success":
                st.success(f"🛒 **Retail action:** {action_text}")
            elif action_kind == "warning":
                st.warning(f"🏷️ **Retail action:** {action_text}")
            else:
                st.error(f"♻️ **Retail action:** {action_text}")

            log_history("photo_history", {
                "time": datetime.now().strftime("%H:%M:%S"),
                "label": label, "confidence": confidence,
            })
        else:
            st.info("Upload or capture a food photo to see the freshness analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get("photo_history"):
        st.markdown('<div class="card"><h4 style="margin-top:0;">🕓 Recent scans (this session)</h4>', unsafe_allow_html=True)
        for h in st.session_state["photo_history"]:
            st.markdown(
                f'<div class="hist-item"><span>{h["time"]}</span>{badge_html(h["label"])}<span>{h["confidence"]}% conf.</span></div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 2: SMART PREDICTOR (real trained model) =================
elif top_nav == "🔮 Smart Predictor":
    if not ASSETS.get("ok"):
        st.error(f"⚠️ Could not load the trained model artifacts: {ASSETS.get('error')}. "
                 f"Make sure freshness_model.joblib, the encoders, and food_lookup.json sit next to app.py.")
    else:
        col_input, col_pred = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown(
                '<div class="card"><h3 style="margin-top:0; color:#065f46;">🔮 Tabular Predictor '
                '<span class="model-pill pill-real">REAL RANDOM FOREST</span></h3>'
                '<p style="color:#64748b; font-size:12.5px; margin-top:-4px;">'
                'This calls the actual <code>freshness_model.joblib</code> trained in train_model.py — '
                '100% test accuracy (see evaluation_report.txt / Tech Specs tab).</p>',
                unsafe_allow_html=True)

            food_items = sorted(ASSETS["food_lookup"].keys())
            food_item = st.selectbox("Food item", food_items)
            shelf_life = ASSETS["food_lookup"][food_item]["typical_shelf_life_days"]
            category = ASSETS["food_lookup"][food_item]["category"]
            st.caption(f"Category: **{category}** · Typical shelf life: **{shelf_life} days**")

            days = st.slider("Days since purchase", 0, max(45, shelf_life * 2), min(3, shelf_life))

            with st.expander("⚙️ Storage conditions & sensor overrides"):
                storage_temp = st.slider("Storage temperature (°C)", 0, 40, 25)
                humidity = st.slider("Humidity (%)", 0, 100, 60)
                use_sensors = st.checkbox("Provide custom sensor readings (otherwise auto-estimated from ratio)")
                color_score = odor_score = texture_score = gas_ppm = moisture_loss = None
                if use_sensors:
                    c1, c2 = st.columns(2)
                    with c1:
                        color_score = st.slider("Colour score (0-10, 10=perfect)", 0.0, 10.0, 7.0)
                        odor_score = st.slider("Odor score (0-10, 10=neutral)", 0.0, 10.0, 7.0)
                        texture_score = st.slider("Texture score (0-10, 10=firm)", 0.0, 10.0, 7.0)
                    with c2:
                        gas_ppm = st.slider("Gas sensor (ppm)", 0.0, 100.0, 10.0)
                        moisture_loss = st.slider("Moisture loss (%)", 0.0, 100.0, 15.0)

            predict_clicked = st.button("🔮 Run Prediction", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_pred:
            st.markdown('<div class="card"><h3 style="margin-top:0; color:#065f46;">🎯 Prediction Result</h3>', unsafe_allow_html=True)

            if predict_clicked:
                result = predict_tabular(
                    food_item, days, storage_temp_C=storage_temp, humidity_percent=humidity,
                    color_score=color_score, odor_score=odor_score, texture_score=texture_score,
                    gas_sensor_ppm=gas_ppm, moisture_loss_percent=moisture_loss,
                )
                st.session_state["last_prediction"] = result
                log_history("predictor_history", {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "food": food_item, "days": days,
                    "label": result["label"], "confidence": result["confidence"],
                })

            result = st.session_state.get("last_prediction")
            if result:
                label = result["label"]
                action_text, action_kind = RETAIL_ACTION[label]

                st.markdown(badge_html(label), unsafe_allow_html=True)
                gc, bc = st.columns([1, 1])
                with gc:
                    st.plotly_chart(confidence_gauge(result["confidence"], label),
                                     use_container_width=True, config={"displayModeBar": False})
                with bc:
                    st.plotly_chart(probability_bar(result["probs"]),
                                     use_container_width=True, config={"displayModeBar": False})

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-tile"><div class="metric-val">{result['shelf_life']}d</div><div class="metric-lbl">Typical shelf life</div></div>
                    <div class="metric-tile"><div class="metric-val">{round(result['ratio']*100)}%</div><div class="metric-lbl">Shelf life used</div></div>
                    <div class="metric-tile"><div class="metric-val">{result['category']}</div><div class="metric-lbl">Category</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"💡 **Advice:** {result['advice']}")
                if action_kind == "success":
                    st.success(f"🛒 **Retail action:** {action_text}")
                elif action_kind == "warning":
                    st.warning(f"🏷️ **Retail action:** {action_text}")
                else:
                    st.error(f"♻️ **Retail action:** {action_text}")
            else:
                st.info("Set the inputs on the left and click **Run Prediction**.")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("predictor_history"):
            hc1, hc2 = st.columns([5, 1])
            with hc1:
                st.markdown('<div class="card"><h4 style="margin-top:0;">🕓 Recent predictions (this session)</h4>', unsafe_allow_html=True)
                hist_df = pd.DataFrame(st.session_state["predictor_history"])
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with hc2:
                st.write("")
                st.write("")
                if st.button("🗑️ Clear log"):
                    st.session_state["predictor_history"] = []
                    st.rerun()

# ================= PAGE 3: BATCH INVENTORY =================
elif top_nav == "📦 Batch Inventory":
    st.subheader("🏬 Inventory Freshness Matrix")
    st.caption("Reading live from 'supermarket_dataset.csv':")

    csv_path = "supermarket_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.warning("⚠️ 'supermarket_dataset.csv' not found next to app.py. Run generate_supermarket_data.py first.")
        df = pd.DataFrame()

    if not df.empty:
        categories = ["All Categories"] + sorted(df["category"].unique().tolist())
        col_c, col_s, col_o = st.columns([1, 2, 1.3])
        with col_c:
            selected_cat = st.selectbox("Filter category:", categories)
        with col_s:
            search_kw = st.text_input("🔍 Search item / mandi / SKU:", placeholder="e.g. Banana, Kolar, SKU1000...").strip().lower()
        with col_o:
            sort_choice = st.selectbox("Sort by:", [
                "Days remaining ↑", "Days remaining ↓", "Item A–Z", "Stock (high→low)"
            ])

        filtered = df.copy()
        if selected_cat != "All Categories":
            filtered = filtered[filtered["category"] == selected_cat]
        if search_kw:
            mask = (filtered["item"].str.lower().str.contains(search_kw) |
                    filtered["mandi"].str.lower().str.contains(search_kw) |
                    filtered["sku"].str.lower().str.contains(search_kw))
            filtered = filtered[mask]

        if sort_choice == "Days remaining ↑":
            filtered = filtered.sort_values("days", ascending=True)
        elif sort_choice == "Days remaining ↓":
            filtered = filtered.sort_values("days", ascending=False)
        elif sort_choice == "Item A–Z":
            filtered = filtered.sort_values("item", ascending=True)
        else:
            filtered = filtered.sort_values("stock", ascending=False)

        f_count = (filtered["status"] == "fresh").sum()
        s_count = (filtered["status"] == "sale").sum()
        c_count = (filtered["status"] == "compost").sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total filtered", f"{len(filtered)} SKUs")
        k2.metric("Fresh (full price)", f"{f_count} lots")
        k3.metric("Clearance markdown", f"{s_count} lots")
        k4.metric("Compost / waste", f"{c_count} lots")

        chart_c1, chart_c2 = st.columns([1, 1.4])
        with chart_c1:
            status_map = {"fresh": "Fresh (full price)", "sale": "Clearance", "compost": "Compost"}
            status_counts = filtered["status"].map(status_map).value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_donut = px.pie(
                status_counts, names="Status", values="Count", hole=0.62,
                color="Status",
                color_discrete_map={"Fresh (full price)": "#22c55e", "Clearance": "#eab308", "Compost": "#ef4444"},
            )
            fig_donut.update_traces(textinfo="percent+label", showlegend=False)
            fig_donut.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                                     title="Status mix (filtered)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        with chart_c2:
            cat_counts = filtered["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_cat = px.bar(cat_counts.sort_values("Count"), x="Count", y="Category", orientation="h",
                              color="Count", color_continuous_scale="Greens", text="Count")
            fig_cat.update_traces(textposition="outside", cliponaxis=False)
            fig_cat.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                                   title="SKUs by category (filtered)", coloraxis_showscale=False,
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

        dl1, dl2 = st.columns([1, 4])
        with dl1:
            st.download_button(
                "⬇️ Download filtered CSV", data=filtered.to_csv(index=False).encode("utf-8"),
                file_name=f"pantryfresh_inventory_{date.today().isoformat()}.csv", mime="text/csv",
                use_container_width=True,
            )

        items_per_page = 25
        total_pages = max(1, (len(filtered) - 1) // items_per_page + 1) if len(filtered) else 1
        current_page = st.number_input(f"Page (1 to {total_pages})", min_value=1, max_value=total_pages, value=1, step=1)
        start = (current_page - 1) * items_per_page
        page_items = filtered.iloc[start:start + items_per_page]

        rows_html = ""
        for _, r in page_items.iterrows():
            status = str(r["status"]).lower()
            if status == "fresh":
                badge = f'<span class="table-badge-fresh">{r["freshness"]} (Fresh)</span>'
            elif status == "sale":
                badge = f'<span class="table-badge-sale">{r["freshness"]} (Clearance)</span>'
            else:
                badge = f'<span class="table-badge-compost">{r["freshness"]} (Compost)</span>'

            rows_html += f"""
            <tr>
                <td><b>{r['sku']}</b></td>
                <td><b>{r['item']}</b><br><small style="color:#64748b;">Origin: {r['mandi']}</small></td>
                <td>{r['intake']}</td>
                <td>{r['expiry']}</td>
                <td><b>{r['days']}</b> days</td>
                <td>{r['stock']}</td>
                <td>{badge}</td>
                <td><strike style="color:#94a3b8; font-size:11px;">{r['mrp']}</strike> <b style="color:#047857;">{r['price']}</b><br>
                    <small style="color:#e11d48; font-weight:700;">{r['markdown']}</small></td>
                <td>{r['bay']}</td>
            </tr>
            """

        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th>SKU</th><th>Item & Origin</th><th>Intake</th><th>Expiry</th>
                <th>Remaining</th><th>Stock</th><th>Freshness</th><th>Price</th><th>Bay</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.caption(f"Showing {start + 1} to {min(start + items_per_page, len(filtered))} of {len(filtered)}")

# ================= PAGE 4: MISSION =================
elif top_nav == "🌱 Mission & Impact":
    st.markdown("""
        <div class="card">
            <h2 style="color:#064e3b; margin-top:0;">Cutting Avoidable Food Waste</h2>
            <p style="font-size:15px; color:#475569; line-height:1.7;">
                A large share of household and retail produce is discarded not because it has spoiled,
                but because nobody tracked it in time. PantryFresh combines a tabular freshness classifier
                (trained on shelf-life and sensor-style features) with a lightweight photo heuristic, so
                items can be flagged for use, markdown, or compost before they're wasted.
            </p>
        </div>
    """, unsafe_allow_html=True)

    csv_path = "supermarket_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        total_items = len(df)
        saved_from_waste = int((df["status"] == "sale").sum())

        def to_num(series):
            return pd.to_numeric(series.astype(str).str.replace("Rs.", "", regex=False), errors="coerce")

        mrp_num = to_num(df["mrp"])
        price_num = to_num(df["price"])
        value_recovered = float(price_num[df["status"] == "sale"].sum())
        value_at_risk = float(mrp_num[df["status"] == "compost"].sum())
    else:
        total_items, saved_from_waste, value_recovered, value_at_risk = 0, 0, 0.0, 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="card" style="text-align:center;"><h3 style="color:#047857; margin:0;">{total_items}</h3><p style="color:#64748b; font-size:12px; margin:4px 0 0 0;">SKUS TRACKED</p></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="card" style="text-align:center;"><h3 style="color:#047857; margin:0;">{saved_from_waste}</h3><p style="color:#64748b; font-size:12px; margin:4px 0 0 0;">LOTS CAUGHT BEFORE SPOILING</p></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="card" style="text-align:center;"><h3 style="color:#047857; margin:0;">Rs.{value_recovered:,.0f}</h3><p style="color:#64748b; font-size:12px; margin:4px 0 0 0;">RETAIL VALUE MOVED VIA MARKDOWN</p></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="card" style="text-align:center;"><h3 style="color:#b91c1c; margin:0;">Rs.{value_at_risk:,.0f}</h3><p style="color:#64748b; font-size:12px; margin:4px 0 0 0;">VALUE COMPOSTED (GOAL: MINIMIZE)</p></div>', unsafe_allow_html=True)

    if os.path.exists(csv_path) and not df.empty:
        st.markdown('<div class="card"><h4 style="margin-top:0;">📊 Status breakdown by category</h4>', unsafe_allow_html=True)
        status_map = {"fresh": "Fresh", "sale": "Clearance", "compost": "Compost"}
        plot_df = df.copy()
        plot_df["Status"] = plot_df["status"].map(status_map)
        grouped = plot_df.groupby(["category", "Status"]).size().reset_index(name="Count")
        fig = px.bar(
            grouped, x="category", y="Count", color="Status", barmode="group",
            color_discrete_map={"Fresh": "#22c55e", "Clearance": "#eab308", "Compost": "#ef4444"},
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ================= PAGE 5: TECH SPECS =================
elif top_nav == "🧬 Tech Specs":
    st.subheader("System Architecture")
    st.markdown("""
        <div class="card">
            <h4 style="color:#065f46; margin-top:0;">🧠 What's actually running under the hood</h4>
            <ul>
                <li><b>Tabular classifier (real, trained):</b> Random Forest on <code>food_freshness_dataset.csv</code>
                    — days-since-purchase, storage temp/humidity, and sensor-style features → Fresh / Slightly Stale / Spoiled.
                    It's wired live into the <b>🔮 Smart Predictor</b> tab above, not just described here.
                    See <code>train_model.py</code> in the project files.</li>
                <li><b>Photo scanner (heuristic, not deep learning):</b> colour/brightness/saturation and dark/brown spot-area
                    analysis on the uploaded image, converted to class probabilities. No CNN is used here because no food-image
                    training dataset or internet access was available in this environment to train or download one.</li>
                <li><b>Batch inventory page:</b> reads <code>supermarket_dataset.csv</code> directly — a CSV pipeline with
                    filter, search, sort, pagination and CSV export, no separate database needed for a project-scale demo.</li>
                <li><b>Upgrade path:</b> to make the photo scanner a real deep-learning model, train a MobileNetV2/ResNet on a
                    labelled fresh/rotten produce dataset (e.g. Kaggle's "Fruits fresh and rotten for classification"),
                    export the weights, and swap them in for <code>heuristic_freshness()</code> in app.py.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    if ASSETS.get("ok"):
        spec_c1, spec_c2 = st.columns([1, 1], gap="large")
        with spec_c1:
            st.markdown('<div class="card"><h4 style="margin-top:0;">📈 Feature importance (live from the loaded model)</h4>', unsafe_allow_html=True)
            model = ASSETS["model"]
            fi = pd.Series(model.feature_importances_, index=ASSETS["feature_cols"]).sort_values(ascending=True)
            fig_fi = px.bar(fi, x=fi.values, y=fi.index, orientation="h",
                             color=fi.values, color_continuous_scale="Emrld")
            fig_fi.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                                  xaxis_title="Importance", yaxis_title=None,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with spec_c2:
            st.markdown('<div class="card"><h4 style="margin-top:0;">🧾 Evaluation report</h4>', unsafe_allow_html=True)
            if os.path.exists("evaluation_report.txt"):
                with open("evaluation_report.txt") as f:
                    report_txt = f.read()
                first_line = report_txt.strip().splitlines()[0]
                st.markdown(f'<div class="metric-grid" style="grid-template-columns:1fr;"><div class="metric-tile"><div class="metric-val">{first_line}</div><div class="metric-lbl">Held-out test set</div></div></div>', unsafe_allow_html=True)
                st.code(report_txt, language=None)
            else:
                st.info("evaluation_report.txt not found next to app.py.")
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", caption="Confusion matrix on the held-out test split", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Model artifacts not found next to app.py — Smart Predictor and the live feature-importance chart need "
                   "freshness_model.joblib, the *_encoder.joblib files, and food_lookup.json in the same folder.")
