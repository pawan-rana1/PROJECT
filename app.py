# ══════════════════════════════════════════════════════════════════════════════
#  Battery SOC Prediction — Streamlit Web App
#  Model: Deep LSTM  |  Inputs: Voltage, Current, Temperature, Time
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from collections import deque

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery SOC Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* overall background */
.main { background-color: #0e1117; }

/* header banner */
.header-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
}
.header-box h1 { color: #e94560; margin: 0 0 4px; font-size: 1.9rem; }
.header-box p  { color: #a0aec0; margin: 0; font-size: 0.92rem; }

/* metric cards */
.soc-card {
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #2d3748;
}
.soc-critical { background: #2d1b1b; border-color: #fc4444; }
.soc-warning  { background: #2d2415; border-color: #f6ad55; }
.soc-nominal  { background: #1a2d1a; border-color: #48bb78; }

.soc-value { font-size: 3rem; font-weight: 700; line-height: 1; }
.soc-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
.soc-status { font-size: 0.9rem; font-weight: 600; margin-top: 6px; }

/* input section label */
.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #718096;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* gauge ring colors */
.critical-color { color: #fc4444; }
.warning-color  { color: #f6ad55; }
.nominal-color  { color: #48bb78; }

/* history table */
.hist-row { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIME_STEPS   = 10           # LSTM window size (must match training)
FEATURES     = ["Voltage(V)", "Current(A)", "Temperature(C)", "Time(s)"]
HISTORY_LEN  = 120          # points shown on live chart

# ── Load model & scaler ───────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model_lstm.keras")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler_lstm.pkl")

@st.cache_resource(show_spinner="Loading LSTM model …")
def load_assets():
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ── Session state: rolling buffer ─────────────────────────────────────────────
def init_state():
    if "buffer"   not in st.session_state:
        st.session_state.buffer   = deque(maxlen=TIME_STEPS)
    if "history"  not in st.session_state:
        st.session_state.history  = []          # list of dicts
    if "soc_now"  not in st.session_state:
        st.session_state.soc_now  = None
    if "step_n"   not in st.session_state:
        st.session_state.step_n   = 0

init_state()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🔋 Battery State of Charge — LSTM Predictor</h1>
  <p>Deep LSTM Neural Network &nbsp;|&nbsp; 10-step time-series window
     &nbsp;|&nbsp; Trained on 0 °C · 25 °C · 45 °C &nbsp;|&nbsp; R² ≈ 93.88%</p>
</div>
""", unsafe_allow_html=True)

# ── Try loading model ─────────────────────────────────────────────────────────
model_loaded = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

if not model_loaded:
    st.error(
        "**Model files not found.**  \n"
        f"Expected:  \n"
        f"- `{MODEL_PATH}`  \n"
        f"- `{SCALER_PATH}`  \n\n"
        "Run the full Jupyter notebook first to generate these files, "
        "then place them in the same folder as `app.py`."
    )
    st.stop()

lstm_model, scaler = load_assets()
st.sidebar.success("✅ LSTM model loaded")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Sensor Input Panel")
    st.caption("Adjust values and click **Add Reading** to feed the model.")

    st.markdown("---")

    voltage = st.slider(
        "⚡ Voltage (V)",
        min_value=2.80, max_value=4.30,
        value=3.75, step=0.01,
        help="Battery terminal voltage. Typical range 3.0 V (empty) → 4.2 V (full)."
    )
    st.markdown(f"<div style='font-size:0.78rem;color:#718096;'>Typical: 3.0 V (0%) → 4.2 V (100%)</div>",
                unsafe_allow_html=True)

    st.markdown("")

    current = st.slider(
        "🔌 Current (A)",
        min_value=-3.00, max_value=3.00,
        value=-1.00, step=0.05,
        help="Negative = discharging (driving). Positive = charging."
    )
    st.markdown(f"<div style='font-size:0.78rem;color:#718096;'>Negative = discharge &nbsp; Positive = charge</div>",
                unsafe_allow_html=True)

    st.markdown("")

    temperature = st.slider(
        "🌡️ Temperature (°C)",
        min_value=-10.0, max_value=60.0,
        value=25.0, step=0.5,
        help="Battery core temperature. Affects available capacity."
    )

    st.markdown("")

    elapsed_time = st.number_input(
        "⏱️ Elapsed Time (s)",
        min_value=0.0, max_value=100000.0,
        value=float(st.session_state.step_n * 10),
        step=10.0,
        help="Seconds since test / drive session started."
    )

    st.markdown("---")

    col_add, col_reset = st.columns(2)
    with col_add:
        add_btn = st.button("➕ Add Reading",
                            type="primary", use_container_width=True)
    with col_reset:
        reset_btn = st.button("🔄 Reset",
                              use_container_width=True)

    st.markdown("---")

    # Buffer fill indicator
    buf_len = len(st.session_state.buffer)
    st.markdown(f"**Buffer:** {buf_len}/{TIME_STEPS} readings")
    st.progress(buf_len / TIME_STEPS)
    if buf_len < TIME_STEPS:
        st.caption(f"Need {TIME_STEPS - buf_len} more readings before first prediction.")
    else:
        st.caption("Buffer full — predictions are live!")

    st.markdown("---")

    # CSV upload
    st.markdown("### 📂 Or upload a CSV")
    uploaded = st.file_uploader(
        "Upload battery log CSV",
        type=["csv"],
        help="Must have columns: Time(s), Voltage(V), Current(A), Temperature(C)"
    )

# ── Reset ─────────────────────────────────────────────────────────────────────
if reset_btn:
    st.session_state.buffer  = deque(maxlen=TIME_STEPS)
    st.session_state.history = []
    st.session_state.soc_now = None
    st.session_state.step_n  = 0
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def predict_soc(buffer_deque):
    """
    Given a deque of TIME_STEPS rows (each row = [V, I, T, t]),
    scale them and run the LSTM.
    Returns SOC as a float 0.0–1.0.
    """
    raw    = np.array(list(buffer_deque))          # (10, 4)
    scaled = scaler.transform(raw)                 # (10, 4)
    X      = scaled[np.newaxis, :, :]              # (1, 10, 4)
    soc    = float(lstm_model.predict(X, verbose=0).flatten()[0])
    return float(np.clip(soc, 0.0, 1.0))

def soc_meta(soc_val):
    pct = soc_val * 100
    if pct < 20:
        return pct, "soc-critical", "#fc4444", "🔴 Critical — Charge immediately"
    elif pct < 50:
        return pct, "soc-warning",  "#f6ad55", "🟡 Low — Consider charging soon"
    else:
        return pct, "soc-nominal",  "#48bb78", "🟢 Good — Normal operation"

# ── Handle manual button ───────────────────────────────────────────────────────
if add_btn:
    row = [voltage, current, temperature, elapsed_time]
    st.session_state.buffer.append(row)
    st.session_state.step_n += 1

    if len(st.session_state.buffer) == TIME_STEPS:
        soc = predict_soc(st.session_state.buffer)
        st.session_state.soc_now = soc
        st.session_state.history.append({
            "Step"          : st.session_state.step_n,
            "Time(s)"       : elapsed_time,
            "Voltage(V)"    : round(voltage, 3),
            "Current(A)"    : round(current, 3),
            "Temperature(C)": round(temperature, 2),
            "SOC(%)"        : round(soc * 100, 2)
        })

# ── Handle CSV upload ──────────────────────────────────────────────────────────
csv_results = None
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    required = set(FEATURES)
    if not required.issubset(df_up.columns):
        st.error(f"CSV must contain: {FEATURES}")
    else:
        df_up = df_up[FEATURES].dropna()
        scaled_all = scaler.transform(df_up[FEATURES].values)
        xs, preds = [], []
        for i in range(len(scaled_all) - TIME_STEPS):
            window = scaled_all[i:i + TIME_STEPS][np.newaxis, :, :]
            xs.append(i + TIME_STEPS)
            preds.append(float(np.clip(
                lstm_model.predict(window, verbose=0).flatten()[0], 0, 1)))
        csv_results = pd.DataFrame({
            "Row"          : xs,
            "Time(s)"      : df_up["Time(s)"].values[TIME_STEPS:],
            "Voltage(V)"   : df_up["Voltage(V)"].values[TIME_STEPS:],
            "Current(A)"   : df_up["Current(A)"].values[TIME_STEPS:],
            "Temperature(C)": df_up["Temperature(C)"].values[TIME_STEPS:],
            "SOC(%)"       : [round(p * 100, 2) for p in preds]
        })
        st.session_state.history = csv_results.to_dict("records")
        st.session_state.soc_now = preds[-1] if preds else None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

# ── Row 1 — current sensor values ─────────────────────────────────────────────
st.markdown("### 📡 Current Sensor Readings")
c1, c2, c3, c4 = st.columns(4)
c1.metric("⚡ Voltage",     f"{voltage:.3f} V")
c2.metric("🔌 Current",     f"{current:+.3f} A",
          delta="Charging" if current > 0 else "Discharging",
          delta_color="normal" if current > 0 else "inverse")
c3.metric("🌡️ Temperature", f"{temperature:.1f} °C")
c4.metric("⏱️ Elapsed Time", f"{elapsed_time:.0f} s")

st.markdown("---")

# ── Row 2 — SOC gauge ─────────────────────────────────────────────────────────
st.markdown("### 🎯 Predicted State of Charge")

if st.session_state.soc_now is not None:
    pct, css_cls, hex_color, status_text = soc_meta(st.session_state.soc_now)

    left_col, right_col = st.columns([1, 2])

    with left_col:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = pct,
            delta = {"reference": 50, "suffix": "%"},
            title = {"text": "SOC (%)", "font": {"color": "#e2e8f0", "size": 16}},
            number= {"suffix": "%", "font": {"color": hex_color, "size": 54}},
            gauge = {
                "axis" : {"range": [0, 100], "tickcolor": "#718096",
                          "tickfont": {"color": "#718096"}},
                "bar"  : {"color": hex_color, "thickness": 0.25},
                "bgcolor": "#1a202c",
                "bordercolor": "#2d3748",
                "steps": [
                    {"range": [0, 20],  "color": "#2d1b1b"},
                    {"range": [20, 50], "color": "#2d2415"},
                    {"range": [50, 100],"color": "#1a2d1a"},
                ],
                "threshold": {
                    "line" : {"color": "#fff", "width": 2},
                    "thickness": 0.8,
                    "value": pct
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor="#0e1117",
            plot_bgcolor ="#0e1117",
            margin=dict(t=40, b=10, l=20, r=20),
            font={"color": "#e2e8f0"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="soc-card {css_cls}" style="margin-top:0;">
          <div class="soc-status" style="color:{hex_color}; font-size:1rem;">
            {status_text}
          </div>
          <div style="font-size:0.78rem;color:#a0aec0;margin-top:8px;">
            <b>Buffer:</b> {len(st.session_state.buffer)}/{TIME_STEPS} readings<br>
            <b>Total steps:</b> {st.session_state.step_n}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        # SOC timeline chart
        if len(st.session_state.history) > 1:
            hist_df = pd.DataFrame(st.session_state.history[-HISTORY_LEN:])

            # Determine x-axis
            x_col = "Time(s)" if "Time(s)" in hist_df.columns else "Step"
            x_vals = hist_df[x_col]
            y_vals = hist_df["SOC(%)"]

            fig_line = go.Figure()

            # Coloured fill zones
            fig_line.add_hrect(y0=0,  y1=20,  fillcolor="#2d1b1b", opacity=0.4, line_width=0)
            fig_line.add_hrect(y0=20, y1=50,  fillcolor="#2d2415", opacity=0.4, line_width=0)
            fig_line.add_hrect(y0=50, y1=100, fillcolor="#1a2d1a", opacity=0.3, line_width=0)

            # Critical / warning lines
            fig_line.add_hline(y=20, line=dict(color="#fc4444", dash="dash", width=1.2),
                               annotation_text="Critical 20%", annotation_font_color="#fc4444")
            fig_line.add_hline(y=50, line=dict(color="#f6ad55", dash="dash", width=1.2),
                               annotation_text="Warning 50%", annotation_font_color="#f6ad55")

            # SOC line
            fig_line.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                line=dict(color=hex_color, width=2.5),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor=f"rgba({','.join(str(int(hex_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15)",
                name="SOC"
            ))

            fig_line.update_layout(
                title=dict(text="SOC Timeline (last readings)", font=dict(color="#e2e8f0")),
                xaxis=dict(title=x_col, color="#718096", gridcolor="#2d3748"),
                yaxis=dict(title="SOC (%)", range=[0, 105], color="#718096", gridcolor="#2d3748"),
                paper_bgcolor="#0e1117",
                plot_bgcolor ="#161b22",
                font=dict(color="#e2e8f0"),
                height=320,
                margin=dict(t=40, b=40, l=60, r=20),
                showlegend=False,
                hovermode="x unified"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("📊 Add more readings to see the SOC timeline chart.")

else:
    # Not enough data yet
    buf_len = len(st.session_state.buffer)
    remaining = TIME_STEPS - buf_len
    st.info(
        f"**Waiting for data.**  \n"
        f"The LSTM needs a {TIME_STEPS}-step window before making the first prediction.  \n"
        f"You have added **{buf_len}** readings — add **{remaining}** more, then the prediction will appear here."
    )

# ── Row 3 — parameter charts ──────────────────────────────────────────────────
if len(st.session_state.history) > 2:
    st.markdown("---")
    st.markdown("### 📈 Sensor Parameter History")

    hist_df = pd.DataFrame(st.session_state.history[-HISTORY_LEN:])
    x_col   = "Time(s)" if "Time(s)" in hist_df.columns else "Step"

    fig_params = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Voltage (V)", "Current (A)", "Temperature (°C)"),
        shared_xaxes=False
    )

    for col_name, color, row, col in [
        ("Voltage(V)",    "#63b3ed", 1, 1),
        ("Current(A)",    "#68d391", 1, 2),
        ("Temperature(C)","#fc8181", 1, 3),
    ]:
        if col_name in hist_df.columns:
            fig_params.add_trace(
                go.Scatter(x=hist_df[x_col], y=hist_df[col_name],
                           mode="lines", line=dict(color=color, width=2),
                           name=col_name),
                row=row, col=col
            )

    fig_params.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor ="#161b22",
        font=dict(color="#e2e8f0"),
        height=260,
        margin=dict(t=40, b=40, l=40, r=20),
        showlegend=False
    )
    for i in range(1, 4):
        fig_params.update_xaxes(title_text=x_col, gridcolor="#2d3748", row=1, col=i)
        fig_params.update_yaxes(gridcolor="#2d3748", row=1, col=i)

    st.plotly_chart(fig_params, use_container_width=True)

# ── Row 4 — data table + download ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📋 Prediction History")

    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df.tail(50), use_container_width=True, hide_index=True)

    csv_out = hist_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full History (CSV)",
        data=csv_out,
        file_name="soc_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:0.78rem;'>"
    "Battery SOC Predictor · Deep LSTM · Trained on SP20 Arbin Data · 0°C / 25°C / 45°C"
    "</div>",
    unsafe_allow_html=True
)
