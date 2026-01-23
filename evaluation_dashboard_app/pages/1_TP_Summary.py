import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")
st.title("TP / Position / Velocity Statistics Viewer")

# ========== Safety Check ==========
if "runA" not in st.session_state:
    st.warning("Please load data from the Overview page first.")
    st.stop()

mode = st.session_state.get("mode", "Single Run")

runA = st.session_state["runA"]
df_a = runA["summary"]
df_b = df_cmp = None
if mode == "Compare Mode":
    runB = st.session_state.get("runB")
    df_b = runB["summary"] if runB else None
    df_cmp = st.session_state.get("df_cmp")

st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{runA['path']}`")
if mode == "Compare Mode":
    st.markdown(f"**Candidate (B):** `{runB['path']}`")
# ========== View Selector ==========
if mode == "Compare Mode":
    view = st.sidebar.selectbox(
        "View",
        ["Baseline (A)", "Candidate (B)", "Delta (B - A)"],
    )
    df_active = {
        "Baseline (A)": df_a,
        "Candidate (B)": df_b,
        "Delta (B - A)": df_cmp
    }.get(view, df_a)
else:
    df_active = df_a
    view = "Single Run"

# ========== Debug Controls ==========
show_debug = st.sidebar.checkbox("Show debug info", value=False)

if show_debug:
    st.subheader("Raw Data Check")
    st.dataframe(df_active.head(10), width="stretch")

# ========== Sidebar Filters ==========
st.sidebar.header("Filters")
tp_values = df_active["TP"]
tp_min, tp_max = st.sidebar.slider(
    "TP range (%)",
    float(tp_values.min()),
    float(tp_values.max()),
    (0.0, 100.0),
)
clip_vel = st.sidebar.checkbox("Clip velocity outliers", value=True)

# ========== Data Filtering ==========
df_f = df_active[(df_active["TP"] >= tp_min) & (df_active["TP"] <= tp_max)].copy()

if clip_vel:
    for c in ("vx", "vy"):
        # Avoid changing the original DataFrame dtype warning
        q1, q99 = df_f[c].quantile([0.01, 0.99]).values
        df_f[c] = df_f[c].clip(q1, q99)

if show_debug:
    st.subheader("Filtered Data")
    st.dataframe(df_f, width="stretch")

# ========== Plots ==========
col1, col2 = st.columns(2)

with col1:
    st.subheader("Position RMS (X vs Y)")
    fig_rms = px.scatter(
        df_f,
        x="xrms",
        y="yrms",
        color="TP",
        hover_data=["id"],
        labels={"xrms": "X RMS", "yrms": "Y RMS"},
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_rms, width="stretch")

with col2:
    st.subheader("Velocity (vx vs vy)")
    fig_vel = px.scatter(
        df_f,
        x="vx",
        y="vy",
        color="TP",
        hover_data=["id"],
        labels={"vx": "Vx", "vy": "Vy"},
        color_continuous_scale="Plasma",
    )
    st.plotly_chart(fig_vel, width="stretch")

# ========== Metric Distribution ==========
st.subheader("Metric Distribution")
metrics = ["xstd", "ystd", "xrms", "yrms", "vx", "vy", "TP"]
metric = st.selectbox("Select metric", metrics)

fig_hist = px.histogram(
    df_f,
    x=metric,
    nbins=30,
    color="TP",
    marginal="box"
)
st.plotly_chart(fig_hist, width="stretch")