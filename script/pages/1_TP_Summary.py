import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DTYPES = {
    "id": "string",
    "TP": "float64",
    "xave": "float64",
    "xstd": "float64",
    "xrms": "float64",
    "yave": "float64",
    "ystd": "float64",
    "yrms": "float64",
    "vx": "float64",
    "vy": "float64",
}

def load_summary(file_or_path):
    return pd.read_csv(
        file_or_path,
        sep=",",
        header=None,
        names=DTYPES.keys(),
        dtype=DTYPES,
    )

st.set_page_config(layout="wide")
st.title("TP / Position / Velocity Statistics Viewer")

# =========================
# Safety check
# =========================
if "runA" not in st.session_state:
    st.warning("Please load data from the Overview page first.")
    st.stop()

mode = st.session_state.get("mode", "Single Run")

runA = st.session_state["runA"]
df_a = runA["summary"]
if mode == "Compare Mode":
    runB = st.session_state.get("runB")
    df_b = runB["summary"]
    df_cmp = st.session_state.get("df_cmp")


# =========================
# View selector (compare only)
# =========================
if mode == "Compare Mode":
    view = st.sidebar.selectbox(
        "View",
        ["Baseline (A)", "Candidate (B)", "Delta (B - A)"],
    )

    if view == "Baseline (A)":
        df_active = df_a.copy()
    elif view == "Candidate (B)":
        df_active = df_b.copy()
    else:
        df_active = df_cmp.copy()
else:
    df_active = df_a.copy()
    view = "Single Run"

# =========================
# Debug controls
# =========================
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# =========================
# Header
# =========================
st.subheader(f"{view} View")

# =========================
# Raw data (if debug)
# =========================
if show_debug:
    st.subheader("Raw Data Check")
    st.dataframe(df_active.head(10), width="stretch")


# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")
tp_min, tp_max = st.sidebar.slider(
    "TP range (%)",
    float(df_active.TP.min()),
    float(df_active.TP.max()),
    (40.0, 80.0),
)

clip_vel = st.sidebar.checkbox("Clip velocity outliers", value=True)

# =========================
# Filtering
# =========================
df_f = df_active[(df_active.TP >= tp_min) & (df_active.TP <= tp_max)].copy()

# Optional velocity clipping (for insane outlier)
if clip_vel:
    for c in ["vx", "vy"]:
        q1, q99 = df_f[c].quantile([0.01, 0.99])
        df_f[c] = df_f[c].clip(q1, q99)

# =========================
# Filtered table
# =========================
st.subheader("Filtered Data")
st.dataframe(df_f, width="stretch")

# =========================
# Scatter plots
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Position RMS (X vs Y)")
    fig = px.scatter(
        df_f,
        x="xrms",
        y="yrms",
        color="TP",
        hover_data=["id"],
        labels={"xrms": "X RMS", "yrms": "Y RMS"},
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("Velocity (vx vs vy)")
    fig = px.scatter(
        df_f,
        x="vx",
        y="vy",
        color="TP",
        hover_data=["id"],
        labels={"vx": "Vx", "vy": "Vy"},
        color_continuous_scale="Plasma",
    )
    st.plotly_chart(fig, width="stretch")

# =========================
# Distribution
# =========================
st.subheader("Metric Distribution")
metric = st.selectbox(
    "Select metric",
    ["xstd", "ystd", "xrms", "yrms", "vx", "vy", "TP"],
)

fig = px.histogram(
    df_f,
    x=metric,
    nbins=30,
    color="TP",
    marginal="box"
)
st.plotly_chart(fig, width="stretch")