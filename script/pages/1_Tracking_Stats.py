import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DEFAULT_A = "data/Summary.csv"
DEFAULT_B = "data/Summary_compare.csv"  # optional
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

# GLOBAL PAGE MODE
st.sidebar.header("Run Selection")

st.sidebar.caption("Default data path:")
st.sidebar.code(DEFAULT_A)
file_a_upload = st.sidebar.file_uploader(
    "Override Run A (optional)",
    type=["csv"],
    key="upload_a",
)

file_a = file_a_upload if file_a_upload else DEFAULT_A


mode = st.sidebar.radio(
    "Mode",
    ["Single Run", "Compare Runs"],
)

if mode == "Single Run":
    st.subheader("Single Run")
elif mode == "Compare Runs":
    st.subheader("Compare Runs")

# Load data

df_a = load_summary(file_a)

if mode == "Compare Runs":
    df_b = load_summary(DEFAULT_B)

st.subheader("Raw Data Check")
st.dataframe(df_a.head(10), width="stretch")


# Sidebar controls
st.sidebar.header("Filters")
tp_min, tp_max = st.sidebar.slider(
    "TP range (%)",
    float(df_a.TP.min()),
    float(df_a.TP.max()),
    (40.0, 80.0),
)

clip_vel = st.sidebar.checkbox("Clip velocity outliers", value=True)

# Filter
df_f = df_a[(df_a.TP >= tp_min) & (df_a.TP <= tp_max)]

# Optional velocity clipping (for insane outlier)
if clip_vel:
    for c in ["vx", "vy"]:
        q1, q99 = df_f[c].quantile([0.01, 0.99])
        df_f[c] = df_f[c].clip(q1, q99)

# Show table
st.subheader("Filtered Data")
st.dataframe(df_f, width="stretch")

# --- Scatter plots ---
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

# --- Distribution ---
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