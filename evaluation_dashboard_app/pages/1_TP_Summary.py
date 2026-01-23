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
# Fail early if compare data is incomplete
if mode == "Compare Mode" and (df_b is None or df_cmp is None):
    st.warning("Compare mode requires both Run B and delta data. Reload from Overview.")
    st.stop()
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
    use_delta = view == "Delta (B - A)"
else:
    df_active = df_a
    view = "Single Run"
    use_delta = False

# ========== Debug Controls ==========
show_debug = st.sidebar.checkbox("Show debug info", value=False)

if show_debug:
    st.subheader("Raw Data Check")
    st.dataframe(df_active.head(10), width="stretch")

# ========== Sidebar Filters ==========
st.sidebar.header("Filters")
tp_col = "TP_delta" if use_delta else "TP"
if tp_col not in df_active.columns:
    st.warning(f"Missing required column: {tp_col}")
    st.stop()
tp_values = df_active[tp_col]
tp_min_val = float(tp_values.min())
tp_max_val = float(tp_values.max())
if use_delta:
    default_range = (tp_min_val, tp_max_val)
else:
    default_low = max(tp_min_val, 0.0)
    default_high = min(tp_max_val, 100.0)
    default_range = (default_low, default_high) if default_low <= default_high else (tp_min_val, tp_max_val)

tp_min, tp_max = st.sidebar.slider(
    "TP delta range (%)" if use_delta else "TP range (%)",
    tp_min_val,
    tp_max_val,
    default_range,
)
clip_vel = st.sidebar.checkbox("Clip velocity outliers", value=True)

# ========== Data Filtering ==========
df_f = df_active[(df_active[tp_col] >= tp_min) & (df_active[tp_col] <= tp_max)].copy()

if clip_vel:
    vx_col = "vx_delta" if use_delta else "vx"
    vy_col = "vy_delta" if use_delta else "vy"
    for c in (vx_col, vy_col):
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
    xrms_col = "xrms_delta" if use_delta else "xrms"
    yrms_col = "yrms_delta" if use_delta else "yrms"
    fig_rms = px.scatter(
        df_f,
        x=xrms_col,
        y=yrms_col,
        color=tp_col,
        hover_data=["id"],
        labels={
            xrms_col: "Δ X RMS" if use_delta else "X RMS",
            yrms_col: "Δ Y RMS" if use_delta else "Y RMS",
            tp_col: "Δ TP" if use_delta else "TP",
        },
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_rms, width="stretch")

with col2:
    st.subheader("Velocity (vx vs vy)")
    vx_col = "vx_delta" if use_delta else "vx"
    vy_col = "vy_delta" if use_delta else "vy"
    fig_vel = px.scatter(
        df_f,
        x=vx_col,
        y=vy_col,
        color=tp_col,
        hover_data=["id"],
        labels={
            vx_col: "Δ Vx" if use_delta else "Vx",
            vy_col: "Δ Vy" if use_delta else "Vy",
            tp_col: "Δ TP" if use_delta else "TP",
        },
        color_continuous_scale="Plasma",
    )
    st.plotly_chart(fig_vel, width="stretch")

# ========== Metric Distribution ==========
st.subheader("Metric Distribution")
metrics = ["xstd", "ystd", "xrms", "yrms", "vx", "vy", "TP"]
metrics_delta = [f"{m}_delta" for m in metrics]
metric_options = metrics_delta if use_delta else metrics
metric = st.selectbox("Select metric", metric_options)

fig_hist = px.histogram(
    df_f,
    x=metric,
    nbins=30,
    color=tp_col,
    marginal="box"
)
st.plotly_chart(fig_hist, width="stretch")