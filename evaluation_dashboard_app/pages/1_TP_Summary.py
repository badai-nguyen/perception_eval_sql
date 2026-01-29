import streamlit as st
import plotly.express as px
import pandas as pd

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
        index=2
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

# ========== Summary ==========
st.subheader("Summary")
count = len(df_f)
tp_mean = df_f[tp_col].mean() if count else 0.0
tp_median = df_f[tp_col].median() if count else 0.0
tp_min_val = df_f[tp_col].min() if count else 0.0
tp_max_val = df_f[tp_col].max() if count else 0.0

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Rows", f"{count:,}")
col_b.metric(f"{'Delta ' if use_delta else ''}TP mean", f"{tp_mean:.2f}")
col_c.metric(f"{'Delta ' if use_delta else ''}TP median", f"{tp_median:.2f}")
col_d.metric(f"{'Delta ' if use_delta else ''}TP range", f"{tp_min_val:.2f} to {tp_max_val:.2f}")

if use_delta and count:
    pos_rate = (df_f[tp_col] > 0).mean() * 100.0
    neg_rate = (df_f[tp_col] < 0).mean() * 100.0
    col_e, col_f = st.columns(2)
    col_e.metric("Delta positive (%)", f"{pos_rate:.1f}")
    col_f.metric("Delta negative (%)", f"{neg_rate:.1f}")

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

# ========== Scenario-level Delta Analysis (Compare Mode) ==========
if mode == "Compare Mode" and use_delta and "id" in df_cmp.columns:
    st.subheader("Per-Scenario TP Delta (B - A)")

    # Aggregate per scenario using mean delta TP
    scenario_delta = (
        df_cmp.groupby("id", as_index=False)["TP_delta"]
        .mean()
        .rename(columns={"TP_delta": "mean_TP_delta"})
    )

    # Attach counts (optionally show for context)
    scenario_counts = df_cmp.groupby("id")["TP_delta"].count().reset_index(name="Count")
    scenario_delta = pd.merge(scenario_delta, scenario_counts, on="id", how="left")

    # Sort/filter methods
    scen_sort = st.radio(
        "Scenario Ranking",
        [
            "Largest Absolute Change (|ΔTP|)",
            "Most Improved (Highest ΔTP)",
            "Most Degraded (Lowest ΔTP)",
            "Custom filter by name",
            "Show all"
        ],
        horizontal=True,
        key="tp_scen_sort"
    )
    scenario_delta_disp = scenario_delta.copy()
    n_scen = st.number_input("Top N scenarios to show", min_value=5, max_value=len(scenario_delta), value=20, key="delta_topN")
    if scen_sort == "Most Improved (Highest ΔTP)":
        scenario_delta_disp = scenario_delta.sort_values("mean_TP_delta", ascending=False).head(int(n_scen))
    elif scen_sort == "Most Degraded (Lowest ΔTP)":
        scenario_delta_disp = scenario_delta.sort_values("mean_TP_delta", ascending=True).head(int(n_scen))
    elif scen_sort == "Largest Absolute Change (|ΔTP|)":
        scenario_delta_disp = scenario_delta.reindex(scenario_delta["mean_TP_delta"].abs().sort_values(ascending=False).index).head(int(n_scen))
    elif scen_sort == "Custom filter by name":
        name_sub = st.text_input("Show scenarios containing (case-insensitive):", "", key="delta_scen_filter")
        if name_sub:
            scenario_delta_disp = scenario_delta[scenario_delta["Scenario"].str.contains(name_sub, case=False, na=False)]
    else:
        scenario_delta_disp = scenario_delta

    st.plotly_chart(
        px.bar(
            scenario_delta_disp,
            x="id",
            y="mean_TP_delta",
            text_auto=".2f",
            title="Mean ΔTP per Scenario (B - A)",
            color="mean_TP_delta",
            color_continuous_scale="RdYlGn",
            labels={"mean_TP_delta": "Mean ΔTP"}
        ),
        width="stretch"
    )
    st.dataframe(scenario_delta_disp, width="stretch")
