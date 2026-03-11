import streamlit as st
import plotly.express as px
import pandas as pd
from lib.path_utils import path_display

st.set_page_config(layout="wide")
st.title("TP / Position / Velocity Statistics Viewer")

# ========== Safety Check ==========
if "runA" not in st.session_state:
    st.warning("Please load data from the Overview page first.")
    st.stop()

mode = st.session_state.get("mode", "Single Run")

runA = st.session_state["runA"]
if runA.get("summary") is None:
    st.warning("This run has no **Summary.csv**. Load a run that includes Summary.csv for this page. Detection Stats and Bounding Box Viewer work with parquet-only runs.")
    st.stop()
df_a = runA["summary"]
df_b = df_cmp = None
if mode == "Compare Mode":
    runB = st.session_state.get("runB")
    df_b = runB["summary"] if runB else None
    df_cmp = st.session_state.get("df_cmp")

st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{path_display(runA['path'])}`")
if mode == "Compare Mode":
    st.markdown(f"**Candidate (B):** `{path_display(runB['path'])}`")
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

# Handle case where min and max are equal (e.g., all TP values are the same)
if tp_min_val == tp_max_val:
    # Add a small epsilon so slider is not degenerate
    epsilon = 0.01 if tp_min_val != 0 else 1.0
    slider_min = tp_min_val - epsilon
    slider_max = tp_max_val + epsilon
    slider_default = (tp_min_val, tp_max_val)
else:
    slider_min = tp_min_val
    slider_max = tp_max_val
    slider_default = default_range

tp_min, tp_max = st.sidebar.slider(
    "TP delta range (%)" if use_delta else "TP range (%)",
    min_value=slider_min,
    max_value=slider_max,
    value=slider_default,
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
    st.subheader("Absolute Position RMS (X vs Y)")
    # Always compare the two sources side by side (before and after/delta)
    if use_delta:
        # Show both reference and target RMS comparisons for X and Y, as well as their deltas
        fig_rms_x_compare = px.scatter(
            df_f,
            x="xrms_B",
            y="xrms",
            color=tp_col,
            hover_data=["id", "xrms_delta", "yrms_delta"],
            labels={
                "xrms_B": "X RMS (B)",
                "xrms": "X RMS (A)",
                tp_col: "Δ TP",
                "xrms_delta": "Δ X RMS",
                "yrms_delta": "Δ Y RMS",
            },
            title="Scatter: X RMS (B) vs X RMS (A)",
            color_continuous_scale="Viridis",
        )
        fig_rms_x_compare.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig_rms_x_compare, width="stretch")
        fig_rms_y_compare = px.scatter(
            df_f,
            x="yrms_B",
            y="yrms",
            color=tp_col,
            hover_data=["id", "xrms_delta", "yrms_delta"],
            labels={
                "yrms_B": "Y RMS (B)",
                "yrms": "Y RMS (A)",
                tp_col: "Δ TP",
                "xrms_delta": "Δ X RMS",
                "yrms_delta": "Δ Y RMS",
            },
            title="Scatter: Y RMS (B) vs Y RMS (A)",
            color_continuous_scale="Viridis",
        )
        fig_rms_y_compare.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig_rms_y_compare, width="stretch")
    else:
        # Just show the submission's RMS (x/y) for standard analysis
        fig_rms = px.scatter(
            df_f,
            x="xrms",
            y="yrms",
            color=tp_col,
            hover_data=["id"],
            labels={
                "xrms": "X RMS",
                "yrms": "Y RMS",
                tp_col: "TP",
            },
            color_continuous_scale="Viridis",
        )
        fig_rms.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig_rms, width="stretch")

with col2:
    st.subheader("Velocity (vx vs vy)")

    def plot_velocity(df, vx, vy, vx_label, vy_label):
        fig = px.scatter(
            df,
            x=vx,
            y=vy,
            color=tp_col,
            hover_data=["id"],
            labels={
                vx: vx_label,
                vy: vy_label,
                tp_col: "TP",
            },
            color_continuous_scale="Plasma",
            title=f"{vx_label} vs {vy_label}",
        )
        st.plotly_chart(fig, width="stretch")

    if use_delta:
        plot_velocity(df_f, "vx", "vy", "Vx (A)", "Vy (A)")
        plot_velocity(df_f, "vx_B", "vy_B", "Vx (B)", "Vy (B)")
    else:
        plot_velocity(df_f, "vx", "vy", "Vx", "Vy")

# ========== Metric Distribution ==========
st.subheader("Metric Distribution")
metrics = ["xstd", "ystd", "xrms", "yrms", "vx", "vy", "TP"]
metrics_delta = [f"{m}_delta" for m in metrics]
metric_options = metrics_delta if use_delta else metrics
default_metric = "TP_delta" if use_delta else "TP"
if default_metric in metric_options:
    default_index = metric_options.index(default_metric)
else:
    default_index = 0
metric = st.selectbox("Select metric", metric_options, index=default_index)

# Show a simple, single-color (monochrome) distribution for clarity
fig_hist = px.histogram(
    df_f,
    x=metric,
    nbins=40,
    color_discrete_sequence=["#636EFA"],  # Plotly blue
    marginal="box",  # Adds a box plot at the top
    opacity=0.85,
)
fig_hist.update_layout(
    showlegend=False,
    bargap=0.04,
    xaxis_title=metric,
    yaxis_title="Count",
)
st.plotly_chart(fig_hist, width="stretch")

# Optionally, add a KDE/violin plot for more insight into distribution
st.subheader("Density (KDE/Violin)")
fig_density = px.violin(
    df_f,
    y=metric,
    box=True,
    points="all",
)
fig_density.update_layout(
    yaxis_title=metric,
    showlegend=False,
)
st.plotly_chart(fig_density, width="stretch")

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
