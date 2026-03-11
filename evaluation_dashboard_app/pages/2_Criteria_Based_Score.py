import streamlit as st
import pandas as pd
import plotly.express as px
from lib.path_utils import path_display

st.set_page_config(layout="wide")
st.title("Criteria-based Evaluation Viewer")

# =========================
# Safety check
# =========================
if "runA" not in st.session_state:
    st.warning("Please load data from the Overview page first.")
    st.stop()

mode = st.session_state.get("mode", "Single Run")

runA = st.session_state["runA"]
df_raw_A = runA.get("score")
if df_raw_A is None:
    st.warning("This run has no **Score.csv**. Load a run that includes Score.csv for this page. Detection Stats and Bounding Box Viewer work with parquet-only runs.")
    st.stop()

runB = st.session_state.get("runB")
df_raw_B = runB["score"] if runB else None

st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{path_display(runA['path'])}`")
if mode == "Compare Mode":
    st.markdown(f"**Candidate (B):** `{path_display(runB['path'])}`")
    if df_raw_B is None:
        st.warning("Compare Mode requires a Candidate (B) run from the Overview page.")
        st.stop()
# =========================
# Constants
# =========================

BASE_COLS = ["Scenario", "Option", "GT_OBJ"]

CRITERIA_COLS = [
    "distance",
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
    "obj_cnts",
]

CRITERIA_COUNT = 4

BLOCK_COLS = [
    "distance",
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
    "obj_cnts",
]

BLOCK_SIZE = len(CRITERIA_COLS)

criteria_idx = st.sidebar.selectbox(
    "Select Criteria",
    list(range(CRITERIA_COUNT)),
    format_func=lambda x: f"criteria{x}",
)

NUM_COLS = [
    "distance",
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
]
show_debug = st.sidebar.checkbox("Show debug info", value=False)


def build_view(df_raw, criteria_idx):
    start = 3 + criteria_idx * BLOCK_SIZE
    end = start + BLOCK_SIZE

    df_view = df_raw.iloc[:, :3].copy()
    df_view.columns = BASE_COLS

    block = df_raw.iloc[:, start:end].copy()
    block.columns = BLOCK_COLS

    df_view = pd.concat([df_view, block], axis=1)
    for c in NUM_COLS:
        df_view[c] = pd.to_numeric(df_view[c], errors="coerce")
    return df_view


st.sidebar.subheader("Visualization Controls")
metric = st.sidebar.selectbox(
    "Metric",
    NUM_COLS,
    index=NUM_COLS.index("pass_rate"),
)

group_by = st.sidebar.selectbox(
    "Group by",
    ["GT_OBJ", "Option"],
)

if mode == "Compare Mode":
    df_view_A = build_view(df_raw_A, criteria_idx)
    df_view_B = build_view(df_raw_B, criteria_idx)

    df_view_A["Run"] = "Baseline (A)"
    df_view_B["Run"] = "Candidate (B)"
    df_compare = pd.concat([df_view_A, df_view_B], axis=0, ignore_index=True)

    if show_debug:
        st.subheader("Raw Data Check — Baseline (A)")
        st.dataframe(df_raw_A.head(10), width="stretch")
        st.subheader("Raw Data Check — Candidate (B)")
        st.dataframe(df_raw_B.head(10), width="stretch")
        st.subheader(f"Criteria {criteria_idx} Data — Baseline (A)")
        st.dataframe(df_view_A, width="stretch")
        st.subheader(f"Criteria {criteria_idx} Data — Candidate (B)")
        st.dataframe(df_view_B, width="stretch")

    st.subheader("Summary")
    count_a = len(df_view_A)
    count_b = len(df_view_B)
    mean_a = df_view_A["pass_rate"].mean() if count_a else 0.0
    mean_b = df_view_B["pass_rate"].mean() if count_b else 0.0
    delta_mean = mean_b - mean_a
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (A)", f"{count_a:,}")
    col2.metric("Rows (B)", f"{count_b:,}")
    col3.metric("Pass rate mean (A)", f"{mean_a:.3f}")
    col4.metric("Pass rate mean (B)", f"{mean_b:.3f}", f"{delta_mean:+.3f}")

    compare_view = st.sidebar.radio(
        "Compare View",
        ["Overlay", "Delta"],
        horizontal=True,
    )

    if compare_view == "Overlay":
        st.subheader(f"{metric} Distribution (A vs B)")
        fig = px.histogram(
            df_compare,
            x=metric,
            color="Run",
            nbins=30,
            barmode="overlay",
            opacity=0.55,
            marginal="box",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader(f"Average {metric} by {group_by} (A vs B)")
        df_avg = (
            df_compare
            .groupby([group_by, "Run"], as_index=False)[metric]
            .mean()
            .sort_values(metric, ascending=False)
        )
        fig = px.bar(
            df_avg,
            x=group_by,
            y=metric,
            color="Run",
            barmode="group",
            text_auto=".2f",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pass Rate Overview (A vs B)")
        fig = px.box(
            df_compare,
            x=group_by,
            y="pass_rate",
            color="Run",
            points="all",
        )
        st.plotly_chart(fig, width="stretch")

        # ---- Improved Per Scenario Pass Rate Compare Plot ----
        st.subheader("Per Scenario Pass Rate Comparison (A vs B)")

        per_scenario_A = df_view_A.groupby("Scenario", as_index=False)["pass_rate"].mean()
        per_scenario_B = df_view_B.groupby("Scenario", as_index=False)["pass_rate"].mean()
        per_scenario = per_scenario_A.merge(per_scenario_B, on="Scenario", suffixes=("_A", "_B"))
        per_scenario["delta"] = per_scenario["pass_rate_B"] - per_scenario["pass_rate_A"]

        # Let user filter/sort the scenario comparison by various methods to make it easier to see
        filter_method = st.radio(
            "Scenario Filter/Sort", 
            ["All", "Top N by Delta", "Top N by Baseline (A)", "Custom contains string"],
            horizontal=True,
        )

        if filter_method == "Top N by Delta":
            N = st.number_input("Show Top N Scenarios by Delta (|delta|):", min_value=5, max_value=100, value=20)
            per_scenario = per_scenario.reindex(per_scenario["delta"].abs().sort_values(ascending=False).index)
            per_scenario_vis = per_scenario.head(N)
        elif filter_method == "Top N by Baseline (A)":
            N = st.number_input("Show Top N Scenarios by Baseline (A) Pass Rate:", min_value=5, max_value=100, value=20)
            per_scenario = per_scenario.sort_values("pass_rate_A", ascending=False)
            per_scenario_vis = per_scenario.head(N)
        elif filter_method == "Custom contains string":
            search = st.text_input("Show scenarios with name containing (case-insensitive):", "")
            per_scenario_vis = per_scenario[per_scenario["Scenario"].str.contains(search, case=False, na=False)] if search else per_scenario
        else:
            per_scenario_vis = per_scenario.copy()

        # Use bar plot instead of scatter, for more useful comparison if there are many scenarios
        per_scenario_vis_long = pd.melt(
            per_scenario_vis,
            id_vars=["Scenario"],
            value_vars=["pass_rate_A", "pass_rate_B"],
            var_name="Run",
            value_name="pass_rate",
        )
        per_scenario_vis_long["Run"] = per_scenario_vis_long["Run"].map({"pass_rate_A": "Baseline (A)", "pass_rate_B": "Candidate (B)"})
        per_scenario_vis_long = per_scenario_vis_long.sort_values(["Scenario", "Run"])

        fig = px.bar(
            per_scenario_vis_long,
            x="Scenario",
            y="pass_rate",
            color="Run",
            barmode="group",
            text_auto=".2f",
            title="Per Scenario Pass Rate by Run (filtered)",
        )
        st.plotly_chart(fig, width="stretch")

        # Show the delta for each scenario as a separate barplot below, sorted by delta
        st.subheader("Per Scenario Pass Rate Delta (B - A)")
        fig2 = px.bar(
            per_scenario_vis.sort_values("delta", key=abs, ascending=False),
            x="Scenario",
            y="delta",
            color="delta",
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
            title="Delta (B - A)"
        )
        st.plotly_chart(fig2, width="stretch")

        # Optionally show a table for the user to inspect
        with st.expander("Show Table: Per Scenario Pass Rates and Delta"):
            # Show dataframe with delta values, but avoid styling due to colormap compatibility issue
            st.dataframe(
                per_scenario_vis[["Scenario", "pass_rate_A", "pass_rate_B", "delta"]],
                width="stretch"
            )

        # The user can still optionally "restore" the scatter plot if they want
        if st.checkbox("Show scatter plot (Baseline (A) Pass Rate vs Candidate (B))", value=False):
            scatter_fig = px.scatter(
                per_scenario_vis,
                x="pass_rate_A",
                y="pass_rate_B",
                text="Scenario",
                labels={
                    "pass_rate_A": "Baseline (A) Pass Rate",
                    "pass_rate_B": "Candidate (B) Pass Rate",
                },
                title="Per Scenario Pass Rate: Baseline (A) vs Candidate (B) (filtered)",
            )
            scatter_fig.add_shape(
                type="line",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(dash='dash', color='gray'),
                xref="x", yref="y"
            )
            scatter_fig.update_traces(textposition="top center")
            st.plotly_chart(scatter_fig, width="stretch")
        # ---- End Improved Per Scenario Pass Rate Compare Plot ----

        st.download_button(
            "Download filtered data as CSV",
            df_compare.to_csv(index=False).encode("utf-8"),
            file_name="criteria_compare_filtered.csv",
            mime="text/csv",
        )

    else:
        merged = df_view_A.merge(
            df_view_B,
            on=BASE_COLS,
            suffixes=("_A", "_B"),
            how="inner",
        )
        merged[f"{metric}_delta"] = merged[f"{metric}_B"] - merged[f"{metric}_A"]
        merged["pass_rate_delta"] = merged["pass_rate_B"] - merged["pass_rate_A"]

        st.subheader(f"{metric} Delta Distribution (B - A)")
        fig = px.histogram(
            merged,
            x=f"{metric}_delta",
            nbins=30,
            marginal="box",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader(f"Average {metric} Delta by {group_by}")
        df_delta = (
            merged
            .groupby(group_by, as_index=False)[f"{metric}_delta"]
            .mean()
            .sort_values(f"{metric}_delta", ascending=False)
        )
        fig = px.bar(
            df_delta,
            x=group_by,
            y=f"{metric}_delta",
            text_auto=".2f",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pass Rate Delta Overview")
        fig = px.box(
            merged,
            x=group_by,
            y="pass_rate_delta",
            points="all",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Largest Absolute Deltas")
        top_changes = merged.copy()
        top_changes["abs_delta"] = top_changes[f"{metric}_delta"].abs()
        top_changes = top_changes.sort_values("abs_delta", ascending=False).head(20)
        cols = BASE_COLS + [f"{metric}_A", f"{metric}_B", f"{metric}_delta"]
        if metric != "pass_rate":
            cols.append("pass_rate_delta")
        st.dataframe(
            top_changes[
                cols
            ],
            width="stretch",
        )

else:
    df_view = build_view(df_raw_A, criteria_idx)
    if show_debug:
        st.subheader("Raw Data Check — Single Mode")
        st.dataframe(df_raw_A.head(10), width="stretch")
        st.subheader(f"Criteria {criteria_idx} Data")
        st.dataframe(df_view, width="stretch")

    st.subheader("Summary")
    count = len(df_view)
    mean_pass = df_view["pass_rate"].mean() if count else 0.0
    median_pass = df_view["pass_rate"].median() if count else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{count:,}")
    col2.metric("Pass rate mean", f"{mean_pass:.3f}")
    col3.metric("Pass rate median", f"{median_pass:.3f}")

    st.subheader(f"{metric} Distribution")
    fig = px.histogram(
        df_view,
        x=metric,
        color=group_by,
        nbins=30,
        marginal="box",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"Average {metric} by {group_by}")
    df_avg = (
        df_view
        .groupby(group_by, as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )
    fig = px.bar(
        df_avg,
        x=group_by,
        y=metric,
        text_auto=".2f",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Pass Rate Overview")
    fig = px.box(
        df_view,
        x=group_by,
        y="pass_rate",
        points="all",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top / Bottom Scenarios by Pass Rate")
    scenario_metric = df_view.groupby("Scenario", as_index=False)["pass_rate"].mean()
    top_n = st.number_input("Top N scenarios", min_value=5, max_value=100, value=20, key="single_top_n")
    sort_order = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True, key="single_scen_order")
    scenario_metric = scenario_metric.sort_values(
        "pass_rate",
        ascending=sort_order == "Lowest first",
    ).head(int(top_n))
    st.dataframe(scenario_metric, width="stretch")
