import streamlit as st
import pandas as pd
from pathlib import Path
from lib.run_loader import load_run

# You may need this import for nicer chart appearance
import plotly.express as px

# =========================
# Config
# =========================
st.set_page_config(
    page_title="Overview",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Constants
# =========================
PRODUCT_LABEL_JA = {
    "Occlusion-Case": "遮蔽ケース",
    "False-Positive-Grass": "草誤検知（草停止）",
    "False-Positive-Ground": "地面誤検知",
    "False-Positive-Splash": "水しぶき 誤検知",
    "False-Positive-Exhaust-Fog": "排ガス・霧 誤検知",
    "Missed-Detection-Animal": "動物ロスト（犬）",
    "Missed-Detection-Falling-Object": "落下物未検知",
    "Missed-Detection-Pedestrian-Child": "歩行者未検知：子供",
    "Missed-Detection-Pedestrian-Umbrella": "歩行者未検知：傘",
    "Missed-Detection-Pedestrian-Crouching": "歩行者未検知：しゃがむ",
    "Missed-Detection-Pedestrian-Near-Structure": "歩行者未検知：構造物に近い",
    "False-Positive-Truck": "トラック誤検知",
    "Pose-Estimation-Yaw-Error": "Yawおかしい",
    "Long-Range-Detection-Failure": "遠方見えない",
    "Ghost-Object": "ミサイル",
    "Sudden-Fast-Vehicle-Ghost": "高速車両の突然出現・急ブレーキ誘発",
    "Misclassification-Structure-Grass-as-Pedestrian": "構造物・草を人に誤検知",
    "Misclassification-Structure-Grass-as-Vehicle": "構造物・草を車両に誤検知",
    "Misclassification-Bike-Motorcycle": "自転車・バイクのミスラベル",
    "Missed-Detection-Unridden-Bike": "人の乗ってないバイク自転車ロスト",
    "Missed-Detection-Traffic-Cone": "カラーコーンが認識できない",
    "Missed-Detection-Other": "その他ロスト",
}


# =========================
# Helper Functions
# =========================

def build_summary_delta(df_a, df_b):
    df_a = df_a.set_index("id")
    df_b = df_b.set_index("id")

    # Get common indices
    common_idx = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_idx]
    df_b = df_b.loc[common_idx]

    # Metrics to compare
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms", "vx", "vy"]

    # Build result using concat for better performance
    result = pd.DataFrame(index=common_idx)
    
    for metric in metrics:
        result[metric] = df_a[metric]
        result[f"{metric}_B"] = df_b[metric]
        result[f"{metric}_delta"] = df_b[metric] - df_a[metric]
    
    return result.reset_index()

def create_filter_widgets(all_runs_data):
    """Create common filter widgets for all runs, called only once"""
    
    # Collect all unique labels from all runs
    all_perception_labels = set()
    all_product_labels = set()
    
    for run_data in all_runs_data:
        summary = run_data["summary"]
        all_perception_labels.update(summary["perception_label"].dropna().unique())
        all_product_labels.update(summary["product_label"].dropna().unique())
    
    # Sort labels
    perception_labels = sorted(all_perception_labels)
    product_labels = sorted(all_product_labels)
    
    # Create product label mapping
    ja_label_lookup = {k: v for k, v in PRODUCT_LABEL_JA.items()}
    label2ja = {label: ja_label_lookup.get(label, label) for label in product_labels}
    ja2label = {v: k for k, v in label2ja.items()}
    ja_cand = [label2ja.get(label, label) for label in product_labels]
    
    # Initialize session state
    if "selected_perception_labels" not in st.session_state:
        st.session_state["selected_perception_labels"] = perception_labels
    
    if "selected_product_ja_labels" not in st.session_state:
        st.session_state["selected_product_ja_labels"] = ja_cand
    
    # Create widgets (called only once)
    selected_perception_labels = st.sidebar.multiselect(
        "Perception Label Filter",
        perception_labels,
        default=st.session_state["selected_perception_labels"],
        key="perception_filter_widget"
    )
    
    selected_ja_labels = st.sidebar.multiselect(
        "Product Label Filter",
        ja_cand,
        default=st.session_state["selected_product_ja_labels"],
        key="product_filter_widget"
    )
    
    # Update session state
    st.session_state["selected_perception_labels"] = selected_perception_labels
    st.session_state["selected_product_ja_labels"] = selected_ja_labels
    
    # Convert Japanese labels back to original labels
    selected_product_labels = [ja2label[ja] for ja in selected_ja_labels] if selected_ja_labels else []
    
    return {
        "perception_labels": selected_perception_labels,
        "product_labels": selected_product_labels,
        "label_mappings": {
            "label2ja": label2ja,
            "ja2label": ja2label
        }
    }
    
def apply_filters(run_data, filters):
    """Apply filters to a single run's data."""
    summary = run_data["summary"].copy()
    
    # Apply perception label filtering
    if filters["perception_labels"]:
        summary = summary[summary["perception_label"].isin(filters["perception_labels"])]
    
    # Apply product label filtering
    if filters["product_labels"]:
        summary = summary[summary["product_label"].isin(filters["product_labels"])]
    
    return {
        **run_data,
        "summary": summary
    }


def display_metric_with_stats(metric_name, series_a, series_b):
    """Display metric with comprehensive statistics for delta (A - B)."""
    st.metric(
        f"{metric_name} mean",
        f"{series_b.mean():.4f}",
        delta=f"{series_b.mean() - series_a.mean():.4f}",
    )
    
    st.caption(
        f"median {series_b.median():.4f} · "
        f"P95 {series_b.quantile(0.95):.4f} · "
        f"min {series_b.min():.4f} · max {series_b.max():.4f}"
    )

def display_metric_with_stats_single(metric_name, series):
    """Display metric with comprehensive statistics (non-delta)."""
    st.metric(
        f"{metric_name} mean",
        f"{series.mean():.4f}",
    )
    st.caption(
        f"median {series.median():.4f} · "
        f"P95 {series.quantile(0.95):.4f} · "
        f"min {series.min():.4f} · max {series.max():.4f}"
    )

def show_grouped_metrics_plot(df, group_col, label_map=None, mode="single", df_b=None):
    """
    Show grouped metrics as side-by-side bar plots for easy comprehension.
    Not hidden in expanders. Handles compare and single mode.
    """
    st.markdown(f"#### Metrics by {group_col.replace('_', ' ').title()}")
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms"]

    if group_col not in df.columns or df.empty or (mode == "compare" and df_b is not None and df_b.empty):
        st.info("No data for group breakdown.")
        return

    df = df.copy()
    group_vals = sorted(df[group_col].dropna().unique())

    if label_map:
        df["__label_jp"] = df[group_col].map(label_map)
    else:
        df["__label_jp"] = df[group_col]
    
    # Prepare coloring and legend order
    show_mode = "compare" if (mode == "compare" and df_b is not None) else "single"
    legend_title = "Run"
    color_discrete_map = {"A": "#31356E", "B": "#008E9B", "Δ(B-A)": "#E86A33"}

    for metric in metrics:
        st.markdown(f"##### {metric.upper()} by {group_col.replace('_', ' ').title()}")
        if show_mode == "single":
            # Single run
            plot_df = df.groupby("__label_jp")[metric].mean().reset_index()
            plot_df = plot_df.rename(columns={metric: "Mean"})
            fig = px.bar(
                plot_df,
                x="__label_jp",
                y="Mean",
                labels={"__label_jp": group_col, "Mean": f"{metric} mean"},
                text_auto=".2f",
                color_discrete_sequence=["#31356E"],  # dark blue
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis_title=f"{metric} Mean",
                showlegend=False,
                height=400,
                margin=dict(t=40, b=0),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            # Compare mode (side-by-side grouped bars)
            # Compute mean by label for both runs and (optional) delta
            df_b = df_b.copy()
            if label_map:
                df_b["__label_jp"] = df_b[group_col].map(label_map)
            else:
                df_b["__label_jp"] = df_b[group_col]
            mean_a = df.groupby("__label_jp")[metric].mean()
            mean_b = df_b.groupby("__label_jp")[metric].mean()
            plot_labels = sorted(set(mean_a.index).union(mean_b.index))
            plot_df = pd.DataFrame({"__label_jp": plot_labels})
            plot_df["A"] = plot_df["__label_jp"].map(mean_a).fillna(0)
            plot_df["B"] = plot_df["__label_jp"].map(mean_b).fillna(0)
            plot_df["Δ(B-A)"] = plot_df["B"] - plot_df["A"]

            # Melt for grouped bars
            melted = plot_df.melt(id_vars="__label_jp", value_vars=["A", "B", "Δ(B-A)"], var_name=legend_title, value_name="Mean")


            bar_order = ["A", "B", "Δ(B-A)"]

            fig = px.bar(
                melted,
                x="__label_jp",
                y="Mean",
                color=legend_title,
                text_auto=".2f",
                category_orders={legend_title: bar_order, "__label_jp": plot_labels},
                barmode="group",
                color_discrete_map=color_discrete_map,
                labels={"__label_jp": group_col, "Mean": f"{metric} mean"}
            )
            fig.update_layout(
                xaxis_title=None,
                yaxis_title=f"{metric} Mean",
                legend_title=legend_title,
                height=400,
                margin=dict(t=40, b=0)
            )
            st.plotly_chart(fig, width="stretch")


# =========================
# Sidebar — Global State
# =========================

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Mode",
    ["Single Mode", "Compare Mode"],
)
st.session_state["mode"] = mode

# =========================
# Config
# =========================
RUN_ROOT = Path("data")

st.title("Overview")

run_dirs = sorted([p for p in RUN_ROOT.iterdir() if p.is_dir()])
run_a_dir = st.sidebar.selectbox(
    "Baseline (A)",
    run_dirs,
    format_func=lambda p: p.name,
)

if mode == "Compare Mode":
    default_index = 1 if len(run_dirs) > 1 else 0
    run_b_dir = st.sidebar.selectbox(
        "Candidate (B)",
        run_dirs,
        index=default_index,
        format_func=lambda p: p.name,
    )

# =========================
# Load Data (ONCE)
# =========================

if mode == "Compare Mode":
    try:
        runA = load_run(run_a_dir)
        runB = load_run(run_b_dir)
    except Exception as e:
        st.error(f"Failed to load Run B: {e}")
        st.stop()

    filters = create_filter_widgets([runA, runB])
    runA = apply_filters(runA, filters)
    runB = apply_filters(runB, filters)
    st.session_state["runA"] = runA
    st.session_state["runB"] = runB
    st.session_state["df_cmp"] = build_summary_delta(
        runA["summary"],
        runB["summary"],
    )
else:
    try:
        runA = load_run(run_a_dir)
    except Exception as e:
        st.error(f"Failed to load Run A: {e}")
        st.stop()
    filters = create_filter_widgets([runA])
    runA = apply_filters(runA, filters)
    st.session_state["runA"] = runA


# =========================
# Main Page — Overview
# =========================
st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{runA['path']}`")

if mode == "Compare Mode":
    st.markdown(f"**Candidate (B):** `{runB['path']}`")


# -------------------------
# High-level metrics
# -------------------------
st.subheader("Summary")

import plotly.graph_objects as go

def show_tp_mean_by_label(df, label_col, label_jp_map=None, run_name=None):
    """Show TP mean by the given label as a bar chart in Streamlit, optionally with Japanese labels."""
    if label_col not in df.columns or df.empty:
        return
    group_tp = df.groupby(label_col)["TP"].mean()
    labels = group_tp.index.tolist()
    tps = group_tp.values.tolist()
    if label_jp_map:
        labels_disp = [label_jp_map.get(label, label) for label in labels]
    else:
        labels_disp = labels
    chart_title = f"TP mean by {label_col.replace('_', ' ').title()}"
    if run_name:
        chart_title += f" — {run_name}"
    st.markdown(f"**{chart_title}**")
    fig = go.Figure(go.Bar(
        x=labels_disp,
        y=tps,
        text=[f"{x:.2f}" for x in tps],
        textposition="auto",
        marker=dict(color="#31356E"),
    ))
    fig.update_layout(
        xaxis_title=label_col.replace('_', ' ').title(),
        yaxis_title="TP mean",
        height=400,
        margin=dict(t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def show_tp_mean_by_label_compare(df_a, df_b, label_col, label_jp_map=None):
    """Show TP mean by label, side by side for A and B, and their delta as a grouped bar chart."""
    if label_col not in df_a.columns or label_col not in df_b.columns:
        return
    group_a = df_a.groupby(label_col)["TP"].mean()
    group_b = df_b.groupby(label_col)["TP"].mean()
    all_labels = sorted(set(group_a.index).union(group_b.index))
    tp_a_vals = [group_a.get(label, float('nan')) for label in all_labels]
    tp_b_vals = [group_b.get(label, float('nan')) for label in all_labels]
    deltas = [b - a if pd.notna(a) and pd.notna(b) else float('nan') for a, b in zip(tp_a_vals, tp_b_vals)]
    if label_jp_map:
        labels_disp = [label_jp_map.get(label, label) for label in all_labels]
    else:
        labels_disp = all_labels

    st.markdown(f"**TP mean by {label_col.replace('_', ' ').title()} (A vs B)**")

    # Grouped bar for A, B, and Delta
    fig = go.Figure([
        go.Bar(
            name="A",
            x=labels_disp,
            y=tp_a_vals,
            marker=dict(color="#31356E"),
            text=[f"{x:.2f}" if pd.notna(x) else "N/A" for x in tp_a_vals],
            textposition="auto",
        ),
        go.Bar(
            name="B",
            x=labels_disp,
            y=tp_b_vals,
            marker=dict(color="#008E9B"),
            text=[f"{x:.2f}" if pd.notna(x) else "N/A" for x in tp_b_vals],
            textposition="auto",
        ),
        go.Bar(
            name="Δ(B-A)",
            x=labels_disp,
            y=deltas,
            marker=dict(color="#E86A33"),
            text=[f"{x:+.2f}" if pd.notna(x) else "N/A" for x in deltas],
            textposition="auto",
        ),
    ])
    fig.update_layout(
        barmode="group",
        xaxis_title=label_col.replace('_', ' ').title(),
        yaxis_title="TP mean",
        height=400,
        margin=dict(t=40, b=0),
        legend_title="Run",
    )
    st.plotly_chart(fig, use_container_width=True)

if mode == "Compare Mode":
    tp_mean_a = runA["summary"]["TP"].mean()
    tp_mean_b = runB["summary"]["TP"].mean()
    df_summary_a = runA["summary"]
    df_summary_b = runB["summary"]
    st.metric(
        "TP mean",
        f"{tp_mean_b:.2f}",
        delta=f"{tp_mean_b - tp_mean_a:+.2f}",
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_with_stats("XRMS", df_summary_a["xrms"], df_summary_b["xrms"])
    with col2:
        display_metric_with_stats("YRMS", df_summary_a["yrms"], df_summary_b["yrms"])
    with col3:
        display_metric_with_stats("XSTD", df_summary_a["xstd"], df_summary_b["xstd"])
    with col4:
        display_metric_with_stats("YSTD", df_summary_a["ystd"], df_summary_b["ystd"])

    show_tp_mean_by_label_compare(df_summary_a, df_summary_b, "perception_label")
    show_tp_mean_by_label_compare(df_summary_a, df_summary_b, "product_label", PRODUCT_LABEL_JA)

    # Show group-by summaries for perception_label and product_label visually
    # Japanese mapping for product labels if available
    prod_label_map = PRODUCT_LABEL_JA

    with st.expander("Show metric breakdowns by label (grouped bar charts)", expanded=False):
        show_grouped_metrics_plot(
            df_summary_a,
            group_col="perception_label",
            label_map=None,
            mode="compare",
            df_b=df_summary_b
        )
        show_grouped_metrics_plot(
            df_summary_a,
            group_col="product_label",
            label_map=prod_label_map,
            mode="compare",
            df_b=df_summary_b
        )
else:
    df_summary = runA["summary"]
    tp_mean_a = df_summary["TP"].mean()
    st.metric("TP mean", f"{tp_mean_a:.2f}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_with_stats_single("XRMS", df_summary["xrms"])
    with col2:
        display_metric_with_stats_single("YRMS", df_summary["yrms"])
    with col3:
        display_metric_with_stats_single("XSTD", df_summary["xstd"])
    with col4:
        display_metric_with_stats_single("YSTD", df_summary["ystd"])

    show_tp_mean_by_label(df_summary, "perception_label")
    show_tp_mean_by_label(df_summary, "product_label", PRODUCT_LABEL_JA)
    
    # Show TP mean by perception_label and product_label for this run
    with st.expander("Show TP mean by label (table format)", expanded=False):
        # Show group-by summaries for perception_label and product_label visually
        show_grouped_metrics_plot(
            df_summary,
            group_col="perception_label",
            label_map=None,
            mode="single"
        )
        show_grouped_metrics_plot(
            df_summary,
            group_col="product_label",
            label_map=PRODUCT_LABEL_JA,
            mode="single"
        )

# Gen2_Perception_DevOps_On_Vehicle_Shiojiri
# https://evaluation.tier4.jp/evaluation/suites/84c2a34e-387d-4218-927a-e06308e6fccc?project_id=x2_dev&tab=catalogs
# https://tier4.atlassian.net/wiki/spaces/CB/pages/4301390239/PDD+D+T+Devops
# Gen2_Perception_DevOps_On_Vehicle_Shiojiri_4
# Gen2_Perception_DevOps_On_Vehicle_Shiojiri_5
# Gen2_Perception_DevOps_On_Vehicle_Shiojiri_6
# Gen2_Perception_DevOps_On_Vehicle_Shiojiri_7
# Gen2_Perception_DevOps_On_Vehicle_Shiojiri_8
# Gen2_Perception_DevOps_On_Vehicle_Shiojiri_9
# Gen2_Perception_DevOps_On_Vehicle_Shiojiri_10