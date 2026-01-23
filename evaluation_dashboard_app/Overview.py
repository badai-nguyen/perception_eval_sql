import streamlit as st
import pandas as pd
from pathlib import Path
from lib.run_loader import load_run
import plotly.express as px
import plotly.graph_objects as go

# ====== CONFIG AND CONSTANTS ======
st.set_page_config(page_title="Overview", layout="wide", initial_sidebar_state="expanded")
RUN_ROOT = Path("data")
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

# ====== HELPER FUNCTIONS ======
def build_summary_delta(df_a, df_b):
    df_a, df_b = df_a.set_index("id"), df_b.set_index("id")
    common_idx = df_a.index.intersection(df_b.index)
    result = pd.DataFrame(index=common_idx)
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms", "vx", "vy"]
    for m in metrics:
        result[m] = df_a.loc[common_idx, m]
        result[f"{m}_B"] = df_b.loc[common_idx, m]
        result[f"{m}_delta"] = df_b.loc[common_idx, m] - df_a.loc[common_idx, m]
    return result.reset_index()

def _safe_default(default_lst, options_lst):
    # Returns only those elements from default_lst that are also in options_lst
    # both expected to be list-like objects of hashables
    if not isinstance(default_lst, list):
        default_lst = list(default_lst) if default_lst is not None else []
    options_set = set(options_lst)
    safe = [x for x in default_lst if x in options_set]
    return safe

def create_filter_widgets(all_runs_data):
    # Collect and sort unique labels, taking care of possible missing or empty labels
    pl, prodl = set(), set()
    for run in all_runs_data:
        summary = run["summary"]
        # Guard against completely missing columns or all-na
        if "perception_label" in summary.columns:
            pl.update(
                [x for x in summary["perception_label"].dropna().unique() if str(x).strip() != ""]
            )
        if "product_label" in summary.columns:
            prodl.update(
                [x for x in summary["product_label"].dropna().unique() if str(x).strip() != ""]
            )
    perception_labels, product_labels = sorted(pl), sorted(prodl)
    label2ja = {label: PRODUCT_LABEL_JA.get(label, label) for label in product_labels}
    ja2label = {v: k for k, v in label2ja.items()}
    ja_cand = [label2ja.get(l, l) for l in product_labels]

    # Preserve UI state but only keep defaults that exist in actual options (avoid StreamlitAPIException)
    prev_perc = st.session_state.get("selected_perception_labels", perception_labels)
    prev_prod_ja = st.session_state.get("selected_product_ja_labels", ja_cand)

    safe_default_perc = _safe_default(prev_perc, perception_labels)
    safe_default_prod_ja = _safe_default(prev_prod_ja, ja_cand)

    st.session_state["selected_perception_labels"] = safe_default_perc
    st.session_state["selected_product_ja_labels"] = safe_default_prod_ja

    selected_perception_labels = st.sidebar.multiselect("Perception Label Filter", perception_labels,
        default=safe_default_perc, key="perception_filter_widget")
    selected_ja_labels = st.sidebar.multiselect("Product Label Filter", ja_cand,
        default=safe_default_prod_ja, key="product_filter_widget")

    st.session_state["selected_perception_labels"] = selected_perception_labels
    st.session_state["selected_product_ja_labels"] = selected_ja_labels
    selected_product_labels = [ja2label.get(ja, ja) for ja in selected_ja_labels] if selected_ja_labels else []

    return {
        "perception_labels": selected_perception_labels,
        "product_labels": selected_product_labels,
        "label_mappings": {"label2ja": label2ja, "ja2label": ja2label}
    }

def apply_filters(run_data, filters):
    s = run_data["summary"]
    if filters["perception_labels"] and "perception_label" in s.columns:
        # Filter only non-empty perception_label
        s = s[s["perception_label"].notna() & (s["perception_label"].astype(str).str.strip() != "")]
        s = s[s["perception_label"].isin(filters["perception_labels"])]
    if filters["product_labels"] and "product_label" in s.columns:
        # Filter only non-empty product_label
        s = s[s["product_label"].notna() & (s["product_label"].astype(str).str.strip() != "")]
        s = s[s["product_label"].isin(filters["product_labels"])]
    return {**run_data, "summary": s}

def display_metric_with_stats(metric, a, b):
    st.metric(f"{metric} mean", f"{b.mean():.4f}", delta=f"{b.mean()-a.mean():.4f}")
    st.caption(f"median {b.median():.4f} · P95 {b.quantile(0.95):.4f} · min {b.min():.4f} · max {b.max():.4f}")

def display_metric_with_stats_single(metric, s):
    st.metric(f"{metric} mean", f"{s.mean():.4f}")
    st.caption(f"median {s.median():.4f} · P95 {s.quantile(0.95):.4f} · min {s.min():.4f} · max {s.max():.4f}")

def show_grouped_metrics_plot(df, group_col, label_map=None, mode="single", df_b=None):
    st.markdown(f"#### Metrics by {group_col.replace('_', ' ').title()}")
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms"]
    # If column is not present or dataframe empty or all values are NaN/empty
    if (group_col not in df.columns or df.empty or
        df[group_col].dropna().astype(str).str.strip().eq("").all() or
        (mode == "compare" and df_b is not None and (df_b.empty or df_b[group_col].dropna().astype(str).str.strip().eq("").all()))
    ):
        st.info("No data for group breakdown."); return
    df, col_map = df.copy(), (label_map if label_map else {})
    # drop rows with NaN or empty
    df = df[df[group_col].notna() & (df[group_col].astype(str).str.strip() != "")]
    df["__label_jp"] = df[group_col].map(col_map) if col_map else df[group_col]
    show_mode = "compare" if (mode == "compare" and df_b is not None) else "single"
    colors = {"A": "#31356E", "B": "#008E9B", "Δ(B-A)": "#E86A33"}
    for m in metrics:
        st.markdown(f"##### {m.upper()} by {group_col.replace('_', ' ').title()}")
        if show_mode == "single":
            if df.empty:
                st.info("No data for group breakdown."); continue
            plot_df = df.groupby("__label_jp")[m].mean().reset_index().rename(columns={m:"Mean"})
            fig = px.bar(plot_df, x="__label_jp", y="Mean", labels={"__label_jp": group_col, "Mean": f"{m} mean"},
                         text_auto=".2f", color_discrete_sequence=["#31356E"])
            fig.update_layout(xaxis_title=None, yaxis_title=f"{m} Mean", showlegend=False, height=400, margin=dict(t=40, b=0))
            st.plotly_chart(fig, width="stretch")
        else:
            df_b_c = df_b.copy()
            if group_col not in df_b_c.columns:
                st.info("No data for group breakdown."); continue
            df_b_c = df_b_c[df_b_c[group_col].notna() & (df_b_c[group_col].astype(str).str.strip() != "")]
            df_b_c["__label_jp"] = df_b_c[group_col].map(col_map) if col_map else df_b_c[group_col]
            mean_a, mean_b = df.groupby("__label_jp")[m].mean(), df_b_c.groupby("__label_jp")[m].mean()
            plot_labels = sorted(set(mean_a.index).union(mean_b.index))
            plot_df = pd.DataFrame({"__label_jp": plot_labels})
            plot_df["A"] = plot_df["__label_jp"].map(mean_a).fillna(0)
            plot_df["B"] = plot_df["__label_jp"].map(mean_b).fillna(0)
            plot_df["Δ(B-A)"] = plot_df["B"] - plot_df["A"]
            melted = plot_df.melt(id_vars="__label_jp", value_vars=["A", "B", "Δ(B-A)"], var_name="Run", value_name="Mean")
            bar_order = ["A", "B", "Δ(B-A)"]
            fig = px.bar(melted, x="__label_jp", y="Mean", color="Run", text_auto=".2f",
                         category_orders={"Run": bar_order, "__label_jp": plot_labels},
                         barmode="group", color_discrete_map=colors,
                         labels={"__label_jp": group_col, "Mean": f"{m} mean"})
            fig.update_layout(xaxis_title=None, yaxis_title=f"{m} Mean", legend_title="Run",
                              height=400, margin=dict(t=40, b=0))
            st.plotly_chart(fig, width="stretch")

# ====== SIDEBAR UI ======
st.sidebar.header("Mode")
mode = st.sidebar.radio("Mode", ["Single Mode", "Compare Mode"]); st.session_state["mode"] = mode
st.title("Overview")
run_dirs = sorted([p for p in RUN_ROOT.iterdir() if p.is_dir()])
run_a_dir = st.sidebar.selectbox("Baseline (A)", run_dirs, format_func=lambda p: p.name)
run_b_dir = None
if mode == "Compare Mode":
    idx = 1 if len(run_dirs) > 1 else 0
    run_b_dir = st.sidebar.selectbox("Candidate (B)", run_dirs, index=idx, format_func=lambda p: p.name)

# ====== LOAD DATA ======
def safe_load_run(path, label='Run'):
    try:
        return load_run(path)
    except Exception as e:
        st.error(f"Failed to load {label}: {e}"); st.stop()

if mode == "Compare Mode":
    runA, runB = safe_load_run(run_a_dir, 'Run A'), safe_load_run(run_b_dir, 'Run B')
    filters = create_filter_widgets([runA, runB])
    runA, runB = apply_filters(runA, filters), apply_filters(runB, filters)
    st.session_state.update({"runA": runA, "runB": runB,
                            "df_cmp": build_summary_delta(runA["summary"], runB["summary"])})
else:
    runA = safe_load_run(run_a_dir, 'Run A')
    filters = create_filter_widgets([runA])
    runA = apply_filters(runA, filters)
    st.session_state["runA"] = runA

# ====== MAIN PAGE METRICS & CHARTS ======
st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{runA['path']}`")
if mode == "Compare Mode":
    st.markdown(f"**Candidate (B):** `{runB['path']}`")

st.subheader("Summary")
def show_tp_mean_by_label(df, label_col, label_jp_map=None, run_name=None):
    if label_col not in df.columns or df.empty:
        return
    # drop rows that are NA or blank
    xdf = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")]
    if xdf.empty:
        st.info(f"No data for {label_col.replace('_', ' ')} breakdown.")
        return
    group_tp = xdf.groupby(label_col)["TP"].mean()
    labels = group_tp.index.tolist()
    labels_disp = [label_jp_map.get(l, l) for l in labels] if label_jp_map else labels
    title = f"TP mean by {label_col.replace('_', ' ').title()}"
    title += f" — {run_name}" if run_name else ""
    st.markdown(f"**{title}**")
    fig = go.Figure(go.Bar(
        x=labels_disp, y=group_tp.values, text=[f"{x:.2f}" for x in group_tp.values],
        textposition="auto", marker=dict(color="#31356E"),
    ))
    fig.update_layout(xaxis_title=label_col.replace('_', ' ').title(),
                      yaxis_title="TP mean", height=400, margin=dict(t=40, b=0))
    st.plotly_chart(fig, width="stretch")

def show_tp_mean_by_label_compare(df_a, df_b, label_col, label_jp_map=None):
    if label_col not in df_a.columns or label_col not in df_b.columns:
        return
    # drop rows with missing or blank for both
    xdf_a = df_a[df_a[label_col].notna() & (df_a[label_col].astype(str).str.strip() != "")]
    xdf_b = df_b[df_b[label_col].notna() & (df_b[label_col].astype(str).str.strip() != "")]
    if xdf_a.empty and xdf_b.empty:
        st.info(f"No data for {label_col.replace('_', ' ')} breakdown.")
        return
    group_a = xdf_a.groupby(label_col)["TP"].mean()
    group_b = xdf_b.groupby(label_col)["TP"].mean()
    all_labels = sorted(set(group_a.index).union(group_b.index))
    a_vals = [group_a.get(l, float('nan')) for l in all_labels]
    b_vals = [group_b.get(l, float('nan')) for l in all_labels]
    deltas = [b-a if pd.notna(a) and pd.notna(b) else float('nan') for a, b in zip(a_vals, b_vals)]
    labels_disp = [label_jp_map.get(l, l) for l in all_labels] if label_jp_map else all_labels
    st.markdown(f"**TP mean by {label_col.replace('_', ' ').title()} (A vs B)**")
    fig = go.Figure([
        go.Bar(name="A", x=labels_disp, y=a_vals, marker=dict(color="#31356E"),
               text=[f"{x:.2f}" if pd.notna(x) else "N/A" for x in a_vals], textposition="auto"),
        go.Bar(name="B", x=labels_disp, y=b_vals, marker=dict(color="#008E9B"),
               text=[f"{x:.2f}" if pd.notna(x) else "N/A" for x in b_vals], textposition="auto"),
        go.Bar(name="Δ(B-A)", x=labels_disp, y=deltas, marker=dict(color="#E86A33"),
               text=[f"{x:+.2f}" if pd.notna(x) else "N/A" for x in deltas], textposition="auto"),
    ])
    fig.update_layout(barmode="group", xaxis_title=label_col.replace('_', ' ').title(),
                      yaxis_title="TP mean", height=400, margin=dict(t=40, b=0), legend_title="Run")
    st.plotly_chart(fig, width="stretch")

if mode == "Compare Mode":
    df_a, df_b = runA["summary"], runB["summary"]
    tp_mean_a, tp_mean_b = df_a["TP"].mean(), df_b["TP"].mean()
    st.metric("TP mean", f"{tp_mean_b:.2f}", delta=f"{tp_mean_b - tp_mean_a:+.2f}")
    cols = st.columns(4)
    metrics = [("XRMS", "xrms"), ("YRMS", "yrms"), ("XSTD", "xstd"), ("YSTD", "ystd")]
    for c, (n, col) in zip(cols, metrics):
        with c: display_metric_with_stats(n, df_a[col], df_b[col])
    show_tp_mean_by_label_compare(df_a, df_b, "perception_label")
    show_tp_mean_by_label_compare(df_a, df_b, "product_label", PRODUCT_LABEL_JA)
    with st.expander("Show metric breakdowns by label", expanded=False):
        show_grouped_metrics_plot(df_a, group_col="perception_label", mode="compare", df_b=df_b)
        show_grouped_metrics_plot(df_a, group_col="product_label", label_map=PRODUCT_LABEL_JA, mode="compare", df_b=df_b)
else:
    df_summary = runA["summary"]
    st.metric("TP mean", f"{df_summary['TP'].mean():.2f}")
    cols = st.columns(4)
    metrics = [("XRMS", "xrms"), ("YRMS", "yrms"), ("XSTD", "xstd"), ("YSTD", "ystd")]
    for c, (n, col) in zip(cols, metrics):
        with c: display_metric_with_stats_single(n, df_summary[col])
    show_tp_mean_by_label(df_summary, "perception_label")
    show_tp_mean_by_label(df_summary, "product_label", PRODUCT_LABEL_JA)
    with st.expander("Show metric breakdowns by label", expanded=False):
        show_grouped_metrics_plot(df_summary, group_col="perception_label", mode="single")
        show_grouped_metrics_plot(df_summary, group_col="product_label", label_map=PRODUCT_LABEL_JA, mode="single")
