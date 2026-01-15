import streamlit as st
import pandas as pd
from pathlib import Path
from lib.run_loader import load_run

# =========================
# Config
# =========================
st.set_page_config(
    page_title="Evaluation Dashboard",
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



def display_metric_with_stats(metric_name, delta_series):
    """Display metric with comprehensive statistics."""
    st.metric(
        f"{metric_name} mean Δ",
        f"{delta_series.mean():+.4f}",
    )
    
    st.caption(
        f"median {delta_series.median():+.4f} · "
        f"P95 {delta_series.quantile(0.95):+.4f} · "
        f"worse {(delta_series > 0).mean() * 100:.1f}%"
    )


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

st.title("Evaluation Dashboard")

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
st.markdown(f"**Run A:** `{runA['path']}`")

if mode == "Compare Mode":
    st.markdown(f"**Run B:** `{runB['path']}`")
st.markdown("""
Use the sidebar to switch pages:

- **Tracking Stats**: position / velocity metrics  
- **Criteria Evaluation**: criteria-based performance metrics
""")


# -------------------------
# High-level metrics
# -------------------------
st.subheader("Summary")

tp_mean_a = runA["summary"]["TP"].mean()

if mode == "Compare Mode":
    tp_mean_b = runB["summary"]["TP"].mean()
    st.metric(
        "TP mean",
        f"{tp_mean_b:.2f}",
        delta=f"{tp_mean_b - tp_mean_a:+.2f}",
    )
else:
    st.metric("TP mean", f"{tp_mean_a:.2f}")

# -------------------------
# Score table (if exists)
# -------------------------
if runA["score"] is not None:
    st.subheader("Score")
    st.dataframe(runA["score"], width="stretch")

# -------------------------
# Comparison quick look
# -------------------------
if mode == "Compare Mode":
    st.subheader("Quick Comparison")

    df_cmp = st.session_state["df_cmp"]

    c1, c2 = st.columns(2)
    def reg_ratio(s):
        return (s > 0).mean() * 100.0


    with c1:
        st.metric(
            "XRMS mean Δ",
            f"{df_cmp['xrms_delta'].mean():+.4f}",
        )
        st.caption(
            f"median {df_cmp['xrms_delta'].median():+.4f} · "
            f"P95 {df_cmp['xrms_delta'].quantile(0.95):+.4f} · "
            f"worse {reg_ratio(df_cmp['xrms_delta']):.1f}%"
        )

    with c2:
        st.metric(
            "YRMS mean Δ",
            f"{df_cmp['yrms_delta'].mean():+.4f}",
        )
        st.caption(
            f"median {df_cmp['yrms_delta'].median():+.4f} · "
            f"P95 {df_cmp['yrms_delta'].quantile(0.95):+.4f} · "
            f"worse {reg_ratio(df_cmp['yrms_delta']):.1f}%"
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