import streamlit as st
import pandas as pd
from pathlib import Path


# =========================
# Config
# =========================
st.set_page_config(
    page_title="Evaluation Dashboard",
    layout="wide",
)

SUMMARY_DTYPES = {
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
    "perception_label": "string",
    "product_label": "string"
}

def build_summary_delta(df_a, df_b):
    df_a = df_a.set_index("id")
    df_b = df_b.set_index("id")

    common = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common]
    df_b = df_b.loc[common]

    metrics = ["TP", "xstd", "ystd", "xrms", "yrms", "vx", "vy"]

    out = pd.DataFrame(index=common)
    for m in metrics:
        out[m] = df_a[m]
        out[f"{m}_B"] = df_b[m]
        out[f"{m}_delta"] = df_b[m] - df_a[m]

    return out.reset_index()

# =========================
# Helpers
# =========================


def load_run(run_dir: Path):
    summary_path = run_dir / "Summary.csv"
    score_path = run_dir / "Score.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found")

    summary = pd.read_csv(
        summary_path,
        header=None,
        names=SUMMARY_DTYPES.keys(),
        dtype=SUMMARY_DTYPES,
    )

    # Add filter for perception_label, default is all selected
    perception_labels = sorted(summary["perception_label"].dropna().unique())
    # Debug: Show available perception_labels
    if st.sidebar.checkbox("Show perception_labels debug info", value=False):
        st.sidebar.write("perception_labels:", perception_labels)
    if len(perception_labels) == 0:
        selected_labels = []
    else:
        selected_labels = st.sidebar.multiselect(
            "Perception Label Filter",
            perception_labels,
            default=perception_labels
        )
        if selected_labels:
            summary = summary[summary["perception_label"].isin(selected_labels)]


    # Product label mapping: English key to Japanese description
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

    product_labels = sorted(summary["product_label"].dropna().unique())
    # Map product label codes (possibly unseen) to Japanese, fall back to original label if unknown
    ja_label_lookup = {k: v for k, v in PRODUCT_LABEL_JA.items()}
    ja_cand = [ja_label_lookup.get(label, label) for label in product_labels]
    label2ja = {label: ja_label_lookup.get(label, label) for label in product_labels}
    ja2label = {v: k for k, v in label2ja.items()}

    if len(product_labels) == 0:
        selected_ja_labels = []
    else:
        selected_ja_labels = st.sidebar.multiselect(
            "Product Label Filter",
            ja_cand,
            default=ja_cand
        )
        if selected_ja_labels:
            selected_labels = [ja2label[ja] for ja in selected_ja_labels]
            summary = summary[summary["product_label"].isin(selected_labels)]


    score = pd.read_csv(score_path, header=None, engine="python", names=["Scenario", "Option", "GT_OBJ", "Distance0", "NM0", "TP/TN0", "ADD0", "AIL0", "UIL0", "PFN/PFP0", "UUID Num0", "Practical Pass Rate0", "MAX_DIST_THRESH0", "OBJ_CNTS0",
    "Distance1", "NM1", "TP/TN1", "ADD1", "AIL1", "UIL1", "PFN/PFP1", "UUID Num1", "Practical Pass Rate1", "MAX_DIST_THRESH1", "OBJ_CNTS1",
    "Distance2", "NM2", "TP/TN2", "ADD2", "AIL2", "UIL2", "PFN/PFP2", "UUID Num2", "Practical Pass Rate2", "MAX_DIST_THRESH2", "OBJ_CNTS2",
    "Distance3", "NM3", "TP/TN3", "ADD3", "AIL3", "UIL3", "PFN/PFP3", "UUID Num3", "Practical Pass Rate3", "MAX_DIST_THRESH3", "OBJ_CNTS3",
    ]) if score_path.exists() else None
    return {
        "path": run_dir,
        "summary": summary,
        "score": score,
    }

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
try:
    runA = load_run(run_a_dir)
except Exception as e:
    st.error(f"Failed to load Run A: {e}")
    st.stop()


st.session_state["runA"] = runA

if mode == "Compare Mode":
    try:
        runB = load_run(run_b_dir)
    except Exception as e:
        st.error(f"Failed to load Run B: {e}")
        st.stop()

    st.session_state["runB"] = runB
    st.session_state["df_cmp"] = build_summary_delta(
        runA["summary"],
        runB["summary"],
    )


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