import streamlit as st
import pandas as pd
import plotly.express as px

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
df_raw_A = runA["score"]

runB = st.session_state.get("runB")
df_raw_B = runB["score"] if runB else None

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


# =========================
# View selector (compare)
# =========================
if mode == "Compare Mode":
    view = st.sidebar.selectbox(
        "View",
        ["Baseline (A)", "Candidate (B)"],
    )
    df_raw = df_raw_A if view == "Baseline (A)" else df_raw_B
else:
    df_raw = df_raw_A
    view = "Single Mode"

# =========================
# Raw data check
# =========================
st.subheader(f"Raw Data Check — {view}")
st.dataframe(df_raw.head(10), width="stretch")

criteria_idx = st.sidebar.selectbox(
    "Select Criteria",
    list(range(CRITERIA_COUNT)),
    format_func=lambda x: f"criteria{x}",
)

start = 3 + criteria_idx * BLOCK_SIZE
end = start + BLOCK_SIZE

df_view = df_raw.iloc[:, :3].copy()
df_view.columns = BASE_COLS

block = df_raw.iloc[:, start:end].copy()
block.columns = BLOCK_COLS

df_view = pd.concat([df_view, block], axis=1)
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

for c in NUM_COLS:
    df_view[c] = pd.to_numeric(df_view[c], errors="coerce")

st.subheader(f"Criteria {criteria_idx} Data")
st.dataframe(df_view, width="stretch")


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

# How is this metric distributed for this criteria
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

