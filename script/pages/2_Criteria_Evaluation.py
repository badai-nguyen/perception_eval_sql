import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Criteria-based Evaluation Viewer")
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
st.subheader("Raw Data Check")
df_raw = pd.read_csv(
    "data/Score.csv",
    header=None,
    engine="python",
    names=["Scenario", "Option", "GT_OBJ", "Distance0", "NM0", "TP/TN0", "ADD0", "AIL0", "UIL0", "PFN/PFP0", "UUID Num0", "Practical Pass Rate0", "MAX_DIST_THRESH0", "OBJ_CNTS0",
    "Distance1", "NM1", "TP/TN1", "ADD1", "AIL1", "UIL1", "PFN/PFP1", "UUID Num1", "Practical Pass Rate1", "MAX_DIST_THRESH1", "OBJ_CNTS1",
    "Distance2", "NM2", "TP/TN2", "ADD2", "AIL2", "UIL2", "PFN/PFP2", "UUID Num2", "Practical Pass Rate2", "MAX_DIST_THRESH2", "OBJ_CNTS2",
    "Distance3", "NM3", "TP/TN3", "ADD3", "AIL3", "UIL3", "PFN/PFP3", "UUID Num3", "Practical Pass Rate3", "MAX_DIST_THRESH3", "OBJ_CNTS3",
    ],
)
st.dataframe(df_raw.head(10), use_container_width=True)

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
st.dataframe(df_view, use_container_width=True)


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

st.plotly_chart(fig, use_container_width=True)

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

st.plotly_chart(fig, use_container_width=True)


st.subheader("Pass Rate Overview")

fig = px.box(
    df_view,
    x=group_by,
    y="pass_rate",
    points="all",
)

st.plotly_chart(fig, use_container_width=True)

