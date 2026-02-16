import duckdb
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, List

from lib.path_utils import path_display

st.set_page_config(layout="wide")
st.title("Bounding Box Viewer")

# =============================
# Session state from Overview (run path)
# =============================
if "runA" not in st.session_state:
    st.warning("Please load data from the **Overview** page first (select mode and run(s)).")
    st.stop()

runA = st.session_state["runA"]


def list_parquets_in_run(run_path) -> List[str]:
    """Return sorted list of absolute paths to .parquet files in the run directory."""
    p = Path(run_path)
    if not p.is_dir():
        return []
    return sorted([str(f.resolve()) for f in p.glob("*.parquet")])


# Parquet files from the run designated on Overview
parquet_list_a = list_parquets_in_run(runA["path"])
if not parquet_list_a:
    st.error(
        f"No parquet files found in run directory: {path_display(runA['path'])}. "
        "Add a .parquet file or generate one from the Download page."
    )
    st.stop()

# ----------------------------
# Loaded Runs (from Overview)
# ----------------------------
st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{path_display(runA['path'])}`")

# ----------------------------
# Sidebar (Filters)
# ----------------------------
with st.sidebar:
    st.header("Filters")

    # Parquet file selection (from run directory)
    if len(parquet_list_a) == 1:
        selected_file = parquet_list_a[0]
    else:
        selected_file = st.selectbox(
            "Target File (Baseline A)",
            parquet_list_a,
            format_func=os.path.basename,
            key="bbox_viewer_target_file",
        )

# DuckDB connection (no cache = 安定優先)
con = duckdb.connect()

# --- Columns (for visibility existence check)
cols = con.execute("DESCRIBE SELECT * FROM parquet_scan(?)", [selected_file]).df()["column_name"].tolist()
has_visibility = "visibility" in cols
has_suite_name = "suite_name" in cols
has_scenario_name = "scenario_name" in cols

# --- Scene selection: one suite + one scenario (when columns exist)
scene_where = "1=1"
scene_params: List[str] = [selected_file]

if has_suite_name:
    suite_list = con.execute(
        "SELECT DISTINCT suite_name AS v FROM parquet_scan(?) WHERE suite_name IS NOT NULL ORDER BY v",
        [selected_file]
    ).df()["v"].dropna().astype(str).tolist()
else:
    suite_list = []

with st.sidebar:
    selected_suite = None
    selected_scenario = None
    if suite_list:
        selected_suite = st.selectbox(
            "Suite name",
            suite_list,
            key="bbox_viewer_suite",
        )
    if has_scenario_name:
        if selected_suite is not None:
            scenario_list = con.execute(
                "SELECT DISTINCT scenario_name AS v FROM parquet_scan(?) WHERE suite_name = ? AND scenario_name IS NOT NULL ORDER BY v",
                [selected_file, selected_suite]
            ).df()["v"].dropna().astype(str).tolist()
        else:
            scenario_list = con.execute(
                "SELECT DISTINCT scenario_name AS v FROM parquet_scan(?) WHERE scenario_name IS NOT NULL ORDER BY v",
                [selected_file]
            ).df()["v"].dropna().astype(str).tolist()
        if scenario_list:
            selected_scenario = st.selectbox(
                "Scenario name",
                scenario_list,
                key="bbox_viewer_scenario",
            )

# Build scene filter for queries (one scene = one suite + one scenario)
if selected_suite is not None:
    scene_where = "suite_name = ?"
    scene_params = [selected_file, selected_suite]
if selected_scenario is not None:
    scene_where = scene_where + " AND scenario_name = ?" if scene_where != "1=1" else "scenario_name = ?"
    scene_params = scene_params + [selected_scenario]
if scene_where == "1=1":
    scene_params = [selected_file]

# --- t4dataset_id (with suite/scenario filter)
t4_ids = con.execute(
    f"SELECT DISTINCT t4dataset_id AS v FROM parquet_scan(?) WHERE {scene_where} ORDER BY v",
    scene_params
).df()["v"].dropna().tolist()
if not t4_ids:
    st.error("No t4dataset_id in file for the selected suite/scenario filters.")
    st.stop()

with st.sidebar:
    selected_t4 = st.selectbox("t4dataset_id", t4_ids)

# --- topic_name（単一選択）
topic_names = con.execute(
    f"SELECT DISTINCT topic_name AS v FROM parquet_scan(?) WHERE {scene_where} AND t4dataset_id=? ORDER BY v",
    scene_params + [selected_t4]
).df()["v"].dropna().tolist()
if not topic_names:
    st.warning("No topic_name for selected t4dataset_id")
    st.stop()

with st.sidebar:
    selected_topic = st.selectbox("topic_name (single)", topic_names)

# --- label（複数選択）
labels = con.execute(
    f"SELECT DISTINCT label AS v FROM parquet_scan(?) WHERE {scene_where} AND t4dataset_id=? ORDER BY v",
    scene_params + [selected_t4]
).df()["v"].dropna().tolist()
if not labels:
    st.warning("No label for selected t4dataset_id")
    st.stop()

with st.sidebar:
    selected_labels = st.multiselect("label(s)", labels, default=labels)

# --- visibility（列があるときだけ。NULLは UNKNOWN で扱う）
selected_visibility = None
if has_visibility:
    vis_list = con.execute(
        f"SELECT DISTINCT COALESCE(visibility,'UNKNOWN') AS v FROM parquet_scan(?) WHERE {scene_where} AND t4dataset_id=? ORDER BY v",
        scene_params + [selected_t4]
    ).df()["v"].tolist()
    with st.sidebar:
        if vis_list:
            selected_visibility = st.multiselect("visibility", vis_list, default=vis_list)
        else:
            st.info("No visibility values found — skipping.")
else:
    with st.sidebar:
        st.info("No 'visibility' column found — skipping visibility filter.")

# Guard
if not selected_labels:
    st.warning("No label selected.")
    st.stop()

# --- invalidオブジェクト表示オプション ---
with st.sidebar:
    show_invalid = st.checkbox("Show invalid (zero-size) objects", value=False)


# ----------------------------
# Build query safely & load data
# ----------------------------
where = [scene_where, "t4dataset_id = ?", "topic_name = ?"]  # topic_name は単一選択
params = scene_params + [selected_t4, selected_topic]

# label IN (...)
where.append(f"label IN ({','.join(['?']*len(selected_labels))})")
params.extend(selected_labels)

# visibility（ある場合のみ、NULLは UNKNOWN で比較）
select_vis = ", visibility" if has_visibility else ""
if has_visibility and selected_visibility:
    where.append(f"COALESCE(visibility,'UNKNOWN') IN ({','.join(['?']*len(selected_visibility))})")
    params.extend(selected_visibility)

sql = f"""
SELECT frame_index, x, y, length, width, yaw, label, topic_name, source, status, uuid
{select_vis}
FROM parquet_scan(?)
WHERE {" AND ".join(where)}
ORDER BY frame_index
"""

df = con.execute(sql, params).df()
if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# frame_index を int に（比較安定化）
if "frame_index" in df.columns and not np.issubdtype(df["frame_index"].dtype, np.integer):
    df["frame_index"] = df["frame_index"].astype("Int64").fillna(0).astype(int)

# ----------------------------
# Color map
# ----------------------------
color_map = {
    ("GT", "TP"): "#00cc66",   # 緑
    ("GT", "FN"): "#ff9933",   # オレンジ
    ("EST", "TP"): "#66b3ff",  # 青
    ("EST", "FP"): "#ff6666",  # 赤
}
def get_color(source, status): return color_map.get((source, status), "#999999")

# ----------------------------
# Frame slider
# ----------------------------
frame = st.slider(
    "Frame index",
    int(df.frame_index.min()),
    int(df.frame_index.max()),
    step=1,
)
df_frame = df[df.frame_index == frame]

total_records = len(df_frame)
valid_records = int(((df_frame["length"] > 0) & (df_frame["width"] > 0)).sum())

# ----------------------------
# Geometry (yaw補正: x前方, y左方 → +π/2)
# ----------------------------
def rotated_rect(
    x: float, y: float,
    length: float, width: float,
    yaw: float,
    step_depth_ratio: float = 0.25,
    step_width_ratio: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    前方左側に段差（凹み）を入れて向きを表す矩形Polylineを返す。
    - yaw: ラジアン
    - step_depth_ratio: 凹みの「奥行き」（length比）
    - step_width_ratio: 凹みの「横幅」（width比）
    """
    if length < width:
        # something is wrong, fix size
        length, width = max(length, width), min(length, width)

    dx, dy = length / 2.0, width / 2.0
    step_depth = length * step_depth_ratio
    step_width = width * step_width_ratio

    # 頂点順序（時計回り）
    # 後ろ左 → 前左(手前側) → 凹み奥 → 前中央左 → 前右 → 後右 → 後ろ左
    corners = np.array([
        [-dx, -dy],                      # 後ろ左
        [ dx, -dy],                      # 前左端
        [ dx, 0],         # 段差上部
        [ dx - step_depth, 0],  # 凹み奥左
        [dx, 0],
        [dx,  dy],                      # 前右端
        [-dx,  dy],                      # 後右
        [-dx, -dy]                       # 戻る
    ])

    # 回転 (+π/2 でBEV向き調整)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = corners @ rot.T

    xs, ys = rotated[:, 0] + x, rotated[:, 1] + y
    return xs, ys


# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()
shown = set()

# 1. 条件に応じたマスクを作成
mask_both_invalid = (df_frame['length'] <= 0) & (df_frame['width'] <= 0)
mask_one_invalid = ((df_frame['length'] <= 0) | (df_frame['width'] <= 0)) & ~mask_both_invalid
mask_valid = (df_frame['length'] > 0) & (df_frame['width'] > 0)

# 共通のhover template
hovertemplate = (
    "X: %{x}<br>"
    "Y: %{y}<br>"
    "Label: %{customdata[0]}<br>"
    "size: %{customdata[1]:.2f} x %{customdata[2]:.2f}<br>"
)

# --- Case A: 両方のサイズが無効なオブジェクト (xマーカー) ---
if show_invalid and not df_frame[mask_both_invalid].empty:
    df_invalid = df_frame[mask_both_invalid]
    # legendに表示しないため、グループ化は不要
    fig.add_trace(go.Scatter(
        x=df_invalid['x'],
        y=df_invalid['y'],
        mode="markers",
        marker=dict(symbol="x", size=8, color=df_invalid.apply(lambda row: get_color(row.source, row.status), axis=1)),
        opacity=0.9,
        showlegend=False,
        hovertemplate=hovertemplate,
        customdata=df_invalid[['label', 'length', 'width']].values,
        name="invalid" # legendには出ないが、内部的な名前として設定
    ))

# --- Case B: 片方のサイズが無効なオブジェクト (円マーカー) ---
if not df_frame[mask_one_invalid].empty:
    df_cylinder = df_frame[mask_one_invalid]
    df_cylinder['name'] = df_cylinder['source'] + '/' + df_cylinder['status']
    
    # name ごとにグループ化してプロット (legend の重複を避けるため)
    for name, group in df_cylinder.groupby('name'):
        fig.add_trace(go.Scatter(
            x=group['x'],
            y=group['y'],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=group[['length', 'width']].max(axis=1),
                color=get_color(group.iloc[0].source, group.iloc[0].status)
            ),
            opacity=0.6,
            name=name,
            legendgroup=name,
            showlegend=name not in shown,
            hovertemplate=hovertemplate,
            customdata=group[['label', 'length', 'width']].values
        ))
        shown.add(name)

# --- Case C: 有効なオブジェクト (矩形ポリゴン) ---
if not df_frame[mask_valid].empty:
    df_valid = df_frame[mask_valid].copy() # SettingWithCopyWarning を避けるため copy()
    df_valid['name'] = df_valid['source'] + '/' + df_valid['status']
    
    # name ごとにグループ化してプロット
    for name, group in df_valid.groupby('name'):
        show = name not in shown
        for _, row in group.iterrows(): # ポリゴン生成は行ごとに行う必要がある
            x_poly, y_poly = rotated_rect(row.x, row.y, row.length, row.width, row.yaw)
            fig.add_trace(go.Scatter(
                x=x_poly, y=y_poly, mode="lines",
                fill="toself", opacity=0.6,
                line=dict(color=get_color(row.source, row.status)),
                name=name,
                legendgroup=name,
                showlegend=show,
                hovertemplate=hovertemplate,
                # customdataはトレース全体で1つしか設定できないため、代表点を設定
                customdata=[[row.label, row.length, row.width]] 
            ))
            show = False # 同じグループの2つ目以降はlegendを非表示
        shown.add(name)

# Ego marker（固定三角形）
fig.add_trace(go.Scatter(
    x=[0, -1.5, -1.5, 0], y=[0, -1, 1, 0],
    mode="lines", fill="toself",
    line=dict(color="black", width=2),
    fillcolor="gray", name="Ego Vehicle", showlegend=True
))

fig.update_layout(
    title=f"{os.path.basename(selected_file)} | t4dataset_id={selected_t4} | "
          f"topic={selected_topic} | Frame {frame} "
          f"| This frame: Total {total_records:,}, Valid {valid_records:,}",
    xaxis=dict(scaleanchor="y", scaleratio=1, title="X [m]"),
    yaxis=dict(scaleanchor="x", scaleratio=1, title="Y [m]"),
    legend=dict(groupclick="togglegroup", title="Source / Status"),
    height=900
)
st.plotly_chart(fig, width="stretch")

# === Frame別 TP/FN カウントと比率 ===
st.markdown("## 📈 Detection Stability over Frames")

# TP/FN集計
frame_stats = (
    df.query("source == 'GT' and status in ['TP', 'FN']")
      .groupby(["frame_index", "status"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)



# 比率 (TP率 = TP / (TP+FN))
# Ensure TP or FN column exists, else fill with 0
for col in ["TP", "FN"]:
    if col not in frame_stats.columns:
        frame_stats[col] = 0
frame_stats["TPR"] = np.where(
    (frame_stats["TP"] + frame_stats["FN"]) > 0,
    frame_stats["TP"] / (frame_stats["TP"] + frame_stats["FN"]),
    np.nan
)

# Plotlyで折れ線プロット
import plotly.express as px
# --- 時系列グラフ ---
fig_tpr = px.line(
    frame_stats,
    x="frame_index",
    y=["TP", "FN"],
    title="TP / FN Counts per Frame",
    labels={"value": "Count", "frame_index": "Frame Index", "variable": "Status"},
)
fig_tpr.update_layout(height=400, legend_title="Status")

# --- 現在Frameに縦破線を追加 ---
fig_tpr.add_vline(
    x=frame,
    line=dict(color="black", dash="dash", width=2),
    annotation_text=f"Frame {frame}",
    annotation_position="top left"
)

st.plotly_chart(fig_tpr, width="stretch")

# TPR比率の推移を別グラフで
fig_ratio = px.line(
    frame_stats,
    x="frame_index",
    y="TPR",
    title="True Positive Rate (TPR) per Frame",
    labels={"TPR": "True Positive Rate", "frame_index": "Frame Index"},
)
fig_ratio.update_yaxes(range=[0, 1])
fig_ratio.add_vline(
    x=frame,
    line=dict(color="black", dash="dash", width=2),
    annotation_text=f"Frame {frame}",
    annotation_position="top left"
)
st.plotly_chart(fig_ratio, width="stretch")

# === Worst-performing objects by FN rate ===
st.markdown("## 🚨 Objects with High FN Rate (GT-based)")

# uuid + label ごとにTP/FNをカウント
uuid_perf = (
    df.query("source == 'GT' and status in ['TP','FN']")
      .groupby(["uuid", "label"])["status"]
      .value_counts()
      .unstack(fill_value=0)
      .reset_index()
)

# Make sure 'TP' and 'FN' columns are present, else fill with 0
for col in ['TP', 'FN']:
    if col not in uuid_perf.columns:
        uuid_perf[col] = 0

uuid_perf["total"] = uuid_perf["TP"] + uuid_perf["FN"]
uuid_perf["FN_rate"] = uuid_perf["FN"] / uuid_perf["total"].replace(0, np.nan)

# total > 0 だけ残す
uuid_perf = uuid_perf[uuid_perf["total"] > 0]

# FN率でソート
uuid_perf_sorted = uuid_perf.sort_values("FN_rate", ascending=False)

# 表示
if not uuid_perf_sorted.empty:
    st.dataframe(
        uuid_perf_sorted[["uuid", "label", "TP", "FN", "total", "FN_rate"]]
            .head(30)
            .style.format({"FN_rate": "{:.2%}"})
    )
else:
    st.info("No GT objects with TP or FN were found.")

st.markdown("### 🔍 Inspect a Specific GT Object")

if not uuid_perf_sorted.empty:
    bad_uuid = st.selectbox("Select UUID to visualize", uuid_perf_sorted["uuid"].head(50))
else:
    bad_uuid = None

if bad_uuid is not None:
    uuid_traj = (
        df[(df["uuid"] == bad_uuid) & (df["source"] == "GT")]
        .sort_values("frame_index")
    )
else:
    uuid_traj = pd.DataFrame()

if not uuid_traj.empty:
    # --- marker symbol by status ---
    symbol_map = {"TP": "circle", "FN": "x", "FP": "triangle-up"}
    uuid_traj["marker_symbol"] = uuid_traj["status"].map(symbol_map).fillna("circle")

    fig_traj = go.Figure()

    # --- 全点を線で繋ぐ ---
    fig_traj.add_trace(go.Scatter(
        x=uuid_traj["x"], y=uuid_traj["y"],
        mode="lines",
        line=dict(color="gray", width=1),
        name=f"Trajectory ({uuid_traj['label'].iloc[0]})"
    ))

    # --- 各点（TP/FN別symbol） ---
    for status, group in uuid_traj.groupby("status"):
        fig_traj.add_trace(go.Scatter(
            x=group["x"], y=group["y"],
            mode="markers",
            marker=dict(
                symbol=symbol_map.get(status, "circle"),
                size=[10 if f == frame else 6 for f in group["frame_index"]],
                color=["red" if f == frame else ("orange" if status == "FN" else "green") for f in group["frame_index"]],
                line=dict(width=1, color="black")
            ),
            name=f"{status} points"
        ))

    fig_traj.update_layout(
        title=f"Trajectory of UUID {bad_uuid} ({uuid_traj['label'].iloc[0]})",
        xaxis=dict(title="X [m]", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y [m]", scaleanchor="x", scaleratio=1),
        height=600,
        legend=dict(title="Status")
    )
    st.plotly_chart(fig_traj, width="stretch")
else:
    st.info("No GT trajectory data for the selected UUID.")
