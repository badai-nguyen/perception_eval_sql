"""
TLR (Traffic Light Recognition) Evaluation Analysis page.
Visualizes criteria matrices, vehicle status vs traffic light type, and critical/priority zones.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from lib.tlr_eval_analyzer import TLREvaluationAnalyzer
from lib.path_utils import get_data_root, path_display, list_tlr_result_directories

st.set_page_config(page_title="TLR Analysis", layout="wide")
st.title("TLR (Traffic Light Recognition) Evaluation Analysis")

# ----- TLR result directory: choose from candidates (dirs that have result.json) -----
data_root = get_data_root()
st.sidebar.header("TLR result directory")
st.sidebar.caption(f"Data root: `{path_display(data_root)}`")

tlr_candidates = list_tlr_result_directories()
resolved_path = None
if tlr_candidates:
    def format_tlr_option(item):
        path, count = item
        try:
            rel = path.relative_to(data_root)
            label = str(rel).replace("\\", "/") if str(rel) != "." else data_root.name or "."
        except ValueError:
            label = path.name or str(path)
        return f"{label} ({count} scenario(s))"

    selected = st.sidebar.selectbox(
        "Choose TLR result directory (with result.json)",
        options=range(len(tlr_candidates)),
        format_func=lambda i: format_tlr_option(tlr_candidates[i]),
        key="tlr_select",
    )
    resolved_path = str(tlr_candidates[selected][0])
    st.sidebar.success(f"Selected: {format_tlr_option(tlr_candidates[selected])}")
else:
    st.sidebar.warning(
        "No TLR result directories found. Under the data root we look for directories that contain "
        "**result.json** (direct subfolders or suite folders whose subfolders have result.json)."
    )

# ----- Load analyzer and cache in session -----
def ensure_analyzer():
    if resolved_path is None:
        return None
    key = "tlr_analyzer"
    path_key = "tlr_analyzer_path"
    if key not in st.session_state or st.session_state.get(path_key) != resolved_path:
        with st.spinner("Loading TLR results..."):
            analyzer = TLREvaluationAnalyzer(resolved_path)
            analyzer.load_all_results()
            if not analyzer.scenario_results:
                return None
            analyzer.extract_criteria_data()
            analyzer.pre_calculate_all_data()
            st.session_state[key] = analyzer
            st.session_state[path_key] = resolved_path
    return st.session_state[key]

analyzer = ensure_analyzer() if resolved_path else None

if analyzer is None:
    st.info(
        "No TLR result directory selected. In the sidebar, choose a directory that contains **result.json** "
        "(direct subfolders with `result.json` per scenario, or suite folders whose testcase subfolders have `result.json`). "
        "Candidates are discovered automatically under the data root."
    )
    st.stop()

# ----- Summary stats -----
stats = analyzer.get_summary_stats()
st.subheader("Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scenarios", stats["num_scenarios"])
c2.metric("Total frames", f"{stats['total_frames']:,}")
c3.metric("Total TP", f"{stats['total_tp']:,}")
c4.metric("Overall TP rate", f"{stats['overall_tp_rate']:.2%}")
c5.metric("Scenarios w/ criteria", stats["num_scenarios_with_criteria"])
if stats.get("best_criteria") is not None:
    st.caption(
        f"Best criteria: **{stats['best_criteria']}** (TP rate {stats['best_tp_rate']:.2%}) — "
        f"Worst: **{stats['worst_criteria']}** (TP rate {stats['worst_tp_rate']:.2%})"
    )

# ----- Tabs: Criteria | Vehicle status | Critical/Priority | Details -----
tab_criteria, tab_vehicle, tab_critical, tab_details = st.tabs([
    "Criteria matrix",
    "Vehicle status vs TLR type",
    "Critical & priority zones",
    "Vehicle status details",
])

with tab_criteria:
    st.subheader("Criteria: TP rate and total frames")
    criteria_df = analyzer.create_criteria_matrix()
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)
    # Plot: TP rate by criteria
    criteria_df = criteria_df.copy()
    criteria_df["criteria_num"] = criteria_df["Criteria"].str.replace("criteria_", "").astype(int)
    fig1 = px.line(
        criteria_df,
        x="criteria_num",
        y="TP rate",
        title="TP rate by criteria",
        markers=True,
    )
    fig1.update_layout(xaxis_title="Criteria number", yaxis_title="TP rate", yaxis_range=[0, 1.1])
    st.plotly_chart(fig1, use_container_width=True)
    # Plot: Total frames by criteria
    fig2 = px.bar(
        criteria_df,
        x="criteria_num",
        y="Number of total frames",
        title="Total frames by criteria",
    )
    fig2.update_layout(xaxis_title="Criteria number")
    st.plotly_chart(fig2, use_container_width=True)

with tab_vehicle:
    st.subheader("Vehicle status vs traffic light type (TP rate)")
    status_df = analyzer.create_vehicle_status_matrix()
    tlr_cols = [c for c in status_df.columns if c != "Vehicle Status"]
    # Heatmap with Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=status_df[tlr_cols].values,
            x=tlr_cols,
            y=status_df["Vehicle Status"].tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in status_df[tlr_cols].values],
            texttemplate="%{text}",
            textfont={"size": 9},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="TP rate: Vehicle status vs traffic light type",
        xaxis_title="",
        yaxis_title="",
        height=400,
        xaxis={"tickangle": -45},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Raw counts (TP / Total)")
    counts_df = analyzer.create_vehicle_status_counts_matrix()
    st.dataframe(counts_df, use_container_width=True, hide_index=True)

with tab_critical:
    st.subheader("Critical (criteria 5–6) and priority (criteria 2–4) zones")
    cp_df = analyzer.create_vehicle_status_critical_priority_matrix()
    tlr_cols_cp = [c for c in cp_df.columns if c != "Vehicle Status"]
    fig_cp = go.Figure(
        data=go.Heatmap(
            z=cp_df[tlr_cols_cp].values,
            x=tlr_cols_cp,
            y=cp_df["Vehicle Status"].tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in cp_df[tlr_cols_cp].values],
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False,
        )
    )
    fig_cp.update_layout(
        title="TP rate: Vehicle status vs traffic light type (critical & priority zones)",
        xaxis_title="",
        yaxis_title="",
        height=400,
        xaxis={"tickangle": -45},
    )
    st.plotly_chart(fig_cp, use_container_width=True)
    st.subheader("Raw counts (TP / Total)")
    cp_counts_df = analyzer.create_vehicle_status_critical_priority_counts_matrix()
    st.dataframe(cp_counts_df, use_container_width=True, hide_index=True)

with tab_details:
    st.subheader("Per-frame vehicle status and TLR details")
    details_df = analyzer.get_vehicle_status_details_df()
    if details_df is not None and not details_df.empty:
        st.caption("One row per frame. Use filters to narrow down by scenario, status, or traffic light type.")
        st.dataframe(details_df, use_container_width=True, hide_index=True)
    else:
        st.info("No vehicle status details available.")
