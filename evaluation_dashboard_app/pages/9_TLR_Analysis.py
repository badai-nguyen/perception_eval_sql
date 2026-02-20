"""
TLR (Traffic Light Recognition) Evaluation Analysis page.
Visualizes criteria matrices, vehicle status vs traffic light type, and critical/priority zones.
Supports Single (one dataset) or Compare (two datasets: Baseline A vs Compare B).
Supports shareable URLs via query params: mode, path_a, path_b.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from urllib.parse import quote

from lib.tlr_eval_analyzer import TLREvaluationAnalyzer
from lib.path_utils import get_data_root, path_display, list_tlr_result_directories

st.set_page_config(page_title="TLR Analysis", layout="wide")

# ====== URL QUERY PARAMS (for shareable links) ======
params = st.query_params
url_mode = params.get("mode")       # "single" / "compare" / None
url_path_a = params.get("path_a")  # relative path under data root
url_path_b = params.get("path_b")  # for compare mode

st.title("TLR (Traffic Light Recognition) Evaluation Analysis")

# ----- Helpers -----
data_root = get_data_root()


def path_to_tlr_key(path: Path) -> str:
    """Stable key for URL: path relative to data root, or '.' for root."""
    try:
        rel = path.resolve().relative_to(data_root.resolve())
        return str(rel).replace("\\", "/") if str(rel) != "." else "."
    except ValueError:
        return path.name or "."


def format_tlr_option(item):
    path, _count = item
    try:
        rel = path.relative_to(data_root)
        label = str(rel).replace("\\", "/") if str(rel) != "." else data_root.name or "."
    except ValueError:
        label = path.name or str(path)
    return label

def get_or_load_analyzer(resolved_path: str):
    """Load analyzer for path; cache in session_state by path."""
    if not resolved_path:
        return None
    cache_key = "tlr_analyzer_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    cache = st.session_state[cache_key]
    if resolved_path not in cache:
        with st.spinner(f"Loading TLR results: {Path(resolved_path).name}..."):
            analyzer = TLREvaluationAnalyzer(resolved_path)
            analyzer.load_all_results()
            if not analyzer.scenario_results:
                return None
            analyzer.extract_criteria_data()
            analyzer.pre_calculate_all_data()
            cache[resolved_path] = analyzer
    return cache[resolved_path]


def _render_single_tabs(analyzer, tab_criteria, tab_vehicle, tab_critical, tab_details):
    with tab_criteria:
        st.subheader("Criteria: TP rate and total frames")
        criteria_df = analyzer.create_criteria_matrix()
        st.dataframe(criteria_df, width='stretch', hide_index=True)
        criteria_df = criteria_df.copy()
        criteria_df["criteria_num"] = criteria_df["Criteria"].str.replace("criteria_", "").astype(int)
        fig1 = px.line(criteria_df, x="criteria_num", y="TP rate", title="TP rate by criteria", markers=True)
        fig1.update_layout(xaxis_title="Criteria number", yaxis_title="TP rate", yaxis_range=[0, 1.1])
        st.plotly_chart(fig1, width='stretch')
        fig2 = px.bar(criteria_df, x="criteria_num", y="Number of total frames", title="Total frames by criteria")
        fig2.update_layout(xaxis_title="Criteria number")
        st.plotly_chart(fig2, width='stretch')

    with tab_vehicle:
        st.subheader("Vehicle status vs traffic light type (TP rate)")
        status_df = analyzer.create_vehicle_status_matrix()
        tlr_cols = [c for c in status_df.columns if c != "Vehicle Status"]
        fig = go.Figure(
            data=go.Heatmap(
                z=status_df[tlr_cols].values,
                x=tlr_cols,
                y=status_df["Vehicle Status"].tolist(),
                colorscale="RdYlGn", zmin=0, zmax=1,
                text=[[f"{v:.3f}" for v in row] for row in status_df[tlr_cols].values],
                texttemplate="%{text}", textfont={"size": 9}, hoverongaps=False,
            )
        )
        fig.update_layout(title="TP rate: Vehicle status vs traffic light type", height=400, xaxis={"tickangle": -45})
        st.plotly_chart(fig, width='stretch')
        st.subheader("Raw counts (TP / Total)")
        st.dataframe(analyzer.create_vehicle_status_counts_matrix(), width='stretch', hide_index=True)

    with tab_critical:
        st.subheader("Critical (criteria 5–6) and priority (criteria 2–4) zones")
        cp_df = analyzer.create_vehicle_status_critical_priority_matrix()
        tlr_cols_cp = [c for c in cp_df.columns if c != "Vehicle Status"]
        fig_cp = go.Figure(
            data=go.Heatmap(
                z=cp_df[tlr_cols_cp].values, x=tlr_cols_cp, y=cp_df["Vehicle Status"].tolist(),
                colorscale="RdYlGn", zmin=0, zmax=1,
                text=[[f"{v:.3f}" for v in row] for row in cp_df[tlr_cols_cp].values],
                texttemplate="%{text}", textfont={"size": 8}, hoverongaps=False,
            )
        )
        fig_cp.update_layout(
            title="TP rate: Vehicle status vs traffic light type (critical & priority zones)",
            height=400, xaxis={"tickangle": -45},
        )
        st.plotly_chart(fig_cp, width='stretch')
        st.subheader("Raw counts (TP / Total)")
        st.dataframe(analyzer.create_vehicle_status_critical_priority_counts_matrix(), width='stretch', hide_index=True)

    with tab_details:
        st.subheader("Per-frame vehicle status and TLR details")
        details_df = analyzer.get_vehicle_status_details_df()
        if details_df is not None and not details_df.empty:
            st.caption("One row per frame. Use filters to narrow down by scenario, status, or traffic light type.")
            st.dataframe(details_df, width='stretch', hide_index=True)
        else:
            st.info("No vehicle status details available.")


def _render_compare_tabs(analyzer_a, analyzer_b, label_a, label_b, tab_criteria, tab_vehicle, tab_critical, tab_details):
    with tab_criteria:
        st.subheader("Criteria: A vs B (TP rate and delta)")
        df_a = analyzer_a.create_criteria_matrix()
        df_b = analyzer_b.create_criteria_matrix()
        compare_criteria = df_a[["Criteria"]].copy()
        compare_criteria["TP rate A"] = df_a["TP rate"].values
        compare_criteria["TP rate B"] = df_b["TP rate"].values
        compare_criteria["Δ (B − A)"] = compare_criteria["TP rate B"] - compare_criteria["TP rate A"]
        st.dataframe(compare_criteria, width='stretch', hide_index=True)
        compare_criteria["criteria_num"] = compare_criteria["Criteria"].str.replace("criteria_", "").astype(int)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=compare_criteria["criteria_num"], y=compare_criteria["TP rate A"], name=label_a, mode="lines+markers"))
        fig.add_trace(go.Scatter(x=compare_criteria["criteria_num"], y=compare_criteria["TP rate B"], name=label_b, mode="lines+markers"))
        fig.update_layout(title="TP rate by criteria: A vs B", xaxis_title="Criteria number", yaxis_title="TP rate", yaxis_range=[0, 1.1])
        st.plotly_chart(fig, width='stretch')
        delta_vals = compare_criteria["Δ (B − A)"].values
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in delta_vals]
        fig_delta = go.Figure(
            data=go.Bar(
                x=compare_criteria["criteria_num"],
                y=delta_vals,
                marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in delta_vals],
                textposition="outside",
            )
        )
        fig_delta.update_layout(
            title="TP rate delta (B − A) by criteria",
            xaxis_title="Criteria number",
            yaxis_title="Δ (B − A)",
            showlegend=False,
        )
        fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_delta, width='stretch')

    with tab_vehicle:
        st.subheader("Vehicle status vs TLR type: A vs B (TP rate delta)")
        status_a = analyzer_a.create_vehicle_status_matrix()
        status_b = analyzer_b.create_vehicle_status_matrix()
        tlr_cols = [c for c in status_a.columns if c != "Vehicle Status"]
        delta_df = status_a[["Vehicle Status"]].copy()
        for c in tlr_cols:
            delta_df[c] = status_b[c].values - status_a[c].values
        fig = go.Figure(
            data=go.Heatmap(
                z=delta_df[tlr_cols].values,
                x=tlr_cols,
                y=delta_df["Vehicle Status"].tolist(),
                colorscale=[[0, "#c0392b"], [0.25, "#e74c3c"], [0.5, "#f5f5f5"], [0.75, "#27ae60"], [1, "#1e8449"]],
                zmin=-1,
                zmax=1,
                zmid=0,
                text=[[f"{v:+.3f}" for v in row] for row in delta_df[tlr_cols].values],
                texttemplate="%{text}", textfont={"size": 8}, hoverongaps=False,
            )
        )
        fig.update_layout(
            title=f"TP rate delta (B − A): Vehicle status vs traffic light type",
            height=400, xaxis={"tickangle": -45},
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Green = B better, Red = A better.")
        with st.expander("Raw A"):
            st.dataframe(analyzer_a.create_vehicle_status_counts_matrix(), width='stretch', hide_index=True)
        with st.expander("Raw B"):
            st.dataframe(analyzer_b.create_vehicle_status_counts_matrix(), width='stretch', hide_index=True)

    with tab_critical:
        st.subheader("Critical & priority zones: A vs B (TP rate delta)")
        cp_a = analyzer_a.create_vehicle_status_critical_priority_matrix()
        cp_b = analyzer_b.create_vehicle_status_critical_priority_matrix()
        tlr_cols_cp = [c for c in cp_a.columns if c != "Vehicle Status"]
        delta_cp = cp_a[["Vehicle Status"]].copy()
        for c in tlr_cols_cp:
            delta_cp[c] = cp_b[c].values - cp_a[c].values
        fig_cp = go.Figure(
            data=go.Heatmap(
                z=delta_cp[tlr_cols_cp].values, x=tlr_cols_cp, y=delta_cp["Vehicle Status"].tolist(),
                colorscale=[[0, "#c0392b"], [0.25, "#e74c3c"], [0.5, "#f5f5f5"], [0.75, "#27ae60"], [1, "#1e8449"]],
                zmin=-1,
                zmax=1,
                zmid=0,
                text=[[f"{v:+.3f}" for v in row] for row in delta_cp[tlr_cols_cp].values],
                texttemplate="%{text}", textfont={"size": 7}, hoverongaps=False,
            )
        )
        fig_cp.update_layout(
            title="TP rate delta (B − A): Critical & priority zones",
            height=400, xaxis={"tickangle": -45},
        )
        st.plotly_chart(fig_cp, width='stretch')
        with st.expander("Raw A"):
            st.dataframe(analyzer_a.create_vehicle_status_critical_priority_counts_matrix(), width='stretch', hide_index=True)
        with st.expander("Raw B"):
            st.dataframe(analyzer_b.create_vehicle_status_critical_priority_counts_matrix(), width='stretch', hide_index=True)

    with tab_details:
        st.subheader("Vehicle status details")
        view_which = st.radio("Show details for", [label_a, label_b], horizontal=True, key="tlr_details_which")
        analyzer = analyzer_b if view_which == label_b else analyzer_a
        details_df = analyzer.get_vehicle_status_details_df()
        if details_df is not None and not details_df.empty:
            st.dataframe(details_df, width='stretch', hide_index=True)
        else:
            st.info("No vehicle status details available.")


# ----- Sidebar: mode and TLR directory selection -----
st.sidebar.header("TLR data")
st.sidebar.caption(f"Data root: `{path_display(data_root)}`")

tlr_candidates = list_tlr_result_directories()
# Build stable keys for URL (path relative to data root)
tlr_keys = [path_to_tlr_key(p) for p, _ in tlr_candidates] if tlr_candidates else []

# URL override for mode
saved_mode = "Single"
if url_mode == "compare":
    saved_mode = "Compare"
elif url_mode == "single":
    saved_mode = "Single"
mode_index = 0 if saved_mode == "Single" else 1
mode = st.sidebar.radio("Mode", ["Single", "Compare"], index=mode_index, horizontal=True, key="tlr_mode")

resolved_path_a = None
resolved_path_b = None
sel_a = 0
sel_b = 0

if tlr_candidates:
    options = list(range(len(tlr_candidates)))
    labels = [format_tlr_option(tlr_candidates[i]) for i in options]

    # Initial index for A from URL (if valid)
    run_a_index = tlr_keys.index(url_path_a) if url_path_a in tlr_keys else 0

    if mode == "Single":
        sel_a = st.sidebar.selectbox(
            "Choose TLR result directory (with result.json)",
            options=options,
            index=run_a_index,
            format_func=lambda i: labels[i],
            key="tlr_select_a",
        )
        resolved_path_a = str(tlr_candidates[sel_a][0])
        st.sidebar.success(f"Selected: {labels[sel_a]}")
    else:
        sel_a = st.sidebar.selectbox(
            "Baseline (A)",
            options=options,
            index=run_a_index,
            format_func=lambda i: labels[i],
            key="tlr_select_a",
        )
        resolved_path_a = str(tlr_candidates[sel_a][0])
        other_options = [i for i in options if i != sel_a]
        if not other_options:
            st.sidebar.warning("Add another TLR result directory under the data root to compare.")
        else:
            run_b_index_in_other = 0
            if url_path_b in tlr_keys and url_path_b != tlr_keys[sel_a]:
                try:
                    run_b_index_in_other = other_options.index(tlr_keys.index(url_path_b))
                except ValueError:
                    pass
            sel_b = st.sidebar.selectbox(
                "Compare (B)",
                options=other_options,
                index=min(run_b_index_in_other, len(other_options) - 1),
                format_func=lambda i: labels[i],
                key="tlr_select_b",
            )
            resolved_path_b = str(tlr_candidates[sel_b][0])
            st.sidebar.success(f"A: {labels[sel_a]}  →  B: {labels[sel_b]}")

    # Sync URL with current selection (shareable link)
    query = {"mode": "single" if mode == "Single" else "compare", "path_a": tlr_keys[sel_a]}
    if mode == "Compare" and resolved_path_b:
        query["path_b"] = tlr_keys[sel_b]
    st.query_params.update(query)
else:
    st.sidebar.warning(
        "No TLR result directories found. Under the data root we look for directories that contain "
        "**result.json** (direct subfolders or suite folders whose subfolders have result.json)."
    )

# ----- Load analyzer(s) -----
analyzer_a = get_or_load_analyzer(resolved_path_a) if resolved_path_a else None
analyzer_b = get_or_load_analyzer(resolved_path_b) if (mode == "Compare" and resolved_path_b) else None

if analyzer_a is None:
    st.info(
        "No TLR result directory selected. In the sidebar, choose a directory that contains **result.json**. "
        "Candidates are discovered automatically under the data root."
    )
    st.stop()

# ----- Labels for compare mode -----
label_a = Path(resolved_path_a).name if resolved_path_a else "A"
label_b = Path(resolved_path_b).name if resolved_path_b else "B"

# ========== SINGLE MODE ==========
if mode == "Single":
    stats = analyzer_a.get_summary_stats()
    st.subheader("Overview")
    share_q = f"mode=single&path_a={quote(tlr_keys[sel_a], safe='/')}"
    st.caption(f"Share this view: append `?{share_q}` to the TLR Analysis page URL.")
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

    tab_criteria, tab_vehicle, tab_critical, tab_details = st.tabs([
        "Criteria matrix", "Vehicle status vs TLR type", "Critical & priority zones", "Vehicle status details",
    ])
    _render_single_tabs(analyzer_a, tab_criteria, tab_vehicle, tab_critical, tab_details)
    st.stop()

# ========== COMPARE MODE ==========
if analyzer_b is None:
    st.info("Select a second TLR result directory (Compare B) in the sidebar to compare.")
    st.stop()

stats_a = analyzer_a.get_summary_stats()
stats_b = analyzer_b.get_summary_stats()

st.subheader("Overview: A vs B")
share_q_compare = f"mode=compare&path_a={quote(tlr_keys[sel_a], safe='/')}&path_b={quote(tlr_keys[sel_b], safe='/')}"
st.caption(f"Share this view: append `?{share_q_compare}` to the TLR Analysis page URL.")
col_a, col_delta, col_b = st.columns(3)
with col_a:
    st.markdown(f"**{label_a} (Baseline)**")
    st.metric("Scenarios", stats_a["num_scenarios"])
    st.metric("Total frames", f"{stats_a['total_frames']:,}")
    st.metric("Total TP", f"{stats_a['total_tp']:,}")
    st.metric("Overall TP rate", f"{stats_a['overall_tp_rate']:.2%}")
with col_delta:
    st.markdown("**Δ (B − A)**")
    st.metric("Scenarios", stats_b["num_scenarios"] - stats_a["num_scenarios"])
    st.metric("Total frames", f"{stats_b['total_frames'] - stats_a['total_frames']:+,}")
    st.metric("Total TP", f"{stats_b['total_tp'] - stats_a['total_tp']:+,}")
    delta_rate = stats_b["overall_tp_rate"] - stats_a["overall_tp_rate"]
    st.metric("Overall TP rate", f"{delta_rate:+.2%}")
with col_b:
    st.markdown(f"**{label_b} (Compare)**")
    st.metric("Scenarios", stats_b["num_scenarios"])
    st.metric("Total frames", f"{stats_b['total_frames']:,}")
    st.metric("Total TP", f"{stats_b['total_tp']:,}")
    st.metric("Overall TP rate", f"{stats_b['overall_tp_rate']:.2%}")

tab_criteria, tab_vehicle, tab_critical, tab_details = st.tabs([
    "Criteria matrix", "Vehicle status vs TLR type", "Critical & priority zones", "Vehicle status details",
])
_render_compare_tabs(analyzer_a, analyzer_b, label_a, label_b, tab_criteria, tab_vehicle, tab_critical, tab_details)
