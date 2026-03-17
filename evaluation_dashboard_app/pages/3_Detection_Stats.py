import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Tuple

from lib.path_utils import path_display

st.set_page_config(layout="wide", page_title="Object Detection")
st.title("Object Detection Evaluation Dashboard")

# =============================
# Session state from Overview (mode, run paths)
# =============================
if "runA" not in st.session_state:
    st.warning("Please load data from the **Overview** page first (select mode and run(s)).")
    st.stop()

mode = st.session_state.get("mode", "Single Mode")
runA = st.session_state["runA"]
# Multi-run compare: use all_runs and run_labels when available (Overview sets these in Compare Mode)
all_runs = st.session_state.get("all_runs")
run_labels = st.session_state.get("run_labels")
if mode == "Compare Mode" and all_runs and run_labels and len(all_runs) >= 2:
    runs = all_runs
    run_labels_list = run_labels
else:
    runs = [runA]
    run_labels_list = ["A"]
    if mode == "Compare Mode":
        runB = st.session_state.get("runB")
        if runB is not None:
            runs = [runA, runB]
            run_labels_list = ["A", "B"]
single_mode = len(runs) == 1

def list_parquets_in_run(run_path) -> List[str]:
    """Return sorted list of absolute paths to .parquet files in the run directory."""
    p = Path(run_path)
    if not p.is_dir():
        return []
    return sorted([str(f.resolve()) for f in p.glob("*.parquet")])

# =============================
# DuckDB Connection
# =============================
_duckdb_connection: Optional[duckdb.DuckDBPyConnection] = None

def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Return a shared DuckDB connection for all queries."""
    global _duckdb_connection
    if _duckdb_connection is None:
        _duckdb_connection = duckdb.connect()
    return _duckdb_connection

# =============================
# Helper Functions
# =============================
def validate_parquet_file(con, path: str) -> Tuple[bool, str]:
    """
    Try to read the parquet file. Returns (True, "") if ok,
    (False, error_message) if the file cannot be read (e.g. empty or invalid schema).
    """
    try:
        con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [path])
        return True, ""
    except Exception as e:
        err = str(e).strip()
        if "non-root column" in err or "Need at least one" in err:
            return False, (
                "This Parquet file has no readable columns (DuckDB: 'Need at least one non-root column'). "
                "The file may be empty, corrupt, or written with a schema DuckDB cannot use. "
                "Try re-generating the parquet from the Download page or check the source data."
            )
        return False, err

def list_values(con, pq: str, expr: str, where: Optional[str] = None) -> List:
    """Get distinct values from parquet file."""
    q = f"SELECT DISTINCT {expr} FROM parquet_scan('{pq}')"
    if where:
        q += f" WHERE {where}"
    q += " ORDER BY 1"
    df_ = con.execute(q).df()
    if df_.empty:
        return []
    return df_.iloc[:, 0].dropna().tolist()

def create_view_eval_flat(con, target_file: str, view_name: str = "view_eval_flat"):
    """Create view_eval_flat with distance bins."""
    query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    WITH src AS (
        SELECT * FROM parquet_scan('{target_file}')
        UNION BY NAME
        SELECT CAST(NULL AS VARCHAR) AS visibility,
               CAST(NULL AS VARCHAR) AS suite_name,
               CAST(NULL AS VARCHAR) AS scenario_name,
               CAST(NULL AS VARCHAR) AS t4dataset_name
        WHERE FALSE
    ),
    base AS (
        SELECT
            * REPLACE (coalesce(CAST(visibility AS VARCHAR), 'not available') AS visibility),
            sqrt(CAST(x AS DOUBLE)*CAST(x AS DOUBLE) + CAST(y AS DOUBLE)*CAST(y AS DOUBLE)) AS dist_h
        FROM src
        WHERE x IS NOT NULL AND y IS NOT NULL
    ),
    bins AS (
        SELECT * FROM (
            VALUES
                (0.0,   10.0,   '[0,10)',     10),
                (10.0,  20.0,   '[10,20)',    20),
                (20.0,  30.0,   '[20,30)',    30),
                (30.0,  40.0,   '[30,40)',    40),
                (40.0,  50.0,   '[40,50)',    50),
                (50.0,  60.0,   '[50,60)',    60),
                (60.0,  70.0,   '[60,70)',    70),
                (70.0,  80.0,   '[70,80)',    80),
                (80.0,  90.0,   '[80,90)',    90),
                (90.0,  100.0,  '[90,100)',  100),
                (100.0, 110.0,  '[100,110)', 110),
                (110.0, 120.0,  '[110,120)', 120),
                (120.0, 130.0,  '[120,130)', 130),
                (130.0, 140.0,  '[130,140)', 140),
                (140.0, 150.0,  '[140,150)', 150),
                (150.0, 1e12,   '[150,inf)', 160)
        ) AS t(bin_start, bin_end, distance_bin, bin_idx)
    )
    SELECT
        bse.*,
        b.distance_bin,
        b.bin_idx,
        (status = 'TP') AS is_tp,
        (status = 'FP') AS is_fp,
        (status = 'FN') AS is_fn
    FROM base bse
    JOIN bins b
        ON bse.dist_h >= b.bin_start AND bse.dist_h < b.bin_end
    """
    con.execute(query)

def create_view_tpr_fpr(con, view_name: str = "view_tpr_fpr_by_class_dist_topic", source_eval_flat: str = "view_eval_flat"):
    """Create TPR/FPR view. source_eval_flat is the name of the eval_flat view to read from."""
    query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    WITH stats AS (
        SELECT
            t4dataset_id,
            topic_name,
            label,
            distance_bin,
            bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available') AS visibility,
            coalesce(try(CAST(suite_name AS VARCHAR)), '') AS suite_name,
            COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
            COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
            COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est
        FROM {source_eval_flat}
        GROUP BY
            t4dataset_id, topic_name, label, distance_bin, bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available'),
            coalesce(try(CAST(suite_name AS VARCHAR)), '')
    )
    SELECT
        *,
        CASE WHEN gt_total > 0 THEN CAST(tp_gt AS DOUBLE) / gt_total ELSE NULL END AS tpr,
        CASE WHEN est_total > 0 THEN CAST(fp_est AS DOUBLE) / est_total ELSE NULL END AS fpr
    FROM stats
    """
    con.execute(query)

def build_filter_clause(filters: dict,*, enable_dist_h: bool = True) -> str:
    """Build WHERE clause from filters."""
    conditions = []
    
    if filters.get('topic_name') and filters['topic_name'] != '__all__':
        conditions.append(f"topic_name = '{filters['topic_name']}'")
    
    if filters.get('label'):
        if isinstance(filters['label'], list) and len(filters['label']) > 0:
            # Escape single quotes in labels
            labels_escaped = [str(l).replace("'", "''") for l in filters['label']]
            labels_str = "', '".join(labels_escaped)
            conditions.append(f"label IN ('{labels_str}')")
        elif not isinstance(filters['label'], list) and filters['label'] != '__all__':
            label_escaped = str(filters['label']).replace("'", "''")
            conditions.append(f"label = '{label_escaped}'")
    
    if filters.get('suites'):
        if isinstance(filters['suites'], list) and len(filters['suites']) > 0:
            suite_escaped = [str(s).replace("'", "''") for s in filters['suites']]
            suite_str = "', '".join(suite_escaped)
            conditions.append(f"COALESCE(CAST(suite_name AS VARCHAR), '') IN ('{suite_str}')")
        elif not isinstance(filters['suites'], list) and filters['suites'] != '__all__':
            s_escaped = str(filters['suites']).replace("'", "''")
            conditions.append(f"COALESCE(CAST(suite_name AS VARCHAR), '') = '{s_escaped}'")
    
    if filters.get('visibility'):
        if isinstance(filters['visibility'], list) and len(filters['visibility']) > 0:
            # Escape single quotes in visibility values
            vis_escaped = [str(v).replace("'", "''") for v in filters['visibility']]
            vis_str = "', '".join(vis_escaped)
            conditions.append(f"COALESCE(visibility, 'not available') IN ('{vis_str}')")
        elif not isinstance(filters['visibility'], list):
            vis_escaped = str(filters['visibility']).replace("'", "''")
            conditions.append(f"COALESCE(visibility, 'not available') = '{vis_escaped}'")
    
    if enable_dist_h and filters.get('max_eval_range'):
        conditions.append(f"dist_h < {filters['max_eval_range']}")
    
    return " AND ".join(conditions) if conditions else "1=1"

# =============================
# Parquet files from run path(s)
# =============================
parquet_lists = [list_parquets_in_run(r["path"]) for r in runs]
for i, (r, pl) in enumerate(zip(runs, parquet_lists)):
    if not pl:
        label = run_labels_list[i] if i < len(run_labels_list) else str(i)
        st.error(f"No parquet files found in run ({label}): {path_display(r['path'])}. Add a .parquet file or generate one from the Download page.")
        st.stop()

# =============================
# Loaded Runs (from Overview)
# =============================
st.subheader("Loaded Runs")
for i, r in enumerate(runs):
    lbl = run_labels_list[i] if i < len(run_labels_list) else str(i)
    prefix = "Baseline (A):" if lbl == "A" else f"Candidate ({lbl}):"
    st.markdown(f"**{prefix}** `{path_display(r['path'])}`")

# =============================
# Sidebar - Filters
# =============================
# File selection per run
target_files = []
with st.sidebar:
    st.header("Filters / Inputs")
    for i, (pl, lbl) in enumerate(zip(parquet_lists, run_labels_list)):
        if len(pl) == 1:
            target_files.append(pl[0])
        else:
            tf = st.selectbox(
                f"Run ({lbl}) File",
                pl,
                format_func=lambda p: os.path.basename(p),
                index=min(i, len(pl) - 1),
                key=f"target_file_{lbl}"
            )
            target_files.append(tf)

con = get_duckdb_connection()
for i, (path, lbl) in enumerate(zip(target_files, run_labels_list)):
    ok, msg = validate_parquet_file(con, path)
    if not ok:
        st.sidebar.error(f"**Run ({lbl}) file** cannot be read: {msg}")
        st.stop()

# Create one eval_flat + tpr_fpr view per run (view_eval_flat_1, view_tpr_fpr_1, ...)
try:
    for i, path in enumerate(target_files):
        v_flat = "view_eval_flat" if i == 0 else f"view_eval_flat_{i}"
        v_tpr = "view_tpr_fpr_by_class_dist_topic" if i == 0 else f"view_tpr_fpr_{i}"
        create_view_eval_flat(con, path, v_flat)
        create_view_tpr_fpr(con, v_tpr, source_eval_flat=v_flat)
except Exception as e:
    st.error(f"Error creating views: {e}")
    st.stop()

# Filter options from first file (applied to all runs)
target_file = target_files[0]
with st.sidebar:
    topics = list_values(con, target_file, "topic_name")
    topic_name = st.selectbox("Topic Name", ["__all__"] + topics, key="topic_name") if topics else "__all__"
    labels = list_values(con, target_file, "label")
    selected_labels = st.multiselect("Label(s)", labels, default=labels[:5] if labels and len(labels) > 5 else (labels or []), key="labels")
    try:
        suite_options = list_values(con, target_file, "COALESCE(CAST(suite_name AS VARCHAR), '')")
    except Exception:
        suite_options = []
    selected_suites = st.multiselect("Suites", suite_options, default=suite_options, key="suites", help="Filter by suite(s). Default: all included.") if suite_options else []
    vis_options = list_values(con, target_file, "COALESCE(CAST(visibility AS VARCHAR), 'not available') AS visibility")
    selected_visibility = st.multiselect("Visibility", vis_options, default=vis_options, key="visibility") if vis_options else []
    max_eval_range = st.selectbox("Max Evaluation Range [m]", [50, 80, 100, 120, 150], index=0, key="max_eval_range")

# Build filters (same values for all runs)
filters_base = {
    'topic_name': topic_name,
    'label': selected_labels,
    'suites': selected_suites,
    'visibility': selected_visibility,
    'max_eval_range': max_eval_range
}
filters_list = [filters_base] * len(runs)

# =============================
# Main Content
# =============================

if st.checkbox("Debug: Inspect Parquet (All Runs)" if not single_mode else "Debug: Inspect Parquet"):
    cols_used = st.columns(len(target_files))
    file_labels = [(f"Run ({run_labels_list[i]}) File", target_files[i]) for i in range(len(target_files))]
    schema_results = []
    for col, (label, file_path) in zip(cols_used, file_labels):
        with col:
            st.markdown(f"### {label}")
            # Schema
            schema_df = con.execute("""
                DESCRIBE SELECT * FROM read_parquet(?)
            """, [file_path]).df()
            schema_results.append((label, schema_df))
            st.write("**Schema (Column Names, Types)**")
            st.markdown("Shows the schema (column names and their DuckDB/Parquet data types) of the selected Parquet file. Useful to check data structure and types as interpreted by DuckDB.")
            st.dataframe(schema_df)

            # Preview rows
            row_options = [10, 20, 50, 100, 200, "All"]
            preview_key = f"preview_row_limit_{label.replace(' ', '_').lower()}"
            row_choice = st.selectbox(f"Preview rows to show ({label})", row_options, index=1, key=preview_key)
            if row_choice == "All":
                limit_clause = ""
            else:
                limit_clause = f"LIMIT {row_choice}"
            preview_df = con.execute(f"""
                SELECT *
                FROM read_parquet(?)
                {limit_clause}
            """, [file_path]).df()
            st.write(f"**Preview (First {row_choice} rows)**")
            st.markdown(f"Shows the first {row_choice} preview rows from the Parquet file. Use this preview to examine example data contents and check that your file is as expected.")
            st.dataframe(preview_df)

            # Stats
            stats_df = con.execute("""
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(t4dataset_id) AS non_null_ids,
                    COUNT(DISTINCT t4dataset_id) AS distinct_ids
                FROM read_parquet(?)
            """, [file_path]).df()
            st.write("**Stats (Row Count, t4dataset_id non-null count, Distinct t4dataset_id count)**")
            st.markdown("""
            - `total_rows`: Total rows in the file  
            - `non_null_ids`: Rows where t4dataset_id is not null  
            - `distinct_ids`: Unique t4dataset_id values

            This helps rapidly assess the completeness and distribution of the key ID field.
            """)
            st.dataframe(stats_df)

    # --- Show info about schema differences (compare mode only) ---
    if not single_mode and len(schema_results) >= 2:
        with st.expander("⚖️ Difference between schemas", expanded=(len(schema_results) == 2)):
            if len(schema_results) == 2:
                label1, df1 = schema_results[0]
                label2, df2 = schema_results[1]
                names1 = set(df1["column_name"])
                names2 = set(df2["column_name"])
                added, removed = names2 - names1, names1 - names2
                common = names1 & names2
                types1 = {row["column_name"]: row["column_type"] for _, row in df1.iterrows()}
                types2 = {row["column_name"]: row["column_type"] for _, row in df2.iterrows()}
                dtype_changes = [(c, types1.get(c), types2.get(c)) for c in sorted(common) if types1.get(c) != types2.get(c)]
                if not (added or removed or dtype_changes):
                    st.success("✅ The schemas are identical (column names and types match exactly).")
                else:
                    if added:
                        st.error(f"Columns only in `{label2}`: {', '.join(sorted(added))}")
                    if removed:
                        st.error(f"Columns only in `{label1}`: {', '.join(sorted(removed))}")
                    if dtype_changes:
                        st.warning("Columns with different types:")
                        st.dataframe(pd.DataFrame(dtype_changes, columns=["Column", f"Type in {label1}", f"Type in {label2}"]), hide_index=True)
            else:
                st.info(f"{len(schema_results)} runs loaded. Compare schemas per run in the columns above.")



# Helper: view name for run index i (0-based)
def _flat_view(i: int) -> str:
    return "view_eval_flat" if i == 0 else f"view_eval_flat_{i}"

# =============================
# Panel 1: t4dataset Summary
# =============================
st.subheader("t4dataset summary and data parse")

try:
    if single_mode:
        query_base = f"""
        SELECT COUNT(DISTINCT t4dataset_id) AS id_num, '{os.path.basename(target_file)}' AS series
        FROM view_eval_flat
        """
        df_summary = con.execute(query_base).df()
        query_status = """
        SELECT label, status, COUNT(*) AS num
        FROM view_eval_flat
        GROUP BY label, status
        ORDER BY label, status
        """
        df_status = con.execute(query_status).df()
    else:
        parts = [f"SELECT COUNT(DISTINCT t4dataset_id) AS id_num, '{run_labels_list[i]}' AS series FROM {_flat_view(i)}" for i in range(len(runs))]
        query_base = " UNION ALL ".join(parts)
        df_summary = con.execute(query_base).df()
        parts_status = [f"SELECT '{run_labels_list[i]}' AS dataset, label, status, COUNT(*) AS num FROM {_flat_view(i)} GROUP BY label, status" for i in range(len(runs))]
        query_status = " UNION ALL ".join(parts_status) + " ORDER BY dataset, label, status"
        df_status = con.execute(query_status).df()

    if single_mode:
        st.write("**Status Count Table (label × status)**")
        if not df_status.empty:
            df_status_wide = df_status.pivot_table(index='label', columns='status', values='num', fill_value=0).reset_index()
            st.dataframe(df_status_wide)
            fig2 = px.bar(
                df_status,
                x="label",
                y="num",
                color="status",
                barmode="stack",
                title="Status Distribution per Label",
                labels={"num": "Count", "label": "Label", "status": "Status"}
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No status count data available")
    else:
        st.write("**Status Count Table (label × status by run)**")
        if not df_status.empty:
            df_status_wide = df_status.pivot_table(index='label', columns=['dataset', 'status'], values='num', fill_value=0)
            df_status_wide.columns = [f"{col[0]} {col[1]}" for col in df_status_wide.columns]
            df_status_wide = df_status_wide.reset_index()
            st.dataframe(df_status_wide)
            fig2 = px.bar(
                df_status,
                x="label",
                y="num",
                color="status",
                barmode="stack",
                facet_col="dataset",
                title="Status Distribution per Label (by Run)",
                category_orders={"dataset": run_labels_list},
                labels={"num": "Count", "label": "Label", "status": "Status"}
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No status count data available")

except Exception as e:
    st.error(f"Error in summary: {e}")

# =============================
# Panel 2: TP Rate (single) / TP Rate Comparison (compare)
# =============================
st.subheader("TP Rate" + (" Comparison" if not single_mode else ""))

_tpr_query = """
SELECT
    label,
    CASE
        WHEN COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) > 0
        THEN CAST(COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS DOUBLE)
             / COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN'))
        ELSE 0
    END AS tpr
FROM {view}
WHERE {filter_clause}
GROUP BY label
ORDER BY label
"""
if single_mode:
    try:
        filter_clause = build_filter_clause(filters_base)
        query = _tpr_query.format(view="view_eval_flat", filter_clause=filter_clause)
        df_tpr_base = con.execute(query).df()
        if not df_tpr_base.empty:
            fig = px.bar(
                df_tpr_base,
                x='label',
                y='tpr',
                title=f"Total TP rate within {max_eval_range} [m]",
                labels={'tpr': 'TP Rate', 'label': 'Label'}
            )
            fig.update_layout(yaxis_range=[0, 1.2])
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    try:
        dfs_tpr = []
        for i in range(len(runs)):
            fc = build_filter_clause(filters_list[i])
            q = _tpr_query.format(view=_flat_view(i), filter_clause=fc)
            df_i = con.execute(q).df()
            df_i["run"] = run_labels_list[i]
            dfs_tpr.append(df_i)
        df_tpr_all = pd.concat(dfs_tpr, ignore_index=True)
        if not df_tpr_all.empty:
            fig = px.bar(
                df_tpr_all,
                x="label",
                y="tpr",
                color="run",
                barmode="group",
                title=f"Total TP rate within {max_eval_range} [m] by run",
                labels={"tpr": "TP Rate", "label": "Label", "run": "Run"}
            )
            fig.update_layout(yaxis_range=[0, 1.2])
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
    except Exception as e:
        st.error(f"Error: {e}")

def _tpr_fpr_view(i: int) -> str:
    return "view_tpr_fpr_by_class_dist_topic" if i == 0 else f"view_tpr_fpr_{i}"

# =============================
# Panel 3: TP Rate by Distance Bin
# =============================
st.subheader("TP Rate by Distance Bin" + (" (Comparison)" if not single_mode else ""))

try:
    filter_clause_base = build_filter_clause(filters_base, enable_dist_h=False)
    if single_mode:
        query = f"""
        SELECT
            distance_bin,
            CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr
        FROM view_tpr_fpr_by_class_dist_topic
        WHERE {filter_clause_base}
        GROUP BY distance_bin
        ORDER BY CAST(REPLACE(SPLIT_PART(distance_bin, ',', 1), '[', ' ') AS INTEGER)
        """
        df_tpr_dist = con.execute(query).df()
        if not df_tpr_dist.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_tpr_dist['distance_bin'],
                y=df_tpr_dist['tpr'],
                name='TP Rate',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="TP Rate by Distance Bin",
                xaxis_title="Distance Bin",
                yaxis_title="TP Rate",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
    else:
        dfs_dist = []
        for i in range(len(runs)):
            fc = build_filter_clause(filters_list[i], enable_dist_h=False)
            q = f"""
            SELECT distance_bin,
                CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr
            FROM {_tpr_fpr_view(i)}
            WHERE {fc}
            GROUP BY distance_bin
            ORDER BY CAST(REPLACE(SPLIT_PART(distance_bin, ',', 1), '[', ' ') AS INTEGER)
            """
            df_i = con.execute(q).df()
            df_i["run"] = run_labels_list[i]
            dfs_dist.append(df_i)
        df_tpr_dist = pd.concat(dfs_dist, ignore_index=True)
        if not df_tpr_dist.empty:
            fig = go.Figure()
            colors = ["lightblue", "lightcoral", "lightgreen", "#9B59B6", "#1ABC9C", "#E86A33"]
            for i, lbl in enumerate(run_labels_list):
                d = df_tpr_dist[df_tpr_dist["run"] == lbl]
                fig.add_trace(go.Bar(
                    x=d["distance_bin"],
                    y=d["tpr"],
                    name=f"TP Rate ({lbl})",
                    marker_color=colors[i % len(colors)]
                ))
            fig.update_layout(
                title="TP Rate by Distance Bin (all runs)",
                xaxis_title="Distance Bin",
                yaxis_title="TP Rate",
                barmode="group",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
except Exception as e:
    st.error(f"Error: {e}")

# =============================
# Panel 4: FP Rate by Distance Bin
# =============================
st.subheader("FP Rate by Distance Bin")

try:
    if single_mode:
        query = f"""
        SELECT
            distance_bin,
            CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
        FROM view_tpr_fpr_by_class_dist_topic
        WHERE {filter_clause_base}
        GROUP BY distance_bin
        ORDER BY CAST(REPLACE(SPLIT_PART(distance_bin, ',', 1), '[', ' ') AS INTEGER)
        """
        df_fpr_dist = con.execute(query).df()
        if not df_fpr_dist.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_fpr_dist['distance_bin'],
                y=df_fpr_dist['fpr'],
                name='FP Rate',
                marker_color='lightcoral'
            ))
            fig.update_layout(
                title="FP Rate by Distance Bin",
                xaxis_title="Distance Bin",
                yaxis_title="FP Rate",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
    else:
        dfs_fpr = []
        for i in range(len(runs)):
            fc = build_filter_clause(filters_list[i], enable_dist_h=False)
            q = f"""
            SELECT distance_bin,
                CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
            FROM {_tpr_fpr_view(i)}
            WHERE {fc}
            GROUP BY distance_bin
            ORDER BY CAST(REPLACE(SPLIT_PART(distance_bin, ',', 1), '[', ' ') AS INTEGER)
            """
            df_i = con.execute(q).df()
            df_i["run"] = run_labels_list[i]
            dfs_fpr.append(df_i)
        df_fpr_dist = pd.concat(dfs_fpr, ignore_index=True)
        if not df_fpr_dist.empty:
            fig = go.Figure()
            colors = ["lightblue", "lightcoral", "lightgreen", "#9B59B6", "#1ABC9C", "#E86A33"]
            for i, lbl in enumerate(run_labels_list):
                d = df_fpr_dist[df_fpr_dist["run"] == lbl]
                fig.add_trace(go.Bar(
                    x=d["distance_bin"],
                    y=d["fpr"],
                    name=f"FP Rate ({lbl})",
                    marker_color=colors[i % len(colors)]
                ))
            fig.update_layout(
                title="FP Rate by Distance Bin (all runs)",
                xaxis_title="Distance Bin",
                yaxis_title="FP Rate",
                barmode="group",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
except Exception as e:
    st.error(f"Error: {e}")

# =============================
# Panel 5: Improved and Degraded Count (compare mode only, each candidate vs A)
# =============================
if not single_mode:
    st.subheader("Improved and Degraded count by objects (each run vs Baseline A)")
    for idx in range(1, len(runs)):
        lbl = run_labels_list[idx]
        try:
            filter_clause_comp_p5 = build_filter_clause(filters_list[idx], enable_dist_h=False)
            comp_flat = _flat_view(idx)
            query = f"""
            WITH base_gt AS (
                SELECT
                    t4dataset_id,
                    frame_index,
                    uuid AS gt_uuid,
                    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
                    COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                    COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                    COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                FROM view_eval_flat
                WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                    AND {filter_clause_base}
                GROUP BY 1,2,3
            ),
            comp_gt AS (
                SELECT
                    t4dataset_id,
                    frame_index,
                    uuid AS gt_uuid,
                    COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
                    COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                    COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                    COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                FROM {comp_flat}
                WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                    AND {filter_clause_comp_p5}
                GROUP BY 1,2,3
            ),
            joined AS (
                SELECT
                    COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                    COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
                    COALESCE(b.gt_uuid, c.gt_uuid) AS gt_uuid,
                    COALESCE(b.tp_base, FALSE) AS tp_base,
                    COALESCE(c.tp_comp, FALSE) AS tp_comp,
                    COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                    COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                    COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
                FROM base_gt b
                FULL OUTER JOIN comp_gt c
                    ON b.t4dataset_id = c.t4dataset_id
                   AND b.frame_index = c.frame_index
                   AND b.gt_uuid = c.gt_uuid
            )
            SELECT
                t4dataset_id,
                CAST(COUNT(*) FILTER (WHERE TRUE) AS DOUBLE) AS total_gt,
                CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
                CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt,
                CAST(COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS DOUBLE) AS both_tp_cnt,
                CAST(COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS DOUBLE) AS both_fn_cnt,
                CAST(SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS DOUBLE) AS net_tp_delta,
                suite_name,
                scenario_name,
                t4dataset_name
            FROM joined
            GROUP BY t4dataset_id, suite_name, scenario_name, t4dataset_name
            ORDER BY net_tp_delta DESC
            """
            df_improved = con.execute(query).df()
            if not df_improved.empty:
                with st.expander(f"Run {lbl} vs A", expanded=(len(runs) == 2)):
                    st.dataframe(df_improved, width="stretch")

                    # By label: improved/degraded counts per object class
                    st.markdown("**By label**")
                    query_label = f"""
                    WITH base_gt AS (
                        SELECT
                            t4dataset_id,
                            frame_index,
                            uuid AS gt_uuid,
                            COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
                            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
                        FROM view_eval_flat
                        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                            AND {filter_clause_base}
                        GROUP BY 1, 2, 3
                    ),
                    comp_gt AS (
                        SELECT
                            t4dataset_id,
                            frame_index,
                            uuid AS gt_uuid,
                            COALESCE(MAX(try_cast(label AS VARCHAR)), '') AS label,
                            COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
                        FROM {comp_flat}
                        WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                            AND {filter_clause_comp_p5}
                        GROUP BY 1, 2, 3
                    ),
                    joined AS (
                        SELECT
                            COALESCE(b.label, c.label) AS label,
                            COALESCE(b.tp_base, FALSE) AS tp_base,
                            COALESCE(c.tp_comp, FALSE) AS tp_comp
                        FROM base_gt b
                        FULL OUTER JOIN comp_gt c
                            ON b.t4dataset_id = c.t4dataset_id
                           AND b.frame_index = c.frame_index
                           AND b.gt_uuid = c.gt_uuid
                    )
                    SELECT
                        label,
                        CAST(COUNT(*) FILTER (WHERE TRUE) AS DOUBLE) AS total_gt,
                        CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
                        CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt,
                        CAST(COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS DOUBLE) AS both_tp_cnt,
                        CAST(COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS DOUBLE) AS both_fn_cnt,
                        CAST(SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS DOUBLE) AS net_tp_delta
                    FROM joined
                    GROUP BY label
                    ORDER BY net_tp_delta DESC
                    """
                    try:
                        df_by_label = con.execute(query_label).df()
                        if not df_by_label.empty:
                            st.dataframe(df_by_label, hide_index=True)
                            fig_label = px.bar(
                                df_by_label,
                                x="label",
                                y="net_tp_delta",
                                title=f"Net TP delta by label ({lbl} vs A)",
                                labels={"label": "Label", "net_tp_delta": "Net TP delta"},
                                color="net_tp_delta",
                                color_continuous_scale="RdYlGn",
                                color_continuous_midpoint=0,
                            )
                            st.plotly_chart(fig_label, use_container_width=True)
                        else:
                            st.caption("No label breakdown.")
                    except Exception as e_label:
                        st.caption(f"By label: {e_label}")

                    # By frame: improved/degraded per frame
                    with st.expander("By frame"):
                        query_frame = f"""
                        WITH base_gt AS (
                            SELECT
                                t4dataset_id,
                                frame_index,
                                uuid AS gt_uuid,
                                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
                                COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                                COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                                COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                            FROM view_eval_flat
                            WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                AND {filter_clause_base}
                            GROUP BY 1, 2, 3
                        ),
                        comp_gt AS (
                            SELECT
                                t4dataset_id,
                                frame_index,
                                uuid AS gt_uuid,
                                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
                                COALESCE(MAX(try_cast(suite_name AS VARCHAR)), '') AS suite_name,
                                COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name,
                                COALESCE(MAX(try_cast(t4dataset_name AS VARCHAR)), '') AS t4dataset_name
                            FROM {comp_flat}
                            WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                AND {filter_clause_comp_p5}
                            GROUP BY 1, 2, 3
                        ),
                        joined AS (
                            SELECT
                                COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                                COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
                                COALESCE(b.gt_uuid, c.gt_uuid) AS gt_uuid,
                                COALESCE(b.tp_base, FALSE) AS tp_base,
                                COALESCE(c.tp_comp, FALSE) AS tp_comp,
                                COALESCE(b.suite_name, c.suite_name, '') AS suite_name,
                                COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name,
                                COALESCE(b.t4dataset_name, c.t4dataset_name, '') AS t4dataset_name
                            FROM base_gt b
                            FULL OUTER JOIN comp_gt c
                                ON b.t4dataset_id = c.t4dataset_id
                               AND b.frame_index = c.frame_index
                               AND b.gt_uuid = c.gt_uuid
                        )
                        SELECT
                            t4dataset_id,
                            frame_index,
                            scenario_name,
                            suite_name,
                            t4dataset_name,
                            CAST(COUNT(*) FILTER (WHERE TRUE) AS DOUBLE) AS total_gt,
                            CAST(COUNT(*) FILTER (WHERE NOT tp_base AND tp_comp) AS DOUBLE) AS improved_cnt,
                            CAST(COUNT(*) FILTER (WHERE tp_base AND NOT tp_comp) AS DOUBLE) AS degraded_cnt,
                            CAST(COUNT(*) FILTER (WHERE tp_base AND tp_comp) AS DOUBLE) AS both_tp_cnt,
                            CAST(COUNT(*) FILTER (WHERE NOT tp_base AND NOT tp_comp) AS DOUBLE) AS both_fn_cnt,
                            CAST(SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS DOUBLE) AS net_tp_delta
                        FROM joined
                        GROUP BY t4dataset_id, frame_index, suite_name, scenario_name, t4dataset_name
                        ORDER BY net_tp_delta DESC
                        """
                        try:
                            df_by_frame = con.execute(query_frame).df()
                            if not df_by_frame.empty:
                                st.dataframe(df_by_frame, hide_index=True)
                            else:
                                st.caption("No frame breakdown.")
                        except Exception as e_frame:
                            st.caption(f"By frame: {e_frame}")

                    # By object: per-object list with change_type, label, dist_h
                    with st.expander("By object"):
                        change_type_filter = st.selectbox(
                            "Change type",
                            ["all", "improved", "degraded", "both_tp", "both_fn"],
                            key=f"change_type_{lbl}_{idx}",
                            help="Filter objects by TP change between runs.",
                        )
                        query_object = f"""
                        WITH base_gt AS (
                            SELECT
                                t4dataset_id,
                                frame_index,
                                uuid AS gt_uuid,
                                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base,
                                COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name
                            FROM view_eval_flat
                            WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                AND {filter_clause_base}
                            GROUP BY 1, 2, 3
                        ),
                        comp_gt AS (
                            SELECT
                                t4dataset_id,
                                frame_index,
                                uuid AS gt_uuid,
                                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp,
                                COALESCE(MAX(try_cast(scenario_name AS VARCHAR)), '') AS scenario_name
                            FROM {comp_flat}
                            WHERE source = 'GT' AND uuid IS NOT NULL AND frame_index IS NOT NULL
                                AND {filter_clause_comp_p5}
                            GROUP BY 1, 2, 3
                        ),
                        joined AS (
                            SELECT
                                COALESCE(CAST(b.t4dataset_id AS VARCHAR), CAST(c.t4dataset_id AS VARCHAR)) AS t4dataset_id,
                                COALESCE(CAST(b.frame_index AS VARCHAR), CAST(c.frame_index AS VARCHAR)) AS frame_index,
                                COALESCE(b.gt_uuid, c.gt_uuid) AS gt_uuid,
                                COALESCE(b.tp_base, FALSE) AS tp_base,
                                COALESCE(c.tp_comp, FALSE) AS tp_comp,
                                COALESCE(b.scenario_name, c.scenario_name, '') AS scenario_name
                            FROM base_gt b
                            FULL OUTER JOIN comp_gt c
                                ON b.t4dataset_id = c.t4dataset_id
                               AND b.frame_index = c.frame_index
                               AND b.gt_uuid = c.gt_uuid
                        ),
                        obj_attrs AS (
                            SELECT
                                t4dataset_id,
                                frame_index,
                                uuid,
                                MAX(CAST(label AS VARCHAR)) AS label,
                                MAX(dist_h) AS dist_h
                            FROM view_eval_flat
                            WHERE source = 'GT'
                            GROUP BY 1, 2, 3
                        )
                        SELECT
                            j.t4dataset_id,
                            j.frame_index,
                            j.gt_uuid,
                            COALESCE(e.label, '') AS label,
                            COALESCE(e.dist_h, 0.0) AS dist_h,
                            j.scenario_name,
                            CASE
                                WHEN NOT j.tp_base AND j.tp_comp THEN 'improved'
                                WHEN j.tp_base AND NOT j.tp_comp THEN 'degraded'
                                WHEN j.tp_base AND j.tp_comp THEN 'both_tp'
                                ELSE 'both_fn'
                            END AS change_type,
                            j.tp_base,
                            j.tp_comp
                        FROM joined j
                        LEFT JOIN obj_attrs e
                            ON CAST(j.t4dataset_id AS VARCHAR) = CAST(e.t4dataset_id AS VARCHAR)
                           AND j.frame_index = CAST(e.frame_index AS VARCHAR)
                           AND j.gt_uuid = e.uuid
                        ORDER BY change_type, j.t4dataset_id, j.frame_index
                        """
                        try:
                            df_by_object = con.execute(query_object).df()
                            if not df_by_object.empty:
                                if change_type_filter != "all":
                                    df_by_object = df_by_object[df_by_object["change_type"] == change_type_filter]
                                if len(df_by_object) > 500:
                                    st.caption(f"Showing first 500 of {len(df_by_object)} objects.")
                                    df_by_object = df_by_object.head(500)
                                st.dataframe(df_by_object, hide_index=True)
                            else:
                                st.caption("No object breakdown.")
                        except Exception as e_obj:
                            st.caption(f"By object: {e_obj}")
            else:
                st.caption(f"Run {lbl} vs A: No data.")
        except Exception as e:
            st.error(f"Error (Run {lbl} vs A): {e}")

# =============================
# Single mode: Frame / Object level — Where are the misses?
# =============================
if single_mode:
    st.subheader("Frame / Object level: Where are the misses?")
    try:
        with st.expander("FN by frame and by object", expanded=True):
            # By frame: FN count per (t4dataset_id, frame_index, scenario_name)
            query_fn_frame = f"""
            SELECT
                t4dataset_id,
                frame_index,
                COALESCE(MAX(CAST(scenario_name AS VARCHAR)), '') AS scenario_name,
                COALESCE(MAX(CAST(suite_name AS VARCHAR)), '') AS suite_name,
                COUNT(*) AS fn_cnt
            FROM view_eval_flat
            WHERE source = 'GT' AND status = 'FN' AND {filter_clause_base}
            GROUP BY t4dataset_id, frame_index
            ORDER BY fn_cnt DESC
            """
            df_fn_frame = con.execute(query_fn_frame).df()
            if not df_fn_frame.empty:
                st.markdown("**FN count by frame**")
                st.dataframe(df_fn_frame, hide_index=True)
            else:
                st.caption("No FN by frame.")

            # By object: list of FN objects
            query_fn_object = f"""
            SELECT
                t4dataset_id,
                frame_index,
                uuid,
                COALESCE(CAST(label AS VARCHAR), '') AS label,
                dist_h,
                COALESCE(CAST(scenario_name AS VARCHAR), '') AS scenario_name,
                COALESCE(CAST(suite_name AS VARCHAR), '') AS suite_name
            FROM view_eval_flat
            WHERE source = 'GT' AND status = 'FN' AND {filter_clause_base}
            ORDER BY t4dataset_id, frame_index, uuid
            """
            df_fn_object = con.execute(query_fn_object).df()
            if not df_fn_object.empty:
                st.markdown("**FN objects**")
                if len(df_fn_object) > 500:
                    st.caption(f"Showing first 500 of {len(df_fn_object)} FN objects.")
                    st.dataframe(df_fn_object.head(500), hide_index=True)
                else:
                    st.dataframe(df_fn_object, hide_index=True)
            else:
                st.caption("No FN objects.")
    except Exception as e:
        st.error(f"Error in FN by frame/object: {e}")

# =============================
# Panel 6: Mean Error (single) / Mean Error Comparison (compare)
# =============================
st.subheader("Mean Error" + (" Comparison" if not single_mode else ""))

try:
    sample_query = "SELECT * FROM view_eval_flat LIMIT 1"
    sample_df = con.execute(sample_query).df()
    has_error_cols = all(col in sample_df.columns for col in ['x_error', 'y_error', 'yaw_error'])
except Exception:
    has_error_cols = False

if not has_error_cols:
    st.info("Error columns (x_error, y_error, yaw_error) not found in data. Skipping error analysis.")
else:
    if single_mode:
        try:
            query = f"""
            SELECT
                label,
                AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (
                    WHERE status = 'TP' AND x_error IS NOT NULL
                ) AS mean_abs_x_error,
                AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (
                    WHERE status = 'TP' AND y_error IS NOT NULL
                ) AS mean_abs_y_error,
                AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (
                    WHERE status = 'TP' AND yaw_error IS NOT NULL
                ) AS mean_abs_yaw_error
            FROM view_eval_flat
            WHERE {filter_clause_base}
            GROUP BY label
            ORDER BY label
            """
            df_error_base = con.execute(query).df()
            if not df_error_base.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_error_base['label'],
                    y=df_error_base['mean_abs_x_error'],
                    name='X Error',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=df_error_base['label'],
                    y=df_error_base['mean_abs_y_error'],
                    name='Y Error',
                    marker_color='lightcoral'
                ))
                fig.add_trace(go.Bar(
                    x=df_error_base['label'],
                    y=df_error_base['mean_abs_yaw_error'],
                    name='Yaw Error',
                    marker_color='lightgreen'
                ))
                fig.update_layout(
                    title=f"Mean Error within {max_eval_range} [m]",
                    xaxis_title="Label",
                    yaxis_title="Error [m] or [rad]",
                    barmode='group'
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        try:
            dfs_err = []
            for i in range(len(runs)):
                fc = build_filter_clause(filters_list[i])
                q = f"""
                SELECT
                    label,
                    AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND x_error IS NOT NULL) AS mean_abs_x_error,
                    AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND y_error IS NOT NULL) AS mean_abs_y_error,
                    AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (WHERE status = 'TP' AND yaw_error IS NOT NULL) AS mean_abs_yaw_error
                FROM {_flat_view(i)}
                WHERE {fc}
                GROUP BY label
                ORDER BY label
                """
                df_i = con.execute(q).df()
                df_i["run"] = run_labels_list[i]
                dfs_err.append(df_i)
            df_err_melt = pd.concat(dfs_err, ignore_index=True)
            if not df_err_melt.empty:
                for err_type, col in [("X Error", "mean_abs_x_error"), ("Y Error", "mean_abs_y_error"), ("Yaw Error", "mean_abs_yaw_error")]:
                    fig = px.bar(
                        df_err_melt,
                        x="label",
                        y=col,
                        color="run",
                        barmode="group",
                        title=f"Mean {err_type} within {max_eval_range} [m] by run",
                        labels={"label": "Label", col: err_type, "run": "Run"}
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("Difference of mean absolute error (each run − Baseline A)")
        for idx in range(1, len(runs)):
            lbl = run_labels_list[idx]
            try:
                fc_c = build_filter_clause(filters_list[idx])
                query = f"""
                WITH topic_a AS (
                    SELECT label,
                        AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS x_a,
                        AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS y_a,
                        AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS yaw_a
                    FROM view_eval_flat
                    WHERE {filter_clause_base}
                    GROUP BY label
                ),
                topic_c AS (
                    SELECT label,
                        AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS x_c,
                        AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS y_c,
                        AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS yaw_c
                    FROM {_flat_view(idx)}
                    WHERE {fc_c}
                    GROUP BY label
                )
                SELECT a.label,
                    (c.x_c - a.x_a) AS x_diff,
                    (c.y_c - a.y_a) AS y_diff,
                    (c.yaw_c - a.yaw_a) AS yaw_diff
                FROM topic_a a
                JOIN topic_c c USING (label)
                ORDER BY label
                """
                df_ed = con.execute(query).df()
                if not df_ed.empty:
                    with st.expander(f"Run {lbl} − A", expanded=(len(runs) == 2)):
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=df_ed["label"], y=df_ed["x_diff"], name="X Diff", marker_color="lightblue"))
                        fig.add_trace(go.Bar(x=df_ed["label"], y=df_ed["y_diff"], name="Y Diff", marker_color="lightcoral"))
                        fig.add_trace(go.Bar(x=df_ed["label"], y=df_ed["yaw_diff"], name="Yaw Diff", marker_color="lightgreen"))
                        fig.update_layout(title=f"Error diff ({lbl} − A) within {max_eval_range} [m]", xaxis_title="Label", yaxis_title="Error Difference [m] or [rad]", barmode="group")
                        st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Error (Run {lbl} − A): {e}")

# =============================
# Panel 8: Object Count with Distance
# =============================
st.subheader("Object count with distance")

try:
    if single_mode:
        query = f"""
        SELECT dist_h, label
        FROM view_eval_flat
        WHERE {filter_clause_base}
        """
        df_dist = con.execute(query).df()
    else:
        dfs_d = []
        for i in range(len(runs)):
            fc = build_filter_clause(filters_list[i])
            q = f"SELECT dist_h, label FROM {_flat_view(i)} WHERE {fc}"
            df_i = con.execute(q).df()
            df_i["run"] = run_labels_list[i]
            dfs_d.append(df_i)
        df_dist = pd.concat(dfs_d, ignore_index=True)
    if not df_dist.empty:
        if single_mode:
            fig = px.histogram(
                df_dist,
                x='dist_h',
                color='label',
                nbins=50,
                title="Object Count by Distance",
                labels={'dist_h': 'Distance [m]', 'label': 'Label'}
            )
        else:
            fig = px.histogram(
                df_dist,
                x='dist_h',
                color='run',
                nbins=50,
                barmode='overlay',
                opacity=0.6,
                title="Object Count by Distance (by run)",
                labels={'dist_h': 'Distance [m]', 'run': 'Run'}
            )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data available")
except Exception as e:
    st.error(f"Error: {e}")
