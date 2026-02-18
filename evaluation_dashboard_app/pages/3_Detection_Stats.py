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
runB = st.session_state.get("runB")
single_mode = mode == "Single Mode"

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
        SELECT CAST(NULL AS VARCHAR) AS visibility, CAST(NULL AS VARCHAR) AS suite_name
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

def create_view_tpr_fpr(con, view_name: str = "view_tpr_fpr_by_class_dist_topic"):
    """Create TPR/FPR view."""
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
        FROM view_eval_flat
        GROUP BY
            t4dataset_id, topic_name, label, distance_bin, bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available'),
            coalesce(try(CAST(suite_name AS VARCHAR)), '')
    )
    SELECT
        *,
        CASE
            WHEN gt_total > 0 THEN CAST(tp_gt AS DOUBLE) / gt_total
            ELSE NULL
        END AS tpr,
        CASE
            WHEN est_total > 0 THEN CAST(fp_est AS DOUBLE) / est_total
            ELSE NULL
        END AS fpr
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
parquet_list_a = list_parquets_in_run(runA["path"])
if not parquet_list_a:
    st.error(f"No parquet files found in run directory: {path_display(runA['path'])}. Add a .parquet file or generate one from the Download page.")
    st.stop()

if single_mode:
    parquet_list_b = []
    compared_target_file = parquet_list_a[0]  # unused in single mode
else:
    if runB is None:
        st.warning("Compare Mode requires a Candidate (B) run. Select runs on the Overview page.")
        st.stop()
    parquet_list_b = list_parquets_in_run(runB["path"])
    if not parquet_list_b:
        st.error(f"No parquet files found in Candidate run: {path_display(runB['path'])}.")
        st.stop()

# =============================
# Loaded Runs (from Overview)
# =============================
st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{path_display(runA['path'])}`")
if not single_mode and runB:
    st.markdown(f"**Candidate (B):** `{path_display(runB['path'])}`")

# =============================
# Sidebar - Filters
# =============================
with st.sidebar:
    st.header("Filters / Inputs")
    
    # File selection (from run directories)
    if len(parquet_list_a) == 1:
        target_file = parquet_list_a[0]
    else:
        target_file = st.selectbox(
            "Baseline (A) File",
            parquet_list_a,
            format_func=lambda p: os.path.basename(p),
            key="target_file"
        )
    
    if not single_mode:
        if len(parquet_list_b) == 1:
            compared_target_file = parquet_list_b[0]
        else:
            compared_target_file = st.selectbox(
                "Candidate (B) File",
                parquet_list_b,
                format_func=lambda p: os.path.basename(p),
                index=min(1, len(parquet_list_b) - 1),
                key="compared_target_file"
            )
    
    con = get_duckdb_connection()
    print("target_file", target_file)
    # Validate parquet files are readable before creating views
    ok, msg = validate_parquet_file(con, target_file)
    if not ok:
        st.error(f"**Baseline (A) file** cannot be read: {msg}")
        st.stop()
    if not single_mode:
        ok_comp, msg_comp = validate_parquet_file(con, compared_target_file)
        if not ok_comp:
            st.error(f"**Candidate (B) file** cannot be read: {msg_comp}")
            st.stop()
    
    # Create views
    try:
        create_view_eval_flat(con, target_file, "view_eval_flat")
        create_view_tpr_fpr(con, "view_tpr_fpr_by_class_dist_topic")
        if not single_mode:
            create_view_eval_flat(con, compared_target_file, "view_eval_flat_comp")
            create_view_tpr_fpr(con, "view_tpr_fpr_by_class_dist_topic_c")
            con.execute("""
                CREATE OR REPLACE VIEW view_tpr_fpr_by_class_dist_topic_c AS
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
                    FROM view_eval_flat_comp
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
            """)
    except Exception as e:
        st.error(f"Error creating views: {e}")
        st.stop()
    
    # Topic selection
    topics = list_values(con, target_file, "topic_name")
    if topics:
        topic_name = st.selectbox(
            "Topic Name",
            ["__all__"] + topics,
            key="topic_name"
        )
    else:
        topic_name = "__all__"
    
    if not single_mode:
        compared_topics = list_values(con, compared_target_file, "topic_name")
        if compared_topics:
            compared_topic_name = st.selectbox(
                "Candidate (B) Topic Name",
                ["__all__"] + compared_topics,
                key="compared_topic_name"
            )
        else:
            compared_topic_name = "__all__"
    else:
        compared_topic_name = topic_name
    
    # Label selection
    labels = list_values(con, target_file, "label")
    if labels:
        selected_labels = st.multiselect(
            "Label(s)",
            labels,
            default=labels[:5] if len(labels) > 5 else labels,
            key="labels"
        )
    else:
        selected_labels = []
    
    # Suites selection (default: include all)
    try:
        suite_options = list_values(con, target_file, "COALESCE(CAST(suite_name AS VARCHAR), '')")
    except Exception:
        suite_options = []
    if suite_options:
        selected_suites = st.multiselect(
            "Suites",
            suite_options,
            default=suite_options,
            key="suites",
            help="Filter by suite(s). Default: all included."
        )
    else:
        selected_suites = []
    
    # Visibility selection
    vis_options = list_values(con, target_file, "COALESCE(CAST(visibility AS VARCHAR), 'not available') AS visibility")
    if vis_options:
        selected_visibility = st.multiselect(
            "Visibility",
            vis_options,
            default=vis_options,
            key="visibility"
        )
    else:
        selected_visibility = []
    
    # Max evaluation range
    max_eval_range = st.selectbox(
        "Max Evaluation Range [m]",
        [50, 80, 100, 120, 150],
        index=0,
        key="max_eval_range"
    )

# =============================
# Build Filters
# =============================
filters_base = {
    'topic_name': topic_name,
    'label': selected_labels,
    'suites': selected_suites,
    'visibility': selected_visibility,
    'max_eval_range': max_eval_range
}

filters_comp = {
    'topic_name': compared_topic_name,
    'label': selected_labels,
    'suites': selected_suites,
    'visibility': selected_visibility,
    'max_eval_range': max_eval_range
}

# =============================
# Main Content
# =============================

if st.checkbox("Debug: Inspect Parquet (Both Files)" if not single_mode else "Debug: Inspect Parquet"):
    if single_mode:
        col_left, _ = st.columns([1, 1])
        cols_used = [col_left]
        file_labels = [("Baseline (A) File", target_file)]
    else:
        col_left, col_right = st.columns(2)
        cols_used = [col_left, col_right]
        file_labels = [
            ("Baseline (A) File", target_file),
            ("Candidate (B) File", compared_target_file)
        ]
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
    if not single_mode and len(schema_results) == 2:
        label1, df1 = schema_results[0]
        label2, df2 = schema_results[1]
        cols1 = set(zip(df1["column_name"], df1["column_type"]))
        cols2 = set(zip(df2["column_name"], df2["column_type"]))
        names1 = set(df1["column_name"])
        names2 = set(df2["column_name"])

        added = names2 - names1
        removed = names1 - names2
        common = names1 & names2

        # Changed types
        types1 = {row["column_name"]: row["column_type"] for _, row in df1.iterrows()}
        types2 = {row["column_name"]: row["column_type"] for _, row in df2.iterrows()}
        dtype_changes = []
        for cname in sorted(common):
            t1 = types1.get(cname)
            t2 = types2.get(cname)
            if t1 != t2:
                dtype_changes.append((cname, t1, t2))

        with st.expander("⚖️ Difference between schemas", expanded=True):
            if not (added or removed or dtype_changes):
                st.success("✅ The schemas are identical (column names and types match exactly).")
            else:
                st.markdown("#### Schema differences (between the two files):")

                if added:
                    st.error(f"Columns only in `{label2}`: {', '.join(sorted(added))}")
                if removed:
                    st.error(f"Columns only in `{label1}`: {', '.join(sorted(removed))}")
                if dtype_changes:
                    st.warning("Columns with different types:")
                    dtype_df = pd.DataFrame(dtype_changes, columns=["Column", f"Type in {label1}", f"Type in {label2}"])
                    st.dataframe(dtype_df, hide_index=True)
            if added or removed or dtype_changes:
                # Show details for debugging purposes
                st.caption("Check above differences to adapt code or troubleshoot data loading issues.")



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
        query_base = f"""
        WITH base AS (
            SELECT COUNT(DISTINCT t4dataset_id) AS id_num, '{os.path.basename(target_file)}' AS series
            FROM view_eval_flat
        ),
        comp AS (
            SELECT COUNT(DISTINCT t4dataset_id) AS id_num, '{os.path.basename(compared_target_file)}' AS series
            FROM view_eval_flat_comp
        )
        SELECT * FROM base
        UNION ALL
        SELECT * FROM comp
        """
        df_summary = con.execute(query_base).df()
        query_status = """
        SELECT 'Baseline (A)' AS dataset, label, status, COUNT(*) AS num
        FROM view_eval_flat
        GROUP BY label, status
        UNION ALL
        SELECT 'Candidate (B)' AS dataset, label, status, COUNT(*) AS num
        FROM view_eval_flat_comp
        GROUP BY label, status
        ORDER BY dataset, label, status
        """
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
        st.write("**Status Count Table (rows=label, cols=Baseline (A) / Candidate (B) × Status)**")
        st.markdown("""
        Status count for both datasets. Columns are in the form "Baseline (A) TP", "Baseline (A) FP", etc., to allow easy comparison.
        """)
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
                title="Status Distribution per Label (by File)",
                category_orders={"dataset": ["Baseline (A)", "Candidate (B)"]},
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

if single_mode:
    try:
        filter_clause = build_filter_clause(filters_base)
        query = f"""
        SELECT
            label,
            CASE 
                WHEN COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) > 0 
                THEN CAST(COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS DOUBLE) 
                     / COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN'))
                ELSE 0
            END AS tpr
        FROM view_eval_flat
        WHERE {filter_clause}
        GROUP BY label
        ORDER BY label
        """
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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline (A) Data**")
        try:
            filter_clause = build_filter_clause(filters_base)
            query = f"""
            SELECT
                label,
                CASE 
                    WHEN COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) > 0 
                    THEN CAST(COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS DOUBLE) 
                         / COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN'))
                    ELSE 0
                END AS tpr
            FROM view_eval_flat
            WHERE {filter_clause}
            GROUP BY label
            ORDER BY label
            """
            df_tpr_base = con.execute(query).df()
            if not df_tpr_base.empty:
                fig = px.bar(
                    df_tpr_base,
                    x='label',
                    y='tpr',
                    title=f"Total TP rate within {max_eval_range} [m] with Baseline (A) data",
                    labels={'tpr': 'TP Rate', 'label': 'Label'}
                )
                fig.update_layout(yaxis_range=[0, 1.2])
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Error: {e}")
    with col2:
        st.markdown("**Candidate (B) Data**")
        try:
            filter_clause = build_filter_clause(filters_comp)
            query = f"""
            SELECT
                label,
                CASE 
                    WHEN COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) > 0 
                    THEN CAST(COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS DOUBLE) 
                         / COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN'))
                    ELSE 0
                END AS tpr
            FROM view_eval_flat_comp
            WHERE {filter_clause}
            GROUP BY label
            ORDER BY label
            """
            df_tpr_comp = con.execute(query).df()
            if not df_tpr_comp.empty:
                fig = px.bar(
                    df_tpr_comp,
                    x='label',
                    y='tpr',
                    title=f"Total TP rate within {max_eval_range} [m] with Candidate (B) data",
                    labels={'tpr': 'TP Rate', 'label': 'Label'}
                )
                fig.update_layout(yaxis_range=[0, 1.2])
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Error: {e}")

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
        filter_clause_comp = build_filter_clause(filters_comp, enable_dist_h=False)
        query = f"""
        WITH base AS (
            SELECT
                distance_bin,
                CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr,
                CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
            FROM view_tpr_fpr_by_class_dist_topic
            WHERE {filter_clause_base}
            GROUP BY distance_bin
        ),
        comp AS (
            SELECT
                distance_bin,
                CASE WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total) ELSE 0 END AS tpr_comp,
                CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr_comp
            FROM view_tpr_fpr_by_class_dist_topic_c
            WHERE {filter_clause_comp}
            GROUP BY distance_bin
        )
        SELECT
            b.distance_bin,
            b.tpr AS tp_rate_before,
            c.tpr_comp AS tp_rate_after,
            (c.tpr_comp - b.tpr) AS tp_rate_diff
        FROM base b
        JOIN comp c USING (distance_bin)
        ORDER BY CAST(REPLACE(SPLIT_PART(b.distance_bin, ',', 1), '[', ' ') AS INTEGER)
        """
        df_tpr_dist = con.execute(query).df()
        if not df_tpr_dist.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_tpr_dist['distance_bin'],
                y=df_tpr_dist['tp_rate_before'],
                name='TP Rate Baseline (A)',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=df_tpr_dist['distance_bin'],
                y=df_tpr_dist['tp_rate_after'],
                name='TP Rate Candidate (B)',
                marker_color='lightcoral'
            ))
            fig.add_trace(go.Bar(
                x=df_tpr_dist['distance_bin'],
                y=df_tpr_dist['tp_rate_diff'],
                name='TP Rate Diff',
                marker_color='lightgreen'
            ))
            fig.update_layout(
                title="TP Rate Comparison by Distance Bin",
                xaxis_title="Distance Bin",
                yaxis_title="TP Rate",
                barmode='group',
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
        filter_clause_comp = build_filter_clause(filters_comp, enable_dist_h=False)
        query = f"""
        WITH base AS (
            SELECT
                distance_bin,
                CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr
            FROM view_tpr_fpr_by_class_dist_topic
            WHERE {filter_clause_base}
            GROUP BY distance_bin
        ),
        comp AS (
            SELECT
                distance_bin,
                CASE WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total) ELSE 0 END AS fpr_comp
            FROM view_tpr_fpr_by_class_dist_topic_c
            WHERE {filter_clause_comp}
            GROUP BY distance_bin
        )
        SELECT
            b.distance_bin,
            b.fpr AS fp_rate_before,
            c.fpr_comp AS fp_rate_after,
            (c.fpr_comp - b.fpr) AS fp_rate_diff
        FROM base b
        JOIN comp c USING (distance_bin)
        ORDER BY CAST(REPLACE(SPLIT_PART(b.distance_bin, ',', 1), '[', ' ') AS INTEGER)
        """
        df_fpr_dist = con.execute(query).df()
        if not df_fpr_dist.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_fpr_dist['distance_bin'],
                y=df_fpr_dist['fp_rate_before'],
                name='FP Rate Baseline (A)',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=df_fpr_dist['distance_bin'],
                y=df_fpr_dist['fp_rate_after'],
                name='FP Rate Candidate (B)',
                marker_color='lightcoral'
            ))
            fig.add_trace(go.Bar(
                x=df_fpr_dist['distance_bin'],
                y=df_fpr_dist['fp_rate_diff'],
                name='FP Rate Diff',
                marker_color='lightgreen'
            ))
            fig.update_layout(
                title="FP Rate Comparison by Distance Bin",
                xaxis_title="Distance Bin",
                yaxis_title="FP Rate",
                barmode='group',
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
except Exception as e:
    st.error(f"Error: {e}")

# =============================
# Panel 5: Improved and Degraded Count (compare mode only)
# =============================
if not single_mode:
    st.subheader("Improved and Degraded count by objects")
    try:
        filter_clause_comp_p5 = build_filter_clause(filters_comp, enable_dist_h=False)
        query = f"""
        WITH base_gt AS (
            SELECT
                t4dataset_id,
                frame_index,
                uuid AS gt_uuid,
                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_base
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
                COUNT(*) FILTER (WHERE status = 'TP') > 0 AS tp_comp
            FROM view_eval_flat_comp
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
                COALESCE(c.tp_comp, FALSE) AS tp_comp
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
            CAST(SUM((CASE WHEN tp_comp THEN 1 ELSE 0 END) - (CASE WHEN tp_base THEN 1 ELSE 0 END)) AS DOUBLE) AS net_tp_delta
        FROM joined
        GROUP BY 1
        ORDER BY net_tp_delta DESC
        """
        df_improved = con.execute(query).df()
        if not df_improved.empty:
            st.dataframe(df_improved, width="stretch")
        else:
            st.info("No data available")
    except Exception as e:
        st.error(f"Error: {e}")

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
        filter_clause_comp_err = build_filter_clause(filters_comp)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline (A) Mean Error**")
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
                        title=f"Baseline (A) Mean Error within {max_eval_range} [m]",
                        xaxis_title="Label",
                        yaxis_title="Error [m] or [rad]",
                        barmode='group'
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No data available")
            except Exception as e:
                st.error(f"Error: {e}")
        with col2:
            st.markdown("**Candidate (B) Mean Error**")
            try:
                query = f"""
                SELECT
                    label,
                    AVG(ABS(CAST(x_error AS DOUBLE))) FILTER (WHERE status = 'TP') AS mean_abs_x_error,
                    AVG(ABS(CAST(y_error AS DOUBLE))) FILTER (WHERE status = 'TP') AS mean_abs_y_error,
                    AVG(ABS(CAST(yaw_error AS DOUBLE))) FILTER (WHERE status = 'TP') AS mean_abs_yaw_error
                FROM view_eval_flat_comp
                WHERE {filter_clause_comp_err}
                GROUP BY label
                ORDER BY label
                """
                df_error_comp = con.execute(query).df()
                if not df_error_comp.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_error_comp['label'],
                        y=df_error_comp['mean_abs_x_error'],
                        name='X Error',
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        x=df_error_comp['label'],
                        y=df_error_comp['mean_abs_y_error'],
                        name='Y Error',
                        marker_color='lightcoral'
                    ))
                    fig.add_trace(go.Bar(
                        x=df_error_comp['label'],
                        y=df_error_comp['mean_abs_yaw_error'],
                        name='Yaw Error',
                        marker_color='lightgreen'
                    ))
                    fig.update_layout(
                        title=f"Candidate (B) Mean Error within {max_eval_range} [m]",
                        xaxis_title="Label",
                        yaxis_title="Error [m] or [rad]",
                        barmode='group'
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No data available")
            except Exception as e:
                st.error(f"Error: {e}")

        # =============================
        # Panel 7: Error Difference (compare mode only)
        # =============================
        st.subheader("Difference of mean absolute error (Candidate (B) − Baseline (A))")
        try:
            query = f"""
            WITH topic_a AS (
                SELECT
                    label,
                    AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS x_a,
                    AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS y_a,
                    AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS yaw_a
                FROM view_eval_flat
                WHERE {filter_clause_base}
                GROUP BY label
            ),
            topic_b AS (
                SELECT
                    label,
                    AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS x_b,
                    AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS y_b,
                    AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS yaw_b
                FROM view_eval_flat_comp
                WHERE {filter_clause_comp_err}
                GROUP BY label
            )
            SELECT
                a.label,
                (x_b - x_a) AS x_diff,
                (y_b - y_a) AS y_diff,
                (yaw_b - yaw_a) AS yaw_diff
            FROM topic_a a
            JOIN topic_b b USING (label)
            ORDER BY label
            """
            df_error_diff = con.execute(query).df()
            if not df_error_diff.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_error_diff['label'],
                    y=df_error_diff['x_diff'],
                    name='X Diff',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=df_error_diff['label'],
                    y=df_error_diff['y_diff'],
                    name='Y Diff',
                    marker_color='lightcoral'
                ))
                fig.add_trace(go.Bar(
                    x=df_error_diff['label'],
                    y=df_error_diff['yaw_diff'],
                    name='Yaw Diff',
                    marker_color='lightgreen'
                ))
                fig.update_layout(
                    title=f"Difference of mean absolute error within {max_eval_range} [m] (Candidate (B) − Baseline (A))",
                    xaxis_title="Label",
                    yaxis_title="Error Difference [m] or [rad]",
                    barmode='group'
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Error: {e}")

# =============================
# Panel 8: Object Count with Distance
# =============================
st.subheader("Object count with distance")

try:
    query = f"""
    SELECT dist_h, label
    FROM view_eval_flat
    WHERE {filter_clause_base}
    """
    df_dist = con.execute(query).df()
    
    if not df_dist.empty:
        fig = px.histogram(
            df_dist,
            x='dist_h',
            color='label',
            nbins=50,
            title="Object Count by Distance",
            labels={'dist_h': 'Distance [m]', 'label': 'Label'}
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data available")
except Exception as e:
    st.error(f"Error: {e}")
