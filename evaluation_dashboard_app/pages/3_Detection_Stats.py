import duckdb
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import glob
import os
from typing import Optional, List, Tuple

st.set_page_config(layout="wide", page_title="Object Detection")
st.title("Object Detection Evaluation Dashboard")

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
        SELECT CAST(NULL AS VARCHAR) AS visibility
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
            COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
            COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
            COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
            COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est
        FROM view_eval_flat
        GROUP BY
            t4dataset_id, topic_name, label, distance_bin, bin_idx,
            coalesce(try(CAST(visibility AS VARCHAR)), 'not available')
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
    
    if filters.get('t4dataset_id'):
        if isinstance(filters['t4dataset_id'], list) and len(filters['t4dataset_id']) > 0:
            # Escape single quotes in IDs
            ids_escaped = [str(id).replace("'", "''") for id in filters['t4dataset_id']]
            ids_str = "', '".join(ids_escaped)
            conditions.append(f"t4dataset_id IN ('{ids_str}')")
        elif not isinstance(filters['t4dataset_id'], list) and filters['t4dataset_id'] != '__all__':
            id_escaped = str(filters['t4dataset_id']).replace("'", "''")
            conditions.append(f"t4dataset_id = '{id_escaped}'")
    
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
# Sidebar - Filters
# =============================
with st.sidebar:
    st.header("Filters / Inputs")
    
    # File selection
    parquet_files = sorted(glob.glob("data/*.parquet"))
    if not parquet_files:
        st.error("No parquet files found in data/ directory")
        st.stop()
    
    target_file = st.selectbox(
        "Target File",
        parquet_files,
        key="target_file"
    )
    
    compared_target_file = st.selectbox(
        "Compared File",
        parquet_files,
        index=min(1, len(parquet_files)-1),
        key="compared_target_file"
    )
    
    con = get_duckdb_connection()
    
    # Create views
    try:
        create_view_eval_flat(con, target_file, "view_eval_flat")
        create_view_eval_flat(con, compared_target_file, "view_eval_flat_comp")
        create_view_tpr_fpr(con, "view_tpr_fpr_by_class_dist_topic")
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
                    COUNT(*) FILTER (WHERE source='GT' AND status IN ('TP','FN')) AS gt_total,
                    COUNT(*) FILTER (WHERE source='GT' AND status='TP') AS tp_gt,
                    COUNT(*) FILTER (WHERE source='EST' AND status IN ('TP','FP')) AS est_total,
                    COUNT(*) FILTER (WHERE source='EST' AND status='FP') AS fp_est
                FROM view_eval_flat_comp
                GROUP BY
                    t4dataset_id, topic_name, label, distance_bin, bin_idx,
                    coalesce(try(CAST(visibility AS VARCHAR)), 'not available')
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
    
    compared_topics = list_values(con, compared_target_file, "topic_name")
    if compared_topics:
        compared_topic_name = st.selectbox(
            "Compared Topic Name",
            ["__all__"] + compared_topics,
            key="compared_topic_name"
        )
    else:
        compared_topic_name = "__all__"
    
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
    
    # t4dataset_id selection
    t4_ids = list_values(con, target_file, "t4dataset_id")
    if t4_ids:
        selected_t4_ids = st.multiselect(
            "t4dataset_id(s)",
            t4_ids,
            default=t4_ids[:10] if len(t4_ids) > 10 else t4_ids,
            key="t4dataset_ids"
        )
    else:
        selected_t4_ids = []
    
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
    't4dataset_id': selected_t4_ids,
    'visibility': selected_visibility,
    'max_eval_range': max_eval_range
}

filters_comp = {
    'topic_name': compared_topic_name,
    'label': selected_labels,
    't4dataset_id': selected_t4_ids,
    'visibility': selected_visibility,
    'max_eval_range': max_eval_range
}

# =============================
# Main Content
# =============================

if st.checkbox("Debug: Inspect Parquet"):
    schema_df = con.execute("""
        DESCRIBE SELECT * FROM read_parquet(?)
    """, [target_file]).df()

    stats_df = con.execute("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(t4dataset_id) AS non_null_ids,
            COUNT(DISTINCT t4dataset_id) AS distinct_ids
        FROM read_parquet(?)
    """, [target_file]).df()
    views_df = con.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
    """).df()


    st.write("**Schema (Column Names, Types)**")
    st.markdown("Shows the schema (column names and their DuckDB/Parquet datatypes) of the selected Parquet file. Useful to check data structure and types as interpreted by DuckDB.")
    st.dataframe(schema_df)
    row_options = [10, 20, 50, 100, 200, "All"]
    row_choice = st.selectbox("Preview rows to show", row_options, index=1, key="preview_row_limit")
    if row_choice == "All":
        limit_clause = ""
    else:
        limit_clause = f"LIMIT {row_choice}"
    preview_df = con.execute(f"""
        SELECT *
        FROM read_parquet(?)
        {limit_clause}
    """, [target_file]).df()
    st.write("**Preview (First preview rows)**")
    st.markdown(f"Shows the first {row_choice} preview rows from the Parquet file. Use this preview to examine example data contents and check that your file is as expected.")
    st.dataframe(preview_df)

    st.write("**Stats (Row Count, t4dataset_id non-null count, Distinct t4dataset_id count)**")
    st.markdown("""
    - `total_rows`: Total rows in the file  
    - `non_null_ids`: Rows where t4dataset_id is not null  
    - `distinct_ids`: Unique t4dataset_id values

    This helps rapidly assess the completeness and distribution of the key ID field.
    """)
    st.dataframe(stats_df)


# =============================
# Panel 1: t4dataset Summary
# =============================
st.subheader("t4dataset summary and data parse")

try:
    # Basic summary as before
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

    # Get status counts grouped by dataset, label, status
    query_status = f"""
    SELECT 
        'Target' AS dataset,
        label,
        status,
        COUNT(*) AS num
    FROM view_eval_flat
    GROUP BY label, status
    UNION ALL
    SELECT 
        'Compared' AS dataset,
        label,
        status,
        COUNT(*) AS num
    FROM view_eval_flat_comp
    GROUP BY label, status
    ORDER BY dataset, label, status
    """
    df_status = con.execute(query_status).df()

    st.write("**t4dataset Count Summary**")
    if not df_summary.empty:
        fig = px.bar(
            df_summary,
            x='series',
            y='id_num',
            title="t4dataset Count",
            labels={'id_num': 'Count', 'series': 'File'}
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No data available")

    # ---- Enhanced Table View: Status × label, wide format for better visibility ----
    st.write("**Status Count Table (rows=label, cols=Target/Compared × Status)**")
    st.markdown("""
    Status count for both datasets. Columns are in the form "Target TP", "Target FP", etc., to allow easy comparison.
    """)

    if not df_status.empty:
        # Pivot table for wide display: index=label, columns=[dataset, status], values=num
        df_status_wide = df_status.pivot_table(index='label', columns=['dataset', 'status'], values='num', fill_value=0)

        # Flatten columns to e.g. "Target TP"
        df_status_wide.columns = [f"{col[0]} {col[1]}" for col in df_status_wide.columns]
        df_status_wide = df_status_wide.reset_index()

        st.dataframe(df_status_wide)

        # Also: stacked bar plot for label × status for each dataset
        fig2 = px.bar(
            df_status,
            x="label",
            y="num",
            color="status",
            barmode="stack",
            facet_col="dataset",
            title="Status Distribution per Label (by File)",
            category_orders={"dataset": ["Target", "Compared"]},
            labels={"num": "Count", "label": "Label", "status": "Status"}
        )
        st.plotly_chart(fig2, width="stretch")

    else:
        st.info("No status count data available")

except Exception as e:
    st.error(f"Error in summary: {e}")

# =============================
# Panel 2: TP Rate Comparison
# =============================
st.subheader("TP Rate Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Target Data**")
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
                title=f"Total TP rate within {max_eval_range} [m] with target data",
                labels={'tpr': 'TP Rate', 'label': 'Label'}
            )
            fig.update_layout(yaxis_range=[0, 1.2])
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data available")
    except Exception as e:
        st.error(f"Error: {e}")

with col2:
    st.markdown("**Compared Data**")
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
                title=f"Total TP rate within {max_eval_range} [m] with compared data",
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
st.subheader("TP Rate Comparison by Distance Bin")

try:
    filter_clause_base = build_filter_clause(filters_base, enable_dist_h=False)
    filter_clause_comp = build_filter_clause(filters_comp, enable_dist_h=False)
    
    query = f"""
    WITH base AS (
        SELECT
            distance_bin,
            CASE 
                WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total)
                ELSE 0
            END AS tpr,
            CASE 
                WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total)
                ELSE 0
            END AS fpr
        FROM view_tpr_fpr_by_class_dist_topic
        WHERE {filter_clause_base}
        GROUP BY distance_bin
    ),
    comp AS (
        SELECT
            distance_bin,
            CASE 
                WHEN SUM(gt_total) > 0 THEN CAST(SUM(tp_gt) AS DOUBLE) / SUM(gt_total)
                ELSE 0
            END AS tpr_comp,
            CASE 
                WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total)
                ELSE 0
            END AS fpr_comp
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
            name='TP Rate Before',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=df_tpr_dist['distance_bin'],
            y=df_tpr_dist['tp_rate_after'],
            name='TP Rate After',
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
    query = f"""
    WITH base AS (
        SELECT
            distance_bin,
            CASE 
                WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total)
                ELSE 0
            END AS fpr
        FROM view_tpr_fpr_by_class_dist_topic
        WHERE {filter_clause_base}
        GROUP BY distance_bin
    ),
    comp AS (
        SELECT
            distance_bin,
            CASE 
                WHEN SUM(est_total) > 0 THEN CAST(SUM(fp_est) AS DOUBLE) / SUM(est_total)
                ELSE 0
            END AS fpr_comp
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
            name='FP Rate Before',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=df_fpr_dist['distance_bin'],
            y=df_fpr_dist['fp_rate_after'],
            name='FP Rate After',
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
# Panel 5: Improved and Degraded Count
# =============================
st.subheader("Improved and Degraded count by objects")

try:
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
            AND {filter_clause_comp}
        GROUP BY 1,2,3
    ),
    joined AS (
        SELECT
            COALESCE(b.t4dataset_id, c.t4dataset_id) AS t4dataset_id,
            COALESCE(b.frame_index, c.frame_index) AS frame_index,
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
# Panel 6: Mean Error Comparison
# =============================
st.subheader("Mean Error Comparison")

# Check if error columns exist
try:
    sample_query = "SELECT * FROM view_eval_flat LIMIT 1"
    sample_df = con.execute(sample_query).df()
    has_error_cols = all(col in sample_df.columns for col in ['x_error', 'y_error', 'yaw_error'])
except:
    has_error_cols = False

if not has_error_cols:
    st.info("Error columns (x_error, y_error, yaw_error) not found in data. Skipping error analysis.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Target File Mean Error**")
        try:
            query = f"""
            SELECT
                label,
                AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS mean_abs_x_error,
                AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS mean_abs_y_error,
                AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS mean_abs_yaw_error
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
                    title=f"TargetFile Mean Error within {max_eval_range} [m]",
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
        st.markdown("**Compared File Mean Error**")
        try:
            query = f"""
            SELECT
                label,
                AVG(ABS(x_error)) FILTER (WHERE status = 'TP') AS mean_abs_x_error,
                AVG(ABS(y_error)) FILTER (WHERE status = 'TP') AS mean_abs_y_error,
                AVG(ABS(yaw_error)) FILTER (WHERE status = 'TP') AS mean_abs_yaw_error
            FROM view_eval_flat_comp
            WHERE {filter_clause_comp}
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
                    title=f"ComparedFile Mean Error within {max_eval_range} [m]",
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
    # Panel 7: Error Difference
    # =============================
    st.subheader("Difference of mean absolute error (comp - target)")

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
            WHERE {filter_clause_comp}
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
                title=f"Difference of mean absolute error within {max_eval_range} [m] (comp - target)",
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
