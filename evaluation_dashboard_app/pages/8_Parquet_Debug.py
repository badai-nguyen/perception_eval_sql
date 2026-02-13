"""
Parquet Debug page: inspect format and contents of a single parquet file.
"""
import duckdb
import streamlit as st
import pandas as pd
import glob
import os
import pathlib
from typing import Optional

st.set_page_config(layout="wide", page_title="Parquet Debug")
st.title("Parquet File Inspector")

# =============================
# DuckDB
# =============================
def get_con():
    return duckdb.connect()

# =============================
# Sidebar – file selection
# =============================
with st.sidebar:
    st.header("File selection")
    parquet_files = sorted(str(p) for p in pathlib.Path("data").rglob("*.parquet"))
    use_custom = st.checkbox("Use custom file path", value=False, key="pq_debug_custom")
    if use_custom:
        target_file = st.text_input(
            "Parquet file path",
            value="data/example.parquet",
            key="pq_debug_path",
            help="Absolute or relative path to a .parquet file",
        )
        if not target_file or not target_file.strip():
            st.warning("Enter a file path")
            st.stop()
        target_file = target_file.strip()
    else:
        if not parquet_files:
            st.error("No parquet files found in data/")
            st.stop()
        target_file = st.selectbox("Parquet file", parquet_files, key="pq_debug_file")

    if not os.path.isfile(target_file):
        st.error(f"File not found: {target_file}")
        st.stop()

# =============================
# File metadata (size, etc.)
# =============================
def get_file_info(path: str) -> dict:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    return {"path": path, "size_bytes": size}

file_info = get_file_info(target_file)

# =============================
# Optional: PyArrow metadata (row groups, schema from file)
# =============================
def get_pyarrow_metadata(path: str) -> Optional[dict]:
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        return {
            "num_row_groups": meta.num_row_groups,
            "num_rows": meta.num_rows,
            "num_columns": meta.num_columns,
            "serialized_size": meta.serialized_size,
        }
    except Exception as e:
        return {"error": str(e)}

pyarrow_meta = get_pyarrow_metadata(target_file)

# =============================
# DuckDB: schema, row count, sample
# =============================
con = get_con()
try:
    schema_df = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [target_file]
    ).df()
except Exception as e:
    st.error(f"Failed to read parquet: {e}")
    st.stop()

try:
    row_count_df = con.execute(
        "SELECT COUNT(*) AS row_count FROM read_parquet(?)", [target_file]
    ).df()
    row_count = int(row_count_df.at[0, "row_count"])
except Exception:
    row_count = None

# =============================
# Render: file info
# =============================
st.subheader("File info")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Path", target_file)
with col2:
    if file_info["size_bytes"] is not None:
        size_mb = file_info["size_bytes"] / (1024 * 1024)
        st.metric("Size", f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_info['size_bytes'] / 1024:.2f} KB")
    else:
        st.metric("Size", "—")
with col3:
    st.metric("Rows", f"{row_count:,}" if row_count is not None else "—")

if pyarrow_meta and "error" not in pyarrow_meta:
    with st.expander("PyArrow file metadata (row groups, etc.)"):
        st.json(pyarrow_meta)
elif pyarrow_meta and "error" in pyarrow_meta:
    with st.expander("PyArrow metadata (optional)"):
        st.caption("PyArrow not available or error: " + pyarrow_meta["error"])

# =============================
# Schema (column names and types)
# =============================
st.subheader("Schema (column names and types)")
st.caption("DuckDB interpretation of parquet column names and types.")
st.dataframe(schema_df, use_container_width=True)

# =============================
# Preview: first / last N rows
# =============================
st.subheader("Preview rows")
preview_mode = st.radio(
    "Show",
    ["First N rows", "Last N rows", "First and last N rows"],
    horizontal=True,
    key="pq_preview_mode",
)
n_rows = st.slider("Number of rows (N)", 5, 500, 50, key="pq_n_rows")

def run_preview(path: str, limit: int, from_end: bool) -> pd.DataFrame:
    if from_end:
        # Get total count then offset
        total = con.execute("SELECT COUNT(*) AS c FROM read_parquet(?)", [path]).df().at[0, "c"]
        offset = max(0, total - limit)
        return con.execute(
            "SELECT * FROM read_parquet(?) LIMIT ? OFFSET ?",
            [path, limit, offset],
        ).df()
    return con.execute(
        "SELECT * FROM read_parquet(?) LIMIT ?",
        [path, limit],
    ).df()

if preview_mode == "First N rows":
    preview_df = run_preview(target_file, n_rows, from_end=False)
    st.dataframe(preview_df, use_container_width=True)
elif preview_mode == "Last N rows":
    preview_df = run_preview(target_file, n_rows, from_end=True)
    st.dataframe(preview_df, use_container_width=True)
else:
    c1, c2 = st.columns(2)
    with c1:
        st.write("**First N rows**")
        st.dataframe(run_preview(target_file, n_rows, from_end=False), use_container_width=True)
    with c2:
        st.write("**Last N rows**")
        st.dataframe(run_preview(target_file, n_rows, from_end=True), use_container_width=True)

# =============================
# Column statistics (nulls, distinct, min/max for numeric)
# =============================
st.subheader("Column statistics")
st.caption("Null counts, distinct counts, and min/max for numeric columns.")

columns = schema_df["column_name"].tolist()
# One aggregate query: COUNT(*) plus per-column COUNT(col) and COUNT(DISTINCT col)
agg_exprs = ["COUNT(*) AS total"]
for col in columns:
    safe = f'"{col}"' if not col.isidentifier() else col
    ckey = col.replace(" ", "_").replace("-", "_")
    agg_exprs.append(f"COUNT({safe}) AS non_null_{ckey}")
    agg_exprs.append(f"COUNT(DISTINCT {safe}) AS distinct_{ckey}")

try:
    agg_df = con.execute(
        f"SELECT {', '.join(agg_exprs)} FROM read_parquet(?)",
        [target_file],
    ).df()
except Exception as e:
    st.warning(f"Could not compute column stats: {e}")
    agg_df = None

if agg_df is not None and not agg_df.empty:
    total = int(agg_df["total"].iloc[0])
    stats_rows = []
    for col in columns:
        ckey = col.replace(" ", "_").replace("-", "_")
        nn_key = f"non_null_{ckey}"
        dist_key = f"distinct_{ckey}"
        if nn_key not in agg_df.columns or dist_key not in agg_df.columns:
            continue
        non_null = int(agg_df[nn_key].iloc[0])
        distinct = int(agg_df[dist_key].iloc[0])
        null_count = total - non_null
        stats_rows.append({
            "column": col,
            "null_count": null_count,
            "non_null_count": non_null,
            "distinct_count": distinct,
        })
    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(stats_df, use_container_width=True)

# =============================
# Sample values for selected column (for debugging categorical/string)
# =============================
st.subheader("Sample values by column")
col_choice = st.selectbox("Column", columns, key="pq_sample_col")
n_sample = st.slider("Number of sample values to show", 5, 100, 20, key="pq_n_sample")

safe_col = f'"{col_choice}"' if not col_choice.isidentifier() else col_choice
try:
    sample_df = con.execute(
        f"SELECT DISTINCT {safe_col} AS value FROM read_parquet(?) ORDER BY value LIMIT ?",
        [target_file, n_sample],
    ).df()
    st.dataframe(sample_df, use_container_width=True)
except Exception as e:
    st.warning(f"Could not sample column: {e}")

# =============================
# Raw SQL (optional)
# =============================
with st.expander("Run custom SQL on this parquet file"):
    st.caption("Use placeholder ? for the file path. Example: SELECT * FROM read_parquet(?) LIMIT 10")
    custom_sql = st.text_area("SQL", value="SELECT * FROM read_parquet(?) LIMIT 10", key="pq_custom_sql")
    if st.button("Run"):
        try:
            custom_df = con.execute(custom_sql, [target_file]).df()
            st.dataframe(custom_df, use_container_width=True)
        except Exception as e:
            st.error(str(e))
