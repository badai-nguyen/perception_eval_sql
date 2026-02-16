"""
Parquet Debug page: inspect format and contents of parquet and PKL files.
"""
import duckdb
import pickle
import streamlit as st
import pandas as pd
import glob
import os
import pathlib
from typing import Any, List, Optional, Tuple

st.set_page_config(layout="wide", page_title="Parquet & PKL Debug")
st.title("Parquet & PKL File Inspector")

# =============================
# DuckDB
# =============================
def get_con():
    return duckdb.connect()

# =============================
# PKL helpers
# =============================
def _type_summary(obj: Any) -> str:
    """One-line summary: type and len/shape."""
    t = type(obj).__name__
    if isinstance(obj, pd.DataFrame):
        return f"{t} {obj.shape}"
    if hasattr(obj, "shape"):
        return f"{t} shape={getattr(obj, 'shape', '?')}"
    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
        try:
            return f"{t} len={len(obj)}"
        except Exception:
            return t
    return t


def _describe_pkl(obj: Any, depth: int = 0, max_depth: int = 5) -> List[str]:
    """Describe type and structure of a PKL object recursively (full text)."""
    lines = []
    indent = "  " * depth
    if depth > max_depth:
        lines.append(f"{indent}... (max depth)")
        return lines
    t = type(obj).__name__
    if isinstance(obj, pd.DataFrame):
        lines.append(f"{indent}{t} shape={obj.shape}, columns={list(obj.columns)}")
        return lines
    if hasattr(obj, "shape"):
        lines.append(f"{indent}{t} shape={getattr(obj, 'shape', '?')}")
        return lines
    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
        try:
            n = len(obj)
        except Exception:
            n = "?"
        lines.append(f"{indent}{t} len={n}")
    else:
        lines.append(f"{indent}{t}")

    if isinstance(obj, dict):
        for k, v in list(obj.items())[:15]:
            lines.append(f"{indent}  [{repr(k)}]:")
            lines.extend(_describe_pkl(v, depth + 2, max_depth))
        if len(obj) > 15:
            lines.append(f"{indent}  ... and {len(obj) - 15} more keys")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:6]):
            lines.append(f"{indent}  [{i}]:")
            lines.extend(_describe_pkl(v, depth + 2, max_depth))
        if len(obj) > 6:
            lines.append(f"{indent}  ... and {len(obj) - 6} more items")
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, v in list(obj.__dict__.items())[:12]:
            lines.append(f"{indent}  .{k}:")
            lines.extend(_describe_pkl(v, depth + 2, max_depth))
        if len(obj.__dict__) > 12:
            lines.append(f"{indent}  ... and {len(obj.__dict__) - 12} more attrs")
    return lines


def _summary_rows(obj: Any, prefix: str = "") -> List[Tuple[str, str, str]]:
    """List of (attribute_name, type_name, brief) for tables. No deep recursion."""
    rows: List[Tuple[str, str, str]] = []
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:30]:
            name = f"{prefix}[{repr(k)}]" if prefix else repr(k)
            rows.append((name, type(v).__name__, _type_summary(v)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:20]):
            name = f"{prefix}[{i}]" if prefix else f"[{i}]"
            rows.append((name, type(v).__name__, _type_summary(v)))
        if len(obj) > 20:
            rows.append(("...", "", f"+ {len(obj) - 20} more items"))
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for k, v in list(obj.__dict__.items()):
            name = f".{k}" if not prefix else f"{prefix}.{k}"
            rows.append((name, type(v).__name__, _type_summary(v)))
    return rows


def _to_preview_value(obj: Any, max_depth: int = 3) -> Any:
    """Convert object to JSON-serializable preview with actual values (no object addresses)."""
    if max_depth <= 0:
        return _type_summary(obj)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:500] + ("…" if len(obj) > 500 else "")
    if isinstance(obj, bytes):
        return repr(obj)[:200]
    # numpy and pandas scalars
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return {"_type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    except ImportError:
        pass
    if isinstance(obj, pd.DataFrame):
        return {"_type": "DataFrame", "shape": list(obj.shape), "columns": list(obj.columns)}
    if isinstance(obj, dict):
        return {repr(k): _to_preview_value(v, max_depth - 1) for k, v in list(obj.items())[:20]}
    if isinstance(obj, (list, tuple)):
        preview = [_to_preview_value(v, max_depth - 1) for v in obj[:10]]
        if len(obj) > 10:
            preview.append(f"... +{len(obj) - 10} more")
        return preview
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {
            k: _to_preview_value(v, max_depth - 1)
            for k, v in list(obj.__dict__.items())
        }
    # Enum: show value for readability
    if hasattr(obj, "name") and hasattr(obj, "value"):
        return str(obj.value)
    s = repr(obj)
    if s.startswith("<") and " at 0x" in s:
        return _type_summary(obj)  # avoid useless object-address repr
    return s[:300] + ("…" if len(s) > 300 else "")


def _render_tree_expander(obj: Any, label: str, depth: int, max_depth: int, key_prefix: str) -> None:
    """Render one level as expander; children inside it."""
    if depth > max_depth:
        st.caption(f"… {_type_summary(obj)}")
        return
    summary = _type_summary(obj)
    with st.expander(f"**{label}** — {summary}", expanded=(depth < 2)):
        if isinstance(obj, pd.DataFrame):
            st.dataframe(obj.head(20), use_container_width=True)
        elif isinstance(obj, dict):
            for k, v in list(obj.items())[:15]:
                _render_tree_expander(v, f"[{repr(k)}]", depth + 1, max_depth, f"{key_prefix}_{repr(k)}")
            if len(obj) > 15:
                st.caption(f"… and {len(obj) - 15} more keys")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj[:8]):
                _render_tree_expander(v, f"[{i}]", depth + 1, max_depth, f"{key_prefix}_{i}")
            if len(obj) > 8:
                st.caption(f"… and {len(obj) - 8} more items")
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            for k, v in list(obj.__dict__.items()):
                _render_tree_expander(v, f".{k}", depth + 1, max_depth, f"{key_prefix}_{k}")
        else:
            st.text(repr(obj)[:2000] + ("…" if len(repr(obj)) > 2000 else ""))


def load_pkl_file(path: str) -> Any:
    """Load .pkl with pickle, .pkl.z with joblib."""
    path_lower = path.lower()
    if path_lower.endswith(".pkl.z"):
        try:
            import joblib
            return joblib.load(path)
        except ImportError:
            raise ImportError("joblib is required for .pkl.z: pip install joblib")
    with open(path, "rb") as f:
        return pickle.load(f)

# =============================
# Sidebar – file type and selection
# =============================
with st.sidebar:
    st.header("File selection")
    file_type = st.radio("File type", ["Parquet", "PKL"], key="debug_file_type")
    use_custom = st.checkbox("Use custom file path", value=False, key="pq_debug_custom")

    if file_type == "Parquet":
        parquet_files = sorted(str(p) for p in pathlib.Path("data").rglob("*.parquet"))
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
    else:
        pkl_files = sorted(str(p) for p in pathlib.Path("data").rglob("*.pkl"))
        pklz_files = sorted(str(p) for p in pathlib.Path("data").rglob("*.pkl.z"))
        all_pkl = pkl_files + [p for p in pklz_files if p not in pkl_files]
        if use_custom:
            target_file = st.text_input(
                "PKL file path",
                value="data/example/scene_result.pkl",
                key="pkl_debug_path",
                help="Absolute or relative path to a .pkl or .pkl.z file",
            )
            if not target_file or not target_file.strip():
                st.warning("Enter a file path")
                st.stop()
            target_file = target_file.strip()
        else:
            if not all_pkl:
                st.error("No .pkl or .pkl.z files found in data/")
                st.stop()
            target_file = st.selectbox("PKL file", all_pkl, key="pkl_debug_file")

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
# PARQUET VIEW
# =============================
if file_type == "Parquet":
    # Optional: PyArrow metadata (row groups, schema from file)
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

    st.subheader("Schema (column names and types)")
    st.caption("DuckDB interpretation of parquet column names and types.")
    st.dataframe(schema_df, use_container_width=True)

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

    st.subheader("Column statistics")
    st.caption("Null counts, distinct counts, and min/max for numeric columns.")
    columns = schema_df["column_name"].tolist()
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

    with st.expander("Run custom SQL on this parquet file"):
        st.caption("Use placeholder ? for the file path. Example: SELECT * FROM read_parquet(?) LIMIT 10")
        custom_sql = st.text_area("SQL", value="SELECT * FROM read_parquet(?) LIMIT 10", key="pq_custom_sql")
        if st.button("Run"):
            try:
                custom_df = con.execute(custom_sql, [target_file]).df()
                st.dataframe(custom_df, use_container_width=True)
            except Exception as e:
                st.error(str(e))

# =============================
# PKL VIEW
# =============================
else:
    st.subheader("File info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Path", target_file)
    with col2:
        if file_info["size_bytes"] is not None:
            size_mb = file_info["size_bytes"] / (1024 * 1024)
            st.metric("Size", f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_info['size_bytes'] / 1024:.2f} KB")
        else:
            st.metric("Size", "—")

    try:
        pkl_data = load_pkl_file(target_file)
    except Exception as e:
        st.error(f"Failed to load PKL: {e}")
        st.stop()

    st.subheader("Root object")
    st.caption(f"Type: **{type(pkl_data).__module__}.{type(pkl_data).__name__}**")

    st.subheader("Structure")
    view_mode = st.radio(
        "View",
        ["Summary (table)", "Tree (expandable)", "Full text"],
        horizontal=True,
        key="pkl_structure_view",
        help="Summary: compact attribute table. Tree: expand/collapse nodes. Full text: raw structure dump.",
    )

    if view_mode == "Summary (table)":
        # Root summary
        root_summary = _type_summary(pkl_data)
        st.markdown(f"**Root:** `{root_summary}`")
        rows = _summary_rows(pkl_data)
        if rows:
            summary_df = pd.DataFrame(rows, columns=["Attribute", "Type", "Summary"])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        # If root is a list of objects, show "first item" attribute table so users see one frame's shape
        if isinstance(pkl_data, (list, tuple)) and len(pkl_data) > 0:
            first = pkl_data[0]
            if hasattr(first, "__dict__") and not isinstance(first, type):
                st.markdown("**First item attributes** (shape of one element)")
                first_rows = _summary_rows(first, prefix="[0]")
                first_df = pd.DataFrame(first_rows, columns=["Attribute", "Type", "Summary"])
                st.dataframe(first_df, use_container_width=True, hide_index=True)
                idx = st.selectbox(
                    "Or summarize item at index",
                    options=list(range(min(10, len(pkl_data)))),
                    key="pkl_summary_index",
                )
                if idx != 0:
                    other = pkl_data[idx]
                    if hasattr(other, "__dict__"):
                        other_rows = _summary_rows(other, prefix=f"[{idx}]")
                        other_df = pd.DataFrame(other_rows, columns=["Attribute", "Type", "Summary"])
                        st.dataframe(other_df, use_container_width=True, hide_index=True)
    elif view_mode == "Tree (expandable)":
        tree_depth = st.slider("Max expand depth", 1, 5, 2, key="pkl_tree_depth")
        _render_tree_expander(pkl_data, "root", depth=0, max_depth=tree_depth, key_prefix="pkl_tree")
    else:
        structure_lines = _describe_pkl(pkl_data)
        st.text("\n".join(structure_lines))

    # If root is a DataFrame, show preview
    if isinstance(pkl_data, pd.DataFrame):
        st.subheader("DataFrame preview")
        n_pkl = st.slider("Rows to show", 5, 500, 50, key="pkl_n_rows")
        st.dataframe(pkl_data.head(n_pkl), use_container_width=True)
    else:
        # Collect any DataFrames found in the structure for optional preview
        def find_dataframes(obj: Any, path: str = "root") -> List[Tuple[str, pd.DataFrame]]:
            out = []
            if isinstance(obj, pd.DataFrame):
                out.append((path, obj))
                return out
            if isinstance(obj, dict):
                for k, v in obj.items():
                    out.extend(find_dataframes(v, f"{path}[{repr(k)}]"))
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj[:20]):
                    out.extend(find_dataframes(v, f"{path}[{i}]"))
            elif hasattr(obj, "__dict__") and not isinstance(obj, type):
                for k, v in obj.__dict__.items():
                    out.extend(find_dataframes(v, f"{path}.{k}"))
            return out

        dfs_found = find_dataframes(pkl_data)
        if dfs_found:
            st.subheader("DataFrames inside PKL")
            df_choice = st.selectbox(
                "Select DataFrame",
                options=list(range(len(dfs_found))),
                format_func=lambda i: f"{dfs_found[i][0]} (shape {dfs_found[i][1].shape})",
                key="pkl_df_choice",
            )
            path_label, df_preview = dfs_found[df_choice]
            st.caption(f"Path: `{path_label}`")
            n_pkl = st.slider("Rows to show", 5, 500, 50, key="pkl_n_rows")
            st.dataframe(df_preview.head(n_pkl), use_container_width=True)

    st.subheader("Sample values")
    st.caption("Actual content of the first few items (no object addresses).")
    n_sample = st.slider("Number of items to preview", 1, 5, 2, key="pkl_n_sample")
    preview_depth = st.slider("Preview depth", 1, 4, 2, key="pkl_preview_depth")
    try:
        if isinstance(pkl_data, (list, tuple)):
            to_show = pkl_data[:n_sample]
            for i, item in enumerate(to_show):
                preview = _to_preview_value(item, max_depth=preview_depth)
                with st.expander(f"Item [{i}]", expanded=True):
                    st.json(preview)
        else:
            preview = _to_preview_value(pkl_data, max_depth=preview_depth)
            st.json(preview)
    except Exception as e:
        st.warning(f"Could not build preview: {e}. Showing type summary.")
        st.text(_type_summary(pkl_data))

    with st.expander("Debug: raw repr (first 10k chars)"):
        raw_repr = repr(pkl_data)
        if len(raw_repr) > 10_000:
            raw_repr = raw_repr[:10_000] + "\n... (truncated)"
        st.text(raw_repr)
