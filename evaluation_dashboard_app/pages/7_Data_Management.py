"""
Data Management: list runs, show size, delete runs, copy shareable links.
For multi-user server deployment so users can manage evaluation data.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

from lib.path_utils import (
    get_data_root,
    get_data_root_display,
    list_run_directories,
    get_run_info,
    delete_run,
    format_size,
)

st.set_page_config(
    page_title="Data Management",
    page_icon="📁",
    layout="wide",
)

st.title("📁 Data Management")
st.caption("List runs, free space by deleting runs, and copy shareable links to share results with others.")

data_root = get_data_root()
st.sidebar.info(f"**Data root:** `{get_data_root_display()}/`")

run_dirs = list_run_directories()
if not run_dirs:
    st.warning(f"No runs found under `{get_data_root_display()}/`. Add evaluation runs (e.g. from the Download page) to see them here.")
    st.stop()

# Build run info table
rows = []
for run_path in run_dirs:
    info = get_run_info(run_path)
    rows.append({
        "Run name": info["name"],
        "Size": format_size(info["size_bytes"]),
        "Modified": datetime.fromtimestamp(info["mtime"]).strftime("%Y-%m-%d %H:%M") if info["mtime"] else "—",
        "Summary.csv": "✓" if info["has_summary"] else "—",
        "Score.csv": "✓" if info["has_score"] else "—",
    })

st.subheader("Runs")
st.dataframe(rows, width="stretch", hide_index=True)

# Shareable link builder
st.subheader("Share result with others")
st.caption("Append the query below to your server URL to open Overview with the selected run(s).")
col_a, col_b = st.columns(2)
with col_a:
    run_names = [r["Run name"] for r in rows]
    share_run_a = st.selectbox("Baseline (A)", run_names, key="share_run_a")
with col_b:
    share_compare = st.checkbox("Compare with another run", key="share_compare")
    share_run_b = None
    if share_compare:
        share_run_b = st.selectbox(
            "Candidate (B)",
            [n for n in run_names if n != share_run_a],
            key="share_run_b",
        )
mode = "compare" if share_compare and share_run_b else "single"
q = f"mode={mode}&run_a={share_run_a}"
if mode == "compare":
    q += f"&run_b={share_run_b}"
st.code(q, language=None)
st.caption("Example: `https://your-server:8501/?` + the query above.")

# Delete section
st.subheader("Delete a run")
st.caption("Permanently remove a run and free disk space. This cannot be undone.")
del_run_name = st.selectbox(
    "Run to delete",
    options=run_names,
    key="del_run_select",
    format_func=lambda x: x,
)
confirm = st.text_input(
    "Type the run name to confirm",
    placeholder=del_run_name,
    key="del_confirm",
)
if st.button("Delete run", type="primary", key="del_btn"):
    if confirm != del_run_name:
        st.error("Confirmation text does not match the run name. Type it exactly to delete.")
    else:
        ok, msg = delete_run(del_run_name)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)
