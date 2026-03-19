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
from lib.page_chrome import inject_app_page_styles, render_page_hero, section_header

st.set_page_config(
    page_title="Data Management",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_app_page_styles()
render_page_hero(
    kicker="Workspace",
    title="Data management",
    description=(
        "Browse runs under the data root, check Summary/Score/Parquet flags, copy shareable Overview links, "
        "and delete runs to free disk space."
    ),
    mode="Single Run",
)

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
        "Parquet": "✓" if info["has_parquet"] else "—",
    })

section_header("Runs in data root", "Size, modified time, and which artifacts exist per run.")
st.dataframe(rows, width="stretch", hide_index=True)

# Shareable link builder
section_header("Share with teammates", "Compose a query string for Overview — append it to your app base URL.")
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
section_header("Delete a run", "Permanent — frees disk space. Use with care on shared servers.")
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
