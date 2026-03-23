import html
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.path_utils import path_display
from lib.page_chrome import (
    inject_app_page_styles,
    render_loaded_data_section,
    render_page_hero,
    section_header,
)
from lib.criteria_absolute_gates import (
    MetricGateSpec,
    evaluate_scenario_gates,
    export_gate_result,
    failing_scenarios_table,
    gate_summary,
    infer_criteria_count,
)

st.set_page_config(
    layout="wide",
    page_title="Criteria Score",
    page_icon="📊",
    initial_sidebar_state="expanded",
)


# Plotly theme (A/B palette aligned with run cards)
_PX_COLOR_RUN = {"Baseline (A)": "#312e81", "Candidate (B)": "#0f766e"}
_PX_COLOR_QUAL = px.colors.qualitative.Bold


def _plotly_apply_theme(fig, title: str, height: int = 440) -> None:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=16, color="#0f172a"), x=0, xanchor="left", pad=dict(t=8, b=12)),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color="#334155"),
        paper_bgcolor="rgba(248, 250, 252, 0.92)",
        plot_bgcolor="rgba(255, 255, 255, 0.95)",
        margin=dict(l=56, r=28, t=72, b=52),
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="system-ui"),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=False)


def _safe_default(default_lst, options_lst):
    """Keep only defaults that still exist in options (avoids Streamlit multiselect errors)."""
    if not isinstance(default_lst, list):
        default_lst = list(default_lst) if default_lst is not None else []
    options_set = set(options_lst)
    return [x for x in default_lst if x in options_set]


def _perception_label_options_from_runs(*runs) -> list:
    pl: set = set()
    for run in runs:
        if not run:
            continue
        summary = run.get("summary")
        if summary is None or "perception_label" not in summary.columns:
            continue
        pl.update(
            x for x in summary["perception_label"].dropna().unique() if str(x).strip() != ""
        )
    return sorted(pl)


def _filter_df_view_by_perception_labels(
    df_view: pd.DataFrame,
    summary: pd.DataFrame | None,
    selected_labels: list,
) -> pd.DataFrame:
    """Restrict Score rows to scenarios whose Summary perception_label is selected (Overview-style)."""
    if not selected_labels or df_view is None or df_view.empty:
        return df_view
    if summary is None or summary.empty or "perception_label" not in summary.columns:
        return df_view
    if "id" not in summary.columns or "Scenario" not in df_view.columns:
        return df_view
    s = summary[["id", "perception_label"]].copy()
    s["id"] = s["id"].astype(str)
    s = s[s["perception_label"].notna() & (s["perception_label"].astype(str).str.strip() != "")]
    s = s[s["perception_label"].isin(selected_labels)]
    allowed = set(s["id"].unique())
    if not allowed:
        return df_view.iloc[0:0].copy()
    return df_view.loc[df_view["Scenario"].astype(str).isin(allowed)].copy()


# =========================
# Safety check
# =========================
if "runA" not in st.session_state:
    st.warning("Please load data from the Overview page first.")
    st.stop()

mode = st.session_state.get("mode", "Single Run")

runA = st.session_state["runA"]
df_raw_A = runA.get("score")
if df_raw_A is None:
    st.warning("This run has no **Score.csv**. Load a run that includes Score.csv for this page. Detection Stats and Bounding Box Viewer work with parquet-only runs.")
    st.stop()

runB = st.session_state.get("runB")
df_raw_B = runB["score"] if runB else None

inject_app_page_styles()

_path_b = path_display(runB["path"]) if runB else None
if mode == "Compare Mode" and _path_b:
    _ld_entries = [
        ("Baseline · A", path_display(runA["path"])),
        ("Candidate · B", _path_b),
    ]
else:
    _ld_entries = [("Current run", path_display(runA["path"]))]
render_loaded_data_section(_ld_entries)
if mode == "Compare Mode":
    if df_raw_B is None:
        st.warning("Compare Mode requires a Candidate (B) run from the Overview page.")
        st.stop()
# =========================
# Constants
# =========================

BASE_COLS = ["Scenario", "Option", "GT_OBJ"]

CRITERIA_COLS = [
    "distance",
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
    "obj_cnts",
]

BLOCK_COLS = [
    "distance",
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
    "obj_cnts",
]

BLOCK_SIZE = len(CRITERIA_COLS)

NUM_COLS = [
    "distance",
    "nm",
    "tp_tn",
    "add",
    "ail",
    "uil",
    "pfn_pfp",
    "uuid_num",
    "pass_rate",
    "max_dist_thresh",
]

_criteria_n_a = infer_criteria_count(df_raw_A, BLOCK_SIZE)
if mode == "Compare Mode" and df_raw_B is not None:
    CRITERIA_COUNT = min(_criteria_n_a, infer_criteria_count(df_raw_B, BLOCK_SIZE))
else:
    CRITERIA_COUNT = _criteria_n_a

st.sidebar.markdown("##### Scope")
criteria_idx = st.sidebar.selectbox(
    "Criteria block",
    list(range(CRITERIA_COUNT)),
    format_func=lambda x: f"criteria{x}",
    help="Each block is one criteria set from Score.csv (distance, NM, pass rate, …).",
)
show_debug = st.sidebar.checkbox("Show debug tables", value=False)


def build_view(df_raw, criteria_idx):
    start = 3 + criteria_idx * BLOCK_SIZE
    end = start + BLOCK_SIZE

    df_view = df_raw.iloc[:, :3].copy()
    df_view.columns = BASE_COLS

    block = df_raw.iloc[:, start:end].copy()
    block.columns = BLOCK_COLS

    df_view = pd.concat([df_view, block], axis=1)
    for c in NUM_COLS:
        df_view[c] = pd.to_numeric(df_view[c], errors="coerce")
    return df_view


st.sidebar.divider()
st.sidebar.markdown("##### Charts")
metric = st.sidebar.selectbox(
    "Y-axis metric",
    NUM_COLS,
    index=NUM_COLS.index("pass_rate"),
    help="Used for histograms and bar charts (pass rate is 0–100).",
)

group_by = st.sidebar.selectbox(
    "Color / group by",
    ["GT_OBJ", "Option"],
)

st.sidebar.divider()
abs_gates_enabled = st.sidebar.checkbox(
    "Enable absolute pass/fail gates",
    value=False,
    help="Count scenarios that pass/fail fixed thresholds (pass rate 0–100; optional 2nd metric).",
)
with st.sidebar.expander("Absolute pass/fail gates", expanded=abs_gates_enabled):
    _label_runs = [runA]
    if mode == "Compare Mode" and runB:
        _label_runs.append(runB)
    perception_labels_gate = _perception_label_options_from_runs(*_label_runs)
    if not perception_labels_gate:
        st.caption("No perception labels in Summary.csv — gate filter unavailable for this run.")
        abs_gate_perception_labels: list = []
    else:
        prev_pg = st.session_state.get("selected_perception_labels", perception_labels_gate)
        safe_pg = _safe_default(prev_pg, perception_labels_gate)
        st.session_state["selected_perception_labels"] = safe_pg
        abs_gate_perception_labels = st.multiselect(
            "Perception Label Filter",
            perception_labels_gate,
            default=safe_pg,
            key="criteria_abs_gates_perception_filter",
            help=(
                "Limit gate evaluation to scenarios whose perception label appears in Summary.csv "
                "(same idea as Overview). Clear all selections to include every scenario."
            ),
        )
        st.session_state["selected_perception_labels"] = abs_gate_perception_labels

    abs_pass_min = st.number_input(
        "Minimum pass rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=95.0,
        step=0.1,
        help=(
            "Scenario pass rate is from Score.csv (same scale as lsim / eval_summary: 0–100). "
            "If a scenario has multiple table rows, the **mean** pass_rate across those rows is used."
        ),
    )
    abs_use_metric2 = st.checkbox("Second condition (numeric metric)", value=False)
    abs_metric2_col = st.selectbox(
        "Metric column",
        NUM_COLS,
        index=NUM_COLS.index("nm") if "nm" in NUM_COLS else 0,
        disabled=not abs_use_metric2,
    )
    abs_metric2_op = st.selectbox(
        "Operator",
        ["<=", ">="],
        index=0,
        disabled=not abs_use_metric2,
    )
    abs_metric2_threshold = st.number_input(
        "Metric threshold",
        value=0.0,
        format="%.6f",
        disabled=not abs_use_metric2,
    )

render_page_hero(
    kicker="Criteria-based evaluation",
    title="Score & pass-rate insight",
    description=(
        "Distributions, grouped averages, scenario-level comparisons, and optional absolute gates "
        "for a clear pass/fail sign-off."
    ),
    mode=mode,
    secondary_badge_inner_html=(
        f"Active block · <strong>criteria{criteria_idx}</strong> ({CRITERIA_COUNT} available)"
    ),
)

with st.expander("How to use this page", expanded=False):
    st.markdown(
        """
        1. **Pick a criteria block** in the sidebar — each block matches one criteria index from your evaluation pipeline.
        2. **Charts** show how the chosen metric spreads and how it differs by `GT_OBJ` or `Option`.
        3. In **Compare** mode, use the **Overlay / Delta** tabs to see distributions side-by-side or row-level **B − A** changes.
        4. Enable **Absolute pass/fail gates** when you need a binary, scenario-level verdict (e.g. min pass rate %) for sign-off.
        """
    )


def _df_for_absolute_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Columns needed for gating; drop Streamlit helper columns like Run."""
    use = [c for c in BASE_COLS + NUM_COLS if c in df.columns]
    return df.loc[:, use].copy()


def _gate_verdict_banner_html(summ: dict, run_label: str) -> str:
    """Large HTML banner: final gate verdict for one run."""
    rl = html.escape(str(run_label))
    n = summ["n_scenarios"]
    if n == 0:
        return (
            '<div style="background: linear-gradient(135deg, #64748b 0%, #94a3b8 100%); color: white; '
            "padding: 1.1rem 1.25rem; border-radius: 14px; text-align: center; margin-bottom: 0.75rem; "
            'box-shadow: 0 4px 14px rgba(0,0,0,0.12);">'
            f'<div style="font-size: 0.7rem; letter-spacing: 0.2em; opacity: 0.9;">{rl} · GATE VERDICT</div>'
            '<div style="font-size: 1.6rem; font-weight: 800; margin: 0.35rem 0;">NO DATA</div>'
            '<div style="font-size: 0.85rem; opacity: 0.92;">No scenarios to evaluate</div></div>'
        )
    if summ["all_pass"]:
        return (
            '<div style="background: linear-gradient(135deg, #047857 0%, #10b981 55%, #34d399 100%); color: white; '
            "padding: 1.25rem 1.5rem; border-radius: 14px; text-align: center; margin-bottom: 0.75rem; "
            'box-shadow: 0 6px 20px rgba(16,185,129,0.35); border: 2px solid rgba(255,255,255,0.25);">'
            f'<div style="font-size: 0.72rem; letter-spacing: 0.18em; opacity: 0.92;">{rl} · FINAL GATE</div>'
            '<div style="font-size: 2.35rem; font-weight: 900; margin: 0.2rem 0; line-height: 1.1; text-shadow: 0 2px 8px rgba(0,0,0,0.15);">'
            "PASS</div>"
            f'<div style="font-size: 1rem; font-weight: 600; opacity: 0.95;">All {n:,} scenario(s) meet your thresholds</div>'
            '<div style="font-size: 0.8rem; opacity: 0.88; margin-top: 0.35rem;">Ready as a release-style checkpoint</div></div>'
        )
    nf = summ["n_fail"]
    return (
        '<div style="background: linear-gradient(135deg, #991b1b 0%, #dc2626 50%, #f87171 100%); color: white; '
        "padding: 1.25rem 1.5rem; border-radius: 14px; text-align: center; margin-bottom: 0.75rem; "
        'box-shadow: 0 6px 20px rgba(220,38,38,0.35); border: 2px solid rgba(255,255,255,0.2);">'
        f'<div style="font-size: 0.72rem; letter-spacing: 0.18em; opacity: 0.92;">{rl} · FINAL GATE</div>'
        '<div style="font-size: 2.35rem; font-weight: 900; margin: 0.2rem 0; line-height: 1.1; text-shadow: 0 2px 8px rgba(0,0,0,0.15);">'
        "FAIL</div>"
        f'<div style="font-size: 1rem; font-weight: 600; opacity: 0.95;">{nf:,} of {n:,} scenario(s) below threshold</div>'
        '<div style="font-size: 0.8rem; opacity: 0.88; margin-top: 0.35rem;">Review failing scenarios below</div></div>'
    )


def _gate_verdict_donut_fig(summ: dict) -> go.Figure:
    """Donut chart Pass vs Fail — strong visual share."""
    n = summ["n_scenarios"]
    npass = summ["n_pass"]
    nfail = summ["n_fail"]
    pct = summ["pass_pct"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Pass", "Fail"],
                values=[npass, nfail],
                hole=0.58,
                marker=dict(colors=["#22c55e", "#ef4444"], line=dict(color="#ffffff", width=2)),
                textinfo="value",
                textposition="outside",
                textfont=dict(size=15),
                hovertemplate="<b>%{label}</b><br>Scenarios: %{value}<br>%{percent}<extra></extra>",
            )
        ]
    )
    center = f"<b>{pct:.1f}%</b><br><span style='font-size:0.65em;font-weight:normal'>pass</span>"
    if n == 0:
        center = "—"
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
        margin=dict(t=30, b=40, l=24, r=24),
        height=300,
        annotations=[
            dict(text=center, x=0.5, y=0.5, font_size=22, showarrow=False, font_color="#0f172a")
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_absolute_gates_section(
    runs: list,
    *,
    criteria_idx: int,
    criteria_count: int,
):
    """runs: list of (label, df_view) — df may include Run; it is stripped."""
    if not abs_gates_enabled:
        return

    spec = None
    if abs_use_metric2:
        op = "<=" if abs_metric2_op == "<=" else ">="
        spec = MetricGateSpec(abs_metric2_col, op, float(abs_metric2_threshold))

    st.markdown(
        '<hr style="border:none;height:2px;background:linear-gradient(90deg,transparent,#94a3b1,#0d9488,#94a3b1,transparent);margin:2rem 0 1.25rem 0;border-radius:2px;"/>',
        unsafe_allow_html=True,
    )
    section_header(
        "Final gate verdict",
        "Pass rate **0–100** (Score.csv / lsim). Thresholds from the sidebar — **last checkpoint** before sign-off.",
    )

    gate_results = []
    cols = st.columns(len(runs))
    for i, (label, dfv) in enumerate(runs):
        with cols[i]:
            try:
                result = evaluate_scenario_gates(
                    _df_for_absolute_gates(dfv),
                    float(abs_pass_min),
                    spec,
                )
            except Exception as e:
                st.error(f"Gate evaluation failed: {e}")
                continue
            summ = gate_summary(result)
            st.markdown(_gate_verdict_banner_html(summ, label), unsafe_allow_html=True)

            if summ["n_scenarios"] > 0:
                st.markdown("**Scenario pass rate (bar)**")
                pct_frac = min(1.0, max(0.0, summ["pass_pct"] / 100.0))
                st.progress(pct_frac)
                st.caption(
                    f"{summ['pass_pct']:.1f}% scenarios pass ({summ['n_pass']:,} / {summ['n_scenarios']:,})"
                )
                st.plotly_chart(
                    _gate_verdict_donut_fig(summ),
                    use_container_width=True,
                    key=f"gate_donut_{i}",
                    config={"displayModeBar": False},
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scenarios", f"{summ['n_scenarios']:,}")
            m2.metric("Pass", f"{summ['n_pass']:,}")
            m3.metric("Fail", f"{summ['n_fail']:,}")
            m4.metric("Pass %", f"{summ['pass_pct']:.1f}%")

            if summ["n_scenarios"] > 0:
                if summ["all_pass"]:
                    st.success("**Gate cleared** — every scenario satisfies the configured thresholds.")
                else:
                    st.error(
                        f"**Gate not cleared** — {summ['n_fail']:,} scenario(s) still outside thresholds."
                    )

            fails = failing_scenarios_table(result)
            if not fails.empty:
                st.markdown(
                    f'<p style="color:#b91c1c;font-weight:700;font-size:1rem;margin:0.75rem 0 0.35rem 0;">'
                    f"Failing scenarios ({len(fails):,})</p>",
                    unsafe_allow_html=True,
                )
                st.dataframe(fails, width="stretch")
            gate_results.append((label, result, spec))

    if len(gate_results) == 1:
        label, result, sp = gate_results[0]
        exp = export_gate_result(result, sp)
        exp.insert(0, "run", label)
        st.download_button(
            "Download gate results (CSV)",
            exp.to_csv(index=False).encode("utf-8"),
            file_name="criteria_absolute_gates.csv",
            mime="text/csv",
            key="dl_abs_gates_single",
            type="primary",
        )
    elif len(gate_results) > 1:
        parts = []
        for label, result, sp in gate_results:
            exp = export_gate_result(result, sp)
            exp.insert(0, "run", label)
            parts.append(exp)
        combined = pd.concat(parts, ignore_index=True)
        st.download_button(
            "Download gate results (CSV)",
            combined.to_csv(index=False).encode("utf-8"),
            file_name="criteria_absolute_gates_compare.csv",
            mime="text/csv",
            key="dl_abs_gates_compare",
            type="primary",
        )


if mode == "Compare Mode":
    df_view_A = build_view(df_raw_A, criteria_idx)
    df_view_B = build_view(df_raw_B, criteria_idx)

    df_view_A["Run"] = "Baseline (A)"
    df_view_B["Run"] = "Candidate (B)"
    df_compare = pd.concat([df_view_A, df_view_B], axis=0, ignore_index=True)

    if show_debug:
        st.subheader("Raw Data Check — Baseline (A)")
        st.dataframe(df_raw_A.head(10), width="stretch")
        st.subheader("Raw Data Check — Candidate (B)")
        st.dataframe(df_raw_B.head(10), width="stretch")
        st.subheader(f"Criteria {criteria_idx} Data — Baseline (A)")
        st.dataframe(df_view_A, width="stretch")
        st.subheader(f"Criteria {criteria_idx} Data — Candidate (B)")
        st.dataframe(df_view_B, width="stretch")

    section_header("Run summary", "Rows loaded and **mean pass rate** (practical pass %, 0–100) per run.")
    count_a = len(df_view_A)
    count_b = len(df_view_B)
    mean_a = df_view_A["pass_rate"].mean() if count_a else 0.0
    mean_b = df_view_B["pass_rate"].mean() if count_b else 0.0
    delta_mean = mean_b - mean_a
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (A)", f"{count_a:,}")
    col2.metric("Rows (B)", f"{count_b:,}")
    col3.metric("Pass rate mean (A)", f"{mean_a:.3f}")
    col4.metric("Pass rate mean (B)", f"{mean_b:.3f}", f"{delta_mean:+.3f}")

    _render_absolute_gates_section(
        [
            (
                "Baseline (A)",
                _filter_df_view_by_perception_labels(
                    df_view_A, runA.get("summary"), abs_gate_perception_labels
                ),
            ),
            (
                "Candidate (B)",
                _filter_df_view_by_perception_labels(
                    df_view_B, runB.get("summary"), abs_gate_perception_labels
                ),
            ),
        ],
        criteria_idx=criteria_idx,
        criteria_count=CRITERIA_COUNT,
    )

    st.markdown(
        '<p style="margin:0.5rem 0 0.35rem 0;font-size:1.05rem;font-weight:800;color:#0f172a;">Compare workspace</p>'
        '<p style="margin:0 0 0.6rem 0;color:#64748b;font-size:0.9rem;">'
        "<strong>Overlay</strong> — A and B on the same charts · "
        "<strong>Delta</strong> — row-wise <strong>B − A</strong> after matching keys.</p>",
        unsafe_allow_html=True,
    )
    tab_ov, tab_dl = st.tabs(["Overlay: A vs B", "Delta: B − A"])

    with tab_ov:
        section_header(f"{metric} distribution", "Semi-transparent overlay — Baseline (indigo) vs Candidate (teal).")
        fig = px.histogram(
            df_compare,
            x=metric,
            color="Run",
            color_discrete_map=_PX_COLOR_RUN,
            nbins=30,
            barmode="overlay",
            opacity=0.55,
            marginal="box",
        )
        _plotly_apply_theme(fig, f"{metric} · row-level distribution")
        st.plotly_chart(fig, width="stretch")

        section_header(f"Average {metric} by {group_by}", "Grouped means — compare runs side-by-side.")
        df_avg = (
            df_compare
            .groupby([group_by, "Run"], as_index=False)[metric]
            .mean()
            .sort_values(metric, ascending=False)
        )
        fig = px.bar(
            df_avg,
            x=group_by,
            y=metric,
            color="Run",
            color_discrete_map=_PX_COLOR_RUN,
            barmode="group",
            text_auto=".2f",
        )
        _plotly_apply_theme(fig, f"Mean {metric} by {group_by}")
        st.plotly_chart(fig, width="stretch")

        section_header("Pass rate by group", "Box + points — useful for spotting spread and outliers (0–100 scale).")
        fig = px.box(
            df_compare,
            x=group_by,
            y="pass_rate",
            color="Run",
            color_discrete_map=_PX_COLOR_RUN,
            points="all",
        )
        _plotly_apply_theme(fig, "Pass rate overview")
        st.plotly_chart(fig, width="stretch")

        # ---- Improved Per Scenario Pass Rate Compare Plot ----
        section_header("Per-scenario pass rate", "Same scenario on both runs — filter to focus on regressions or wins.")

        per_scenario_A = df_view_A.groupby("Scenario", as_index=False)["pass_rate"].mean()
        per_scenario_B = df_view_B.groupby("Scenario", as_index=False)["pass_rate"].mean()
        per_scenario = per_scenario_A.merge(per_scenario_B, on="Scenario", suffixes=("_A", "_B"))
        per_scenario["delta"] = per_scenario["pass_rate_B"] - per_scenario["pass_rate_A"]

        # Let user filter/sort the scenario comparison by various methods to make it easier to see
        filter_method = st.radio(
            "Scenario Filter/Sort", 
            ["All", "Top N by Delta", "Top N by Baseline (A)", "Custom contains string"],
            horizontal=True,
        )

        if filter_method == "Top N by Delta":
            N = st.number_input("Show Top N Scenarios by Delta (|delta|):", min_value=5, max_value=100, value=20)
            per_scenario = per_scenario.reindex(per_scenario["delta"].abs().sort_values(ascending=False).index)
            per_scenario_vis = per_scenario.head(N)
        elif filter_method == "Top N by Baseline (A)":
            N = st.number_input("Show Top N Scenarios by Baseline (A) Pass Rate:", min_value=5, max_value=100, value=20)
            per_scenario = per_scenario.sort_values("pass_rate_A", ascending=False)
            per_scenario_vis = per_scenario.head(N)
        elif filter_method == "Custom contains string":
            search = st.text_input("Show scenarios with name containing (case-insensitive):", "")
            per_scenario_vis = per_scenario[per_scenario["Scenario"].str.contains(search, case=False, na=False)] if search else per_scenario
        else:
            per_scenario_vis = per_scenario.copy()

        # Use bar plot instead of scatter, for more useful comparison if there are many scenarios
        per_scenario_vis_long = pd.melt(
            per_scenario_vis,
            id_vars=["Scenario"],
            value_vars=["pass_rate_A", "pass_rate_B"],
            var_name="Run",
            value_name="pass_rate",
        )
        per_scenario_vis_long["Run"] = per_scenario_vis_long["Run"].map({"pass_rate_A": "Baseline (A)", "pass_rate_B": "Candidate (B)"})
        per_scenario_vis_long = per_scenario_vis_long.sort_values(["Scenario", "Run"])

        fig = px.bar(
            per_scenario_vis_long,
            x="Scenario",
            y="pass_rate",
            color="Run",
            color_discrete_map=_PX_COLOR_RUN,
            barmode="group",
            text_auto=".2f",
        )
        _plotly_apply_theme(fig, "Per-scenario pass rate (filtered)")
        st.plotly_chart(fig, width="stretch")

        # Show the delta for each scenario as a separate barplot below, sorted by delta
        section_header("Per-scenario delta (B − A)", "Green = improvement on B; red = regression vs baseline.")
        fig2 = px.bar(
            per_scenario_vis.sort_values("delta", key=abs, ascending=False),
            x="Scenario",
            y="delta",
            color="delta",
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
        )
        _plotly_apply_theme(fig2, "Pass rate delta by scenario")
        st.plotly_chart(fig2, width="stretch")

        # Optionally show a table for the user to inspect
        with st.expander("Show Table: Per Scenario Pass Rates and Delta"):
            # Show dataframe with delta values, but avoid styling due to colormap compatibility issue
            st.dataframe(
                per_scenario_vis[["Scenario", "pass_rate_A", "pass_rate_B", "delta"]],
                width="stretch"
            )

        # The user can still optionally "restore" the scatter plot if they want
        if st.checkbox("Show scatter plot (Baseline (A) Pass Rate vs Candidate (B))", value=False):
            scatter_fig = px.scatter(
                per_scenario_vis,
                x="pass_rate_A",
                y="pass_rate_B",
                text="Scenario",
                labels={
                    "pass_rate_A": "Baseline (A) Pass Rate",
                    "pass_rate_B": "Candidate (B) Pass Rate",
                },
                title="Per Scenario Pass Rate: Baseline (A) vs Candidate (B) (filtered)",
            )
            lim = float(
                max(
                    per_scenario_vis["pass_rate_A"].max(),
                    per_scenario_vis["pass_rate_B"].max(),
                    100.0,
                )
            )
            scatter_fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=lim,
                y1=lim,
                line=dict(dash="dash", color="rgba(100,116,139,0.8)", width=2),
                xref="x",
                yref="y",
            )
            scatter_fig.update_xaxes(range=[0, lim])
            scatter_fig.update_yaxes(range=[0, lim])
            scatter_fig.update_traces(textposition="top center", marker=dict(size=10, line=dict(width=0.5, color="white")))
            _plotly_apply_theme(scatter_fig, "A vs B pass rate (parity line = equal)")
            st.plotly_chart(scatter_fig, width="stretch")
        # ---- End Improved Per Scenario Pass Rate Compare Plot ----

        st.download_button(
            "Download overlay data (CSV)",
            df_compare.to_csv(index=False).encode("utf-8"),
            file_name="criteria_compare_filtered.csv",
            mime="text/csv",
            type="primary",
        )

    with tab_dl:
        merged = df_view_A.merge(
            df_view_B,
            on=BASE_COLS,
            suffixes=("_A", "_B"),
            how="inner",
        )
        merged[f"{metric}_delta"] = merged[f"{metric}_B"] - merged[f"{metric}_A"]
        merged["pass_rate_delta"] = merged["pass_rate_B"] - merged["pass_rate_A"]

        section_header(f"{metric} delta distribution", "How much B differs from A on the same row (inner join on Scenario/Option/GT_OBJ).")
        fig = px.histogram(
            merged,
            x=f"{metric}_delta",
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#0d9488"],
        )
        _plotly_apply_theme(fig, f"Δ {metric} (B − A)")
        st.plotly_chart(fig, width="stretch")

        section_header(f"Mean Δ {metric} by {group_by}", "Where the shift concentrates across labels.")
        df_delta = (
            merged
            .groupby(group_by, as_index=False)[f"{metric}_delta"]
            .mean()
            .sort_values(f"{metric}_delta", ascending=False)
        )
        fig = px.bar(
            df_delta,
            x=group_by,
            y=f"{metric}_delta",
            text_auto=".2f",
            color_discrete_sequence=["#312e81"],
        )
        _plotly_apply_theme(fig, f"Grouped mean · Δ {metric}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        section_header("Pass rate delta overview", "Distribution of per-row pass rate change.")
        fig = px.box(
            merged,
            x=group_by,
            y="pass_rate_delta",
            points="all",
            color_discrete_sequence=["#0369a1"],
        )
        _plotly_apply_theme(fig, "Δ pass rate by group")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        section_header("Largest changes", "Top 20 rows by |Δ metric| — inspect regressions or improvements.")
        top_changes = merged.copy()
        top_changes["abs_delta"] = top_changes[f"{metric}_delta"].abs()
        top_changes = top_changes.sort_values("abs_delta", ascending=False).head(20)
        cols = BASE_COLS + [f"{metric}_A", f"{metric}_B", f"{metric}_delta"]
        if metric != "pass_rate":
            cols.append("pass_rate_delta")
        st.dataframe(
            top_changes[
                cols
            ],
            width="stretch",
        )

else:
    df_view = build_view(df_raw_A, criteria_idx)
    if show_debug:
        st.subheader("Raw Data Check — Single Mode")
        st.dataframe(df_raw_A.head(10), width="stretch")
        st.subheader(f"Criteria {criteria_idx} Data")
        st.dataframe(df_view, width="stretch")

    section_header("Run summary", "Dataset size and central pass-rate tendency (0–100 scale).")
    count = len(df_view)
    mean_pass = df_view["pass_rate"].mean() if count else 0.0
    median_pass = df_view["pass_rate"].median() if count else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{count:,}")
    col2.metric("Pass rate mean", f"{mean_pass:.3f}")
    col3.metric("Pass rate median", f"{median_pass:.3f}")

    _render_absolute_gates_section(
        [
            (
                "Current run",
                _filter_df_view_by_perception_labels(
                    df_view, runA.get("summary"), abs_gate_perception_labels
                ),
            ),
        ],
        criteria_idx=criteria_idx,
        criteria_count=CRITERIA_COUNT,
    )

    section_header(f"{metric} distribution", f"How **{metric}** spreads across all rows; colored by **{group_by}**.")
    fig = px.histogram(
        df_view,
        x=metric,
        color=group_by,
        nbins=30,
        marginal="box",
        color_discrete_sequence=_PX_COLOR_QUAL,
    )
    _plotly_apply_theme(fig, f"{metric} · histogram")
    st.plotly_chart(fig, width="stretch")

    section_header(f"Mean {metric} by {group_by}", "Ranked bars — quick read on which labels drive the metric.")
    df_avg = (
        df_view
        .groupby(group_by, as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )
    fig = px.bar(
        df_avg,
        x=group_by,
        y=metric,
        text_auto=".2f",
        color=group_by,
        color_discrete_sequence=_PX_COLOR_QUAL,
    )
    _plotly_apply_theme(fig, f"Mean {metric}")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    section_header("Pass rate overview", "Spread and outliers of pass rate (0–100) within each group.")
    fig = px.box(
        df_view,
        x=group_by,
        y="pass_rate",
        points="all",
        color=group_by,
        color_discrete_sequence=_PX_COLOR_QUAL,
    )
    _plotly_apply_theme(fig, "Pass rate by group")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    section_header("Scenario leaderboard", "Mean pass rate per scenario — tune N and sort direction.")
    scenario_metric = df_view.groupby("Scenario", as_index=False)["pass_rate"].mean()
    top_n = st.number_input("Top N scenarios", min_value=5, max_value=100, value=20, key="single_top_n")
    sort_order = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True, key="single_scen_order")
    scenario_metric = scenario_metric.sort_values(
        "pass_rate",
        ascending=sort_order == "Lowest first",
    ).head(int(top_n))
    st.dataframe(scenario_metric, width="stretch")
