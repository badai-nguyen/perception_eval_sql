import html
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.path_utils import path_display
from lib.criteria_absolute_gates import (
    MetricGateSpec,
    evaluate_scenario_gates,
    export_gate_result,
    failing_scenarios_table,
    gate_summary,
    infer_criteria_count,
)

st.set_page_config(layout="wide")
st.title("Criteria-based Evaluation Viewer")

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

st.subheader("Loaded Runs")
st.markdown(f"**Baseline (A):** `{path_display(runA['path'])}`")
if mode == "Compare Mode":
    st.markdown(f"**Candidate (B):** `{path_display(runB['path'])}`")
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

criteria_idx = st.sidebar.selectbox(
    "Select Criteria",
    list(range(CRITERIA_COUNT)),
    format_func=lambda x: f"criteria{x}",
)

show_debug = st.sidebar.checkbox("Show debug info", value=False)


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


st.sidebar.subheader("Visualization Controls")
metric = st.sidebar.selectbox(
    "Metric",
    NUM_COLS,
    index=NUM_COLS.index("pass_rate"),
)

group_by = st.sidebar.selectbox(
    "Group by",
    ["GT_OBJ", "Option"],
)

with st.sidebar.expander("Absolute pass/fail gates", expanded=False):
    abs_gates_enabled = st.checkbox(
        "Enable scenario-level gates",
        value=False,
        help="Count scenarios that pass/fail fixed thresholds (pass rate 0–100; optional 2nd metric).",
    )
    abs_pass_min = st.number_input(
        "Minimum pass rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=95.0,
        step=0.1,
        help="Scenario pass rate is from Score.csv (same scale as lsim / eval_summary: 0–100).",
    )
    abs_agg_mode = st.radio(
        "Scenario aggregation",
        ["mean", "all_rows"],
        index=0,
        format_func=lambda x: (
            "Mean pass rate (per scenario)" if x == "mean" else "All rows must pass"
        ),
        help="Mean: use mean pass_rate per scenario; 2nd metric uses max (for ≤) or min (for ≥) across rows. "
        "All rows: every Option×GT_OBJ row must satisfy both gates.",
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


def _render_absolute_gates_section(runs: list):
    """runs: list of (label, df_view) — df may include Run; it is stripped."""
    if not abs_gates_enabled:
        return

    spec = None
    if abs_use_metric2:
        op = "<=" if abs_metric2_op == "<=" else ">="
        spec = MetricGateSpec(abs_metric2_col, op, float(abs_metric2_threshold))

    st.markdown("---")
    st.subheader("Final gate verdict (scenario level)")
    st.caption(
        "Pass rate **0–100** (Score.csv / lsim). Thresholds from the sidebar. "
        "This block is meant as the **last checkpoint** before sign-off."
    )

    gate_results = []
    cols = st.columns(len(runs))
    for i, (label, dfv) in enumerate(runs):
        with cols[i]:
            try:
                result = evaluate_scenario_gates(
                    _df_for_absolute_gates(dfv),
                    float(abs_pass_min),
                    abs_agg_mode,
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
            "Download per-scenario gate results (CSV)",
            exp.to_csv(index=False).encode("utf-8"),
            file_name="criteria_absolute_gates.csv",
            mime="text/csv",
            key="dl_abs_gates_single",
        )
    elif len(gate_results) > 1:
        parts = []
        for label, result, sp in gate_results:
            exp = export_gate_result(result, sp)
            exp.insert(0, "run", label)
            parts.append(exp)
        combined = pd.concat(parts, ignore_index=True)
        st.download_button(
            "Download per-scenario gate results (CSV)",
            combined.to_csv(index=False).encode("utf-8"),
            file_name="criteria_absolute_gates_compare.csv",
            mime="text/csv",
            key="dl_abs_gates_compare",
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

    st.subheader("Summary")
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
            ("Baseline (A)", df_view_A),
            ("Candidate (B)", df_view_B),
        ]
    )

    compare_view = st.sidebar.radio(
        "Compare View",
        ["Overlay", "Delta"],
        horizontal=True,
    )

    if compare_view == "Overlay":
        st.subheader(f"{metric} Distribution (A vs B)")
        fig = px.histogram(
            df_compare,
            x=metric,
            color="Run",
            nbins=30,
            barmode="overlay",
            opacity=0.55,
            marginal="box",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader(f"Average {metric} by {group_by} (A vs B)")
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
            barmode="group",
            text_auto=".2f",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pass Rate Overview (A vs B)")
        fig = px.box(
            df_compare,
            x=group_by,
            y="pass_rate",
            color="Run",
            points="all",
        )
        st.plotly_chart(fig, width="stretch")

        # ---- Improved Per Scenario Pass Rate Compare Plot ----
        st.subheader("Per Scenario Pass Rate Comparison (A vs B)")

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
            barmode="group",
            text_auto=".2f",
            title="Per Scenario Pass Rate by Run (filtered)",
        )
        st.plotly_chart(fig, width="stretch")

        # Show the delta for each scenario as a separate barplot below, sorted by delta
        st.subheader("Per Scenario Pass Rate Delta (B - A)")
        fig2 = px.bar(
            per_scenario_vis.sort_values("delta", key=abs, ascending=False),
            x="Scenario",
            y="delta",
            color="delta",
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
            title="Delta (B - A)"
        )
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
            scatter_fig.add_shape(
                type="line",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(dash='dash', color='gray'),
                xref="x", yref="y"
            )
            scatter_fig.update_traces(textposition="top center")
            st.plotly_chart(scatter_fig, width="stretch")
        # ---- End Improved Per Scenario Pass Rate Compare Plot ----

        st.download_button(
            "Download filtered data as CSV",
            df_compare.to_csv(index=False).encode("utf-8"),
            file_name="criteria_compare_filtered.csv",
            mime="text/csv",
        )

    else:
        merged = df_view_A.merge(
            df_view_B,
            on=BASE_COLS,
            suffixes=("_A", "_B"),
            how="inner",
        )
        merged[f"{metric}_delta"] = merged[f"{metric}_B"] - merged[f"{metric}_A"]
        merged["pass_rate_delta"] = merged["pass_rate_B"] - merged["pass_rate_A"]

        st.subheader(f"{metric} Delta Distribution (B - A)")
        fig = px.histogram(
            merged,
            x=f"{metric}_delta",
            nbins=30,
            marginal="box",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader(f"Average {metric} Delta by {group_by}")
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
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Pass Rate Delta Overview")
        fig = px.box(
            merged,
            x=group_by,
            y="pass_rate_delta",
            points="all",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Largest Absolute Deltas")
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

    st.subheader("Summary")
    count = len(df_view)
    mean_pass = df_view["pass_rate"].mean() if count else 0.0
    median_pass = df_view["pass_rate"].median() if count else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{count:,}")
    col2.metric("Pass rate mean", f"{mean_pass:.3f}")
    col3.metric("Pass rate median", f"{median_pass:.3f}")

    _render_absolute_gates_section([("Current run", df_view)])

    st.subheader(f"{metric} Distribution")
    fig = px.histogram(
        df_view,
        x=metric,
        color=group_by,
        nbins=30,
        marginal="box",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"Average {metric} by {group_by}")
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
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Pass Rate Overview")
    fig = px.box(
        df_view,
        x=group_by,
        y="pass_rate",
        points="all",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top / Bottom Scenarios by Pass Rate")
    scenario_metric = df_view.groupby("Scenario", as_index=False)["pass_rate"].mean()
    top_n = st.number_input("Top N scenarios", min_value=5, max_value=100, value=20, key="single_top_n")
    sort_order = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True, key="single_scen_order")
    scenario_metric = scenario_metric.sort_values(
        "pass_rate",
        ascending=sort_order == "Lowest first",
    ).head(int(top_n))
    st.dataframe(scenario_metric, width="stretch")
