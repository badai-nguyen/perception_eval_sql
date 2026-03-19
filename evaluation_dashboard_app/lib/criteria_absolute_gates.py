"""
Scenario-level absolute pass/fail gates for criteria Score.csv views.

Pass rate is on the 0–100 scale (see lib/eval_summary.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import pandas as pd

AggMode = Literal["mean", "all_rows"]
MetricOp = Literal["<=", ">="]

MAX_CRITERIA_DEFAULT = 32


def infer_criteria_count(
    df_raw: pd.DataFrame,
    block_size: int,
    max_criteria: int = MAX_CRITERIA_DEFAULT,
) -> int:
    """
    Number of criteria blocks in a raw Score dataframe (first 3 cols are base).
    """
    if df_raw is None or df_raw.shape[1] < 3:
        return 1
    n = (df_raw.shape[1] - 3) // block_size
    n = max(1, n)
    return int(min(n, max_criteria))


@dataclass(frozen=True)
class MetricGateSpec:
    column: str
    op: MetricOp
    threshold: float


def evaluate_scenario_gates(
    df_view: pd.DataFrame,
    pass_min: float,
    agg_mode: AggMode,
    metric_gate: Optional[MetricGateSpec] = None,
) -> pd.DataFrame:
    """
    Per-scenario gate evaluation.

    Returns columns:
      - Scenario
      - row_count
      - agg_pass_rate (mean pass_rate over rows in scenario)
      - metric_agg (for mean mode: max/min aggregate used for gate 2; for all_rows: worst-case display)
      - scenario_pass
      - pass_rate_gate_ok
      - metric_gate_ok (True if no metric_gate)
    """
    required = {"Scenario", "pass_rate"}
    if not required.issubset(df_view.columns):
        raise ValueError(f"df_view must contain columns {required}")
    if metric_gate is not None and metric_gate.column not in df_view.columns:
        raise ValueError(f"Metric column {metric_gate.column!r} not in df_view")

    empty_cols = [
        "Scenario",
        "row_count",
        "agg_pass_rate",
        "metric_agg",
        "scenario_pass",
        "pass_rate_gate_ok",
        "metric_gate_ok",
    ]
    if df_view.empty:
        return pd.DataFrame(columns=empty_cols)

    d = df_view.copy()
    d["pass_rate"] = pd.to_numeric(d["pass_rate"], errors="coerce")
    if metric_gate is not None:
        d[metric_gate.column] = pd.to_numeric(d[metric_gate.column], errors="coerce")

    if agg_mode == "all_rows":
        pr = d["pass_rate"]
        pass_rate_row_ok = pr.notna() & (pr >= pass_min)
        if metric_gate is None:
            metric_row_ok = pd.Series(True, index=d.index)
        else:
            mv = d[metric_gate.column]
            if metric_gate.op == "<=":
                metric_row_ok = mv.notna() & (mv <= metric_gate.threshold)
            else:
                metric_row_ok = mv.notna() & (mv >= metric_gate.threshold)

        d["_pass_rate_row_ok"] = pass_rate_row_ok
        d["_metric_row_ok"] = metric_row_ok
        d["_row_ok"] = pass_rate_row_ok & metric_row_ok

        row_count = d.groupby("Scenario", observed=True).size().reset_index(name="row_count")
        agg_pass_rate = (
            d.groupby("Scenario", observed=True)["pass_rate"].mean().reset_index(name="agg_pass_rate")
        )
        scenario_pass = d.groupby("Scenario", observed=True)["_row_ok"].all().reset_index(
            name="scenario_pass"
        )
        pass_rate_gate_ok = (
            d.groupby("Scenario", observed=True)["_pass_rate_row_ok"]
            .all()
            .reset_index(name="pass_rate_gate_ok")
        )
        metric_gate_ok = (
            d.groupby("Scenario", observed=True)["_metric_row_ok"]
            .all()
            .reset_index(name="metric_gate_ok")
        )

        out = row_count.merge(agg_pass_rate, on="Scenario")
        out = out.merge(scenario_pass, on="Scenario")
        out = out.merge(pass_rate_gate_ok, on="Scenario")
        out = out.merge(metric_gate_ok, on="Scenario")

        if metric_gate is None:
            out["metric_agg"] = float("nan")
        else:
            if metric_gate.op == "<=":
                mag = d.groupby("Scenario", observed=True)[metric_gate.column].max()
            else:
                mag = d.groupby("Scenario", observed=True)[metric_gate.column].min()
            out = out.merge(mag.reset_index(name="metric_agg"), on="Scenario")

        out = out[empty_cols]
        for c in ("scenario_pass", "pass_rate_gate_ok", "metric_gate_ok"):
            out[c] = pd.Series([bool(x) for x in out[c].tolist()], dtype=object, index=out.index)
        return out

    # --- mean aggregation mode ---
    rows = []
    for scen, grp in d.groupby("Scenario", observed=True):
        rc = len(grp)
        mean_pr = float(grp["pass_rate"].mean())
        pr_ok = bool(not pd.isna(mean_pr) and mean_pr >= pass_min)

        if metric_gate is None:
            m_agg = float("nan")
            m_ok = True
        else:
            col = grp[metric_gate.column]
            if col.isna().all():
                m_agg = float("nan")
                m_ok = False
            elif metric_gate.op == "<=":
                m_agg = float(col.max())
                m_ok = bool(not pd.isna(m_agg) and m_agg <= metric_gate.threshold)
            else:
                m_agg = float(col.min())
                m_ok = bool(not pd.isna(m_agg) and m_agg >= metric_gate.threshold)

        rows.append(
            {
                "Scenario": scen,
                "row_count": rc,
                "agg_pass_rate": mean_pr,
                "metric_agg": m_agg,
                "pass_rate_gate_ok": pr_ok,
                "metric_gate_ok": m_ok,
                "scenario_pass": bool(pr_ok and m_ok),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=empty_cols)
    out = out[empty_cols]
    for c in ("scenario_pass", "pass_rate_gate_ok", "metric_gate_ok"):
        out[c] = pd.Series([bool(x) for x in out[c].tolist()], dtype=object, index=out.index)
    return out


def gate_summary(result: pd.DataFrame) -> Dict[str, Any]:
    """Counts and fractions from evaluate_scenario_gates output."""
    if result.empty:
        return {
            "n_scenarios": 0,
            "n_pass": 0,
            "n_fail": 0,
            "pass_pct": 0.0,
            "all_pass": True,
        }
    n = len(result)
    n_pass = int(result["scenario_pass"].sum())
    n_fail = n - n_pass
    pass_pct = 100.0 * n_pass / n if n else 0.0
    return {
        "n_scenarios": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_pct": pass_pct,
        "all_pass": n_fail == 0,
    }


def failing_scenarios_table(result: pd.DataFrame) -> pd.DataFrame:
    """Subset of result rows where scenario_pass is False."""
    if result.empty:
        return result.copy()
    mask = pd.Series([not bool(x) for x in result["scenario_pass"]], index=result.index)
    return result.loc[mask].copy()


def export_gate_result(result: pd.DataFrame, metric_gate: Optional[MetricGateSpec]) -> pd.DataFrame:
    """Copy suitable for CSV download with a clearer metric column name."""
    out = result.copy()
    if metric_gate is None:
        out = out.drop(columns=["metric_agg"], errors="ignore")
    else:
        label = f"metric_agg_{metric_gate.column}_{'max' if metric_gate.op == '<=' else 'min'}"
        out = out.rename(columns={"metric_agg": label})
    return out
