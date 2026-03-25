"""Criteria-based score page: HTML banners and static callouts."""

from __future__ import annotations

import html
from typing import List


def gate_verdict_banner_html(summ: dict, run_label: str) -> str:
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


def hover_html_for_scenarios(title: str, scenarios: List[str], *, max_lines: int = 45) -> str:
    """Plotly hover HTML: title + count + newline-separated scenario names."""
    n = len(scenarios)
    esc_title = html.escape(title)
    if n == 0:
        return f"<b>{esc_title}</b><br><i>No scenarios</i>"
    head = scenarios[:max_lines]
    lines = "<br>".join(html.escape(s) for s in head)
    more = ""
    if n > max_lines:
        more = f"<br><i>… +{n - max_lines:,} more (see table below)</i>"
    return f"<b>{esc_title}</b> · {n:,} scenario{'s' if n != 1 else ''}<br>{lines}{more}"


def criteria_absolute_gate_section_hr_markup() -> str:
    """Decorative rule above Final gate verdict."""
    return (
        '<hr style="border:none;height:2px;background:linear-gradient(90deg,transparent,#94a3b1,#0d9488,#94a3b1,transparent);'
        'margin:2rem 0 1.25rem 0;border-radius:2px;"/>'
    )


def gate_compare_overlap_intro_markup() -> str:
    """Title + caption for scenario overlap compare block."""
    return (
        '<p style="font-size:1.12rem;font-weight:800;color:#0f172a;margin:1.5rem 0 0.25rem 0;">'
        "Compare · scenario overlap</p>"
        '<p style="color:#64748b;font-size:0.9rem;margin:0 0 0.85rem 0;">'
        "Same scenario IDs in both runs: <strong>who fails where</strong> — recovered, regressed, stable pass, or still failing.</p>"
    )


def criteria_failing_scenarios_heading_markup(count: int) -> str:
    """Red heading above failing-scenarios table."""
    c = int(count)
    return (
        f'<p style="color:#b91c1c;font-weight:700;font-size:1rem;margin:0.75rem 0 0.35rem 0;">'
        f"Failing scenarios ({c:,})</p>"
    )


def criteria_compare_workspace_intro_markup() -> str:
    """Compare workspace tab strip intro."""
    return (
        '<p style="margin:0.5rem 0 0.35rem 0;font-size:1.05rem;font-weight:800;color:#0f172a;">Compare workspace</p>'
        '<p style="margin:0 0 0.6rem 0;color:#64748b;font-size:0.9rem;">'
        "<strong>Overlay</strong> — all selected runs · "
        "<strong>Delta</strong> — row-wise chosen candidate minus baseline after matching keys.</p>"
    )
