"""Download page: transfer progress HUD and hero/pipeline markdown."""

from __future__ import annotations

import html
from typing import List, Optional, Sequence, Tuple

import streamlit as st


def impressive_progress_hud_markup(
    *,
    fraction: Optional[float],
    headline: str,
    detail: str = "",
    foot: str = "",
    indeterminate: bool = False,
) -> str:
    """Build transfer HUD HTML; escape all user-facing text."""
    h = html.escape(headline)
    d_esc = html.escape(detail) if detail else ""
    f_esc = html.escape(foot) if foot else ""
    ring = '<div class="dl-xfer-ring" aria-hidden="true"></div>' if indeterminate else ""

    if indeterminate:
        pct_main = "···"
        suffix = ""
        track = '<div class="dl-xfer-fill dl-xfer-fill--indeterminate"></div>'
    else:
        fr = max(0.0, min(1.0, float(fraction if fraction is not None else 0.0)))
        pct_main = str(int(round(fr * 100)))
        suffix = '<span class="dl-xfer-pct-suffix">%</span>'
        w = f"{fr * 100:.4f}%"
        track = f'<div class="dl-xfer-fill" style="width:{w}"></div>'

    detail_block = f'<div class="dl-xfer-detail">{d_esc}</div>' if d_esc else ""
    foot_block = f'<div class="dl-xfer-foot">{f_esc}</div>' if f_esc else ""

    return f"""
        <div class="dl-xfer-hud" role="status" aria-live="polite">
          <div class="dl-xfer-scan" aria-hidden="true"></div>
          {ring}
          <div class="dl-xfer-top">
            <span class="dl-xfer-label">LIVE TRANSFER</span>
            <span class="dl-xfer-dots" aria-hidden="true"><span></span><span></span><span></span></span>
          </div>
          <div class="dl-xfer-pct-row">
            <span class="dl-xfer-pct">{pct_main}</span>{suffix}
          </div>
          <div class="dl-xfer-track">{track}</div>
          <div class="dl-xfer-headline">{h}</div>
          {detail_block}
          {foot_block}
        </div>
    """


class ImpressiveProgressHUD:
    """Streamlit placeholder that renders the transfer HUD (update via .show / .clear)."""

    __slots__ = ("_slot",)

    def __init__(self) -> None:
        self._slot = st.empty()

    def show(
        self,
        *,
        fraction: Optional[float] = None,
        headline: str = "",
        detail: str = "",
        foot: str = "",
        indeterminate: bool = False,
    ) -> None:
        self._slot.markdown(
            impressive_progress_hud_markup(
                fraction=fraction,
                headline=headline,
                detail=detail,
                foot=foot,
                indeterminate=indeterminate,
            ),
            unsafe_allow_html=True,
        )

    def clear(self) -> None:
        self._slot.empty()


def render_download_hero(*, queue_enabled: bool) -> None:
    k = html.escape("Data pipeline")
    t = html.escape("Autoware Evaluator results downloader")
    d = html.escape(
        "Pull jobs from the evaluator, fetch scenarios, run local eval hooks, build parquet — "
        "with optional worker queue integration when configured."
    )
    q_badge =  ("<!-- no secondary -->")
    if queue_enabled:
        q_badge = (
            '<span class="dl-hero-pill dl-pill-live">'
            '<span class="dl-pulse-dot" aria-hidden="true"></span> Task queue</span>'
        )
    st.markdown(
        f"""
        <div class="dl-hero-wrap">
          <div class="dl-hero-bg" aria-hidden="true"></div>
          <div class="dl-hero-grid" aria-hidden="true"></div>
          <div class="dl-hero-shine" aria-hidden="true"></div>
          <div class="dl-hero-inner">
            <div class="dl-hero-top">
              <div>
                <p class="dl-hero-kicker">{k}</p>
                <h1 class="dl-hero-title">{t}</h1>
                <p class="dl-hero-desc">{d}</p>
              </div>
              <div class="dl-hero-pills">
                <span class="dl-hero-pill">Archives · JSON · Parquet</span>
                {q_badge}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_task_section_header() -> None:
    st.markdown(
        """
        <div class="dl-section-card">
          <div class="dl-section-icon" aria-hidden="true">⚡</div>
          <div>
            <div class="dl-section-kicker">Background jobs</div>
            <div class="dl-section-title">Task status</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_download_status_table_intro(
    *,
    kicker: str = "Batch detail",
    title: str = "Download status",
    subtitle: str = "Per-row outcomes from this download pass.",
) -> None:
    """Call immediately before ``st.dataframe`` for archive/JSON download rows."""
    st.markdown(
        f"""
        <div class="dl-result-shell">
          <div class="dl-result-panel dl-result-panel--table">
            <div class="dl-result-kicker">{html.escape(kicker)}</div>
            <div class="dl-result-title">{html.escape(title)}</div>
            <p class="dl-result-sub">{html.escape(subtitle)}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_download_summary_panel(*, success: int, skipped: int, failed: int) -> None:
    """After scenario YAML downloads: success / skipped / failed tiles."""
    st.markdown(
        f"""
        <div class="dl-result-shell">
          <div class="dl-result-panel">
            <div class="dl-result-kicker">Outcome</div>
            <div class="dl-result-title">Download summary</div>
            <p class="dl-result-sub">Scenario downloads for this run.</p>
            <div class="dl-stat-grid">
              <div class="dl-stat-tile dl-stat-tile--ok">
                <span class="dl-stat-n">{int(success)}</span>
                <span class="dl-stat-l">Success</span>
              </div>
              <div class="dl-stat-tile dl-stat-tile--skip">
                <span class="dl-stat-n">{int(skipped)}</span>
                <span class="dl-stat-l">Skipped</span>
              </div>
              <div class="dl-stat-tile dl-stat-tile--fail">
                <span class="dl-stat-n">{int(failed)}</span>
                <span class="dl-stat-l">Failed</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _output_path_row(output_rel: str) -> str:
    safe = html.escape(output_rel, quote=True)
    return (
        f'<div class="dl-path-row">Output directory <code>{safe}</code></div>'
    )


def render_job_archives_summary_panel(*, scenarios_processed: int, output_rel: str) -> None:
    """Tab1 after ZIP path: total processed + path."""
    st.markdown(
        f"""
        <div class="dl-result-shell">
          <div class="dl-result-panel">
            <div class="dl-result-kicker">Run complete</div>
            <div class="dl-result-title">Summary</div>
            <p class="dl-result-sub">Archives fetched and extracted under your output path.</p>
            <div class="dl-stat-grid">
              <div class="dl-stat-tile dl-stat-tile--neutral">
                <span class="dl-stat-n">{int(scenarios_processed)}</span>
                <span class="dl-stat-l">Scenarios</span>
              </div>
            </div>
            {_output_path_row(output_rel)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_job_json_summary_panel(*, json_files: int, output_rel: str) -> None:
    """Tab1 after result-JSON path."""
    st.markdown(
        f"""
        <div class="dl-result-shell">
          <div class="dl-result-panel">
            <div class="dl-result-kicker">Run complete</div>
            <div class="dl-result-title">Summary</div>
            <p class="dl-result-sub">Result JSON files written to disk.</p>
            <div class="dl-stat-grid">
              <div class="dl-stat-tile dl-stat-tile--neutral">
                <span class="dl-stat-n">{int(json_files)}</span>
                <span class="dl-stat-l">JSON files</span>
              </div>
            </div>
            {_output_path_row(output_rel)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detailed_scenario_download_panel(
    *,
    success_rows: Sequence[Tuple[str, str]],
    json_file_count: int,
) -> None:
    """
    Tab2 expander: success scenarios (name, id) and JSON count.
    success_rows: (scenario_name, scenario_id) for display only.
    """
    parts: List[str] = [
        """
        <div class="dl-result-shell">
          <div class="dl-result-panel">
            <div class="dl-result-kicker">Breakdown</div>
            <div class="dl-result-title">Detailed results</div>
        """
    ]
    if success_rows:
        lis = "".join(
            f"<li>{html.escape(name)} <span style=\"color:#64748b\">(ID: {html.escape(sid)})</span></li>"
            for name, sid in success_rows
        )
        parts.append(
            f'<p class="dl-result-sub">Successfully downloaded scenarios:</p><ul class="dl-mini-list">{lis}</ul>'
        )
    else:
        parts.append('<p class="dl-result-sub">No new successful scenario downloads in this pass.</p>')
    parts.append(
        f"""
            <div class="dl-path-row">Result JSON files downloaded: <strong>{int(json_file_count)}</strong></div>
          </div>
        </div>
        """
    )
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_recent_scenario_downloads_intro() -> None:
    """Tab3: above session-state scenario dataframe."""
    st.markdown(
        """
        <div class="dl-result-shell">
          <div class="dl-result-panel dl-result-panel--table">
            <div class="dl-result-kicker">Session</div>
            <div class="dl-result-title">Recent scenario downloads</div>
            <p class="dl-result-sub">Rows from the latest download on this page (this browser session).</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
