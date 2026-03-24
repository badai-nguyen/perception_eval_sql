"""Download page: transfer progress HUD and hero/pipeline markdown."""

from __future__ import annotations

import html
from typing import Optional

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
