"""Bounding box viewer: static HTML legends for BEV status colors."""

from __future__ import annotations


def bev_status_legend_markup() -> str:
    """Single-run status color legend (GT/TP, GT/FN, EST/TP, EST/FP)."""
    return (
        '<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; '
        "margin-bottom:10px; padding:10px 14px; background:#f0f2f6; border-radius:8px; font-size:0.9em; "
        "border:1px solid #e0e0e0;\">"
        "<span style=\"font-weight:700;\">Status:</span> "
        '<span style="background:#00cc66;color:#000;padding:2px 8px;border-radius:4px;">GT/TP</span> '
        '<span style="background:#ff9933;color:#000;padding:2px 8px;border-radius:4px;">GT/FN</span> '
        '<span style="background:#66b3ff;color:#000;padding:2px 8px;border-radius:4px;">EST/TP</span> '
        '<span style="background:#ff6666;color:#fff;padding:2px 8px;border-radius:4px;">EST/FP</span>'
        "</div>"
    )


def bev_overlay_line_and_status_legend_markup(line_hint_html: str) -> str:
    """
    Multi-run overlay: line style per run + same status colors.
    line_hint_html: pre-built inner HTML (e.g. <strong>Run A</strong> — solid …).
    """
    return (
        '<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; '
        "margin-bottom:10px; padding:10px 14px; background:#f0f2f6; border-radius:8px; font-size:0.9em; "
        "border:1px solid #e0e0e0;\">"
        '<span style="font-weight:700;">Line = Run:</span> '
        f"<span>{line_hint_html}</span>"
        ' &nbsp;&nbsp; '
        '<span style="font-weight:700;">Color = Status:</span> '
        '<span style="background:#00cc66;color:#000;padding:2px 8px;border-radius:4px;">GT/TP</span> '
        '<span style="background:#ff9933;color:#000;padding:2px 8px;border-radius:4px;">GT/FN</span> '
        '<span style="background:#66b3ff;color:#000;padding:2px 8px;border-radius:4px;">EST/TP</span> '
        '<span style="background:#ff6666;color:#fff;padding:2px 8px;border-radius:4px;">EST/FP</span>'
        "</div>"
    )
