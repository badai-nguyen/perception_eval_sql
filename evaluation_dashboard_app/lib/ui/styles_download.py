"""Download page Streamlit CSS (hero, pipeline, transfer HUD)."""

from __future__ import annotations

import streamlit as st


def inject_download_page_styles() -> None:
    """Inject page-specific styles for Download hero, pipeline, and progress HUD."""
    st.markdown(
        """
        <style>
        @keyframes dl-gradient-shift {
          0% { background-position: 0% 40%; }
          50% { background-position: 100% 60%; }
          100% { background-position: 0% 40%; }
        }
        @keyframes dl-shimmer-slide {
          0% { transform: translateX(-120%) skewX(-12deg); opacity: 0; }
          15% { opacity: 0.35; }
          100% { transform: translateX(220%) skewX(-12deg); opacity: 0; }
        }
        @keyframes dl-pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.5); }
          50% { opacity: 0.85; transform: scale(1.05); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
        }
        @keyframes dl-step-glow {
          0%, 100% { border-color: rgba(125, 211, 252, 0.45); }
          50% { border-color: rgba(167, 139, 250, 0.65); }
        }
        .dl-hero-wrap {
          position: relative;
          border-radius: 22px;
          overflow: hidden;
          margin-bottom: 1rem;
          border: 1px solid #cbd5e1;
          box-shadow: 0 10px 40px -12px rgba(15, 23, 42, 0.12);
        }
        .dl-hero-bg {
          position: absolute;
          inset: 0;
          background:
            radial-gradient(ellipse 90% 70% at 12% 8%, rgba(56, 189, 248, 0.2) 0%, transparent 58%),
            radial-gradient(ellipse 75% 55% at 92% 22%, rgba(167, 139, 250, 0.18) 0%, transparent 52%),
            radial-gradient(ellipse 55% 70% at 48% 100%, rgba(45, 212, 191, 0.12) 0%, transparent 50%),
            linear-gradient(135deg, #f8fafc 0%, #ecfeff 42%, #e0f2fe 100%);
          background-size: 220% 220%;
          animation: dl-gradient-shift 22s ease-in-out infinite;
        }
        .dl-hero-grid {
          position: absolute;
          inset: 0;
          opacity: 0.35;
          background-image:
            linear-gradient(rgba(148, 163, 184, 0.12) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.12) 1px, transparent 1px);
          background-size: 48px 48px;
          mask-image: radial-gradient(ellipse 85% 75% at 50% 40%, black 25%, transparent 72%);
        }
        .dl-hero-shine {
          position: absolute;
          inset: 0;
          overflow: hidden;
          pointer-events: none;
        }
        .dl-hero-shine::after {
          content: "";
          position: absolute;
          top: -40%;
          left: 0;
          width: 45%;
          height: 180%;
          background: linear-gradient(
            105deg,
            transparent 0%,
            rgba(255, 255, 255, 0.55) 45%,
            transparent 70%
          );
          animation: dl-shimmer-slide 7s ease-in-out infinite;
        }
        .dl-hero-inner {
          position: relative;
          z-index: 1;
          padding: 1.5rem 1.65rem 1.45rem 1.65rem;
        }
        .dl-hero-top {
          display: flex;
          flex-wrap: wrap;
          align-items: flex-start;
          justify-content: space-between;
          gap: 1.15rem;
        }
        .dl-hero-kicker {
          margin: 0;
          font-size: 0.72rem;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          font-weight: 700;
          color: #64748b;
        }
        .dl-hero-title {
          margin: 0.4rem 0 0 0;
          font-size: clamp(1.5rem, 3.2vw, 2.05rem);
          font-weight: 800;
          letter-spacing: -0.035em;
          line-height: 1.12;
          color: #0f172a;
        }
        .dl-hero-desc {
          margin: 0.65rem 0 0 0;
          max-width: 38rem;
          font-size: 0.95rem;
          line-height: 1.55;
          color: #475569;
        }
        .dl-hero-pills {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 0.5rem;
        }
        .dl-hero-pill {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0.42rem 0.95rem;
          border-radius: 999px;
          font-size: 0.76rem;
          font-weight: 700;
          letter-spacing: 0.03em;
          color: #334155;
          background: rgba(255, 255, 255, 0.85);
          border: 1px solid #e2e8f0;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .dl-pill-live {
          border-color: #6ee7b7;
          background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
          color: #065f46;
        }
        .dl-pulse-dot {
          display: inline-block;
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #34d399;
          animation: dl-pulse-dot 2s ease-in-out infinite;
        }
        .dl-pipeline {
          margin: 0 0 1.25rem 0;
          padding: 0.85rem 1rem;
          border-radius: 16px;
          background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
          border: 1px solid #e2e8f0;
          box-shadow: 0 8px 30px -12px rgba(15, 23, 42, 0.12);
        }
        .dl-pipeline-kicker {
          margin: 0 0 0.65rem 0;
          font-size: 0.68rem;
          font-weight: 700;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #64748b;
        }
        .dl-pipeline-inner {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 0.35rem 0.5rem;
        }
        .dl-step {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.35rem 0.65rem;
          border-radius: 12px;
          background: #fff;
          border: 1px solid #e2e8f0;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .dl-step-n {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 1.65rem;
          height: 1.65rem;
          border-radius: 10px;
          font-size: 0.78rem;
          font-weight: 800;
          color: #0f172a;
          background: linear-gradient(145deg, #e0f2fe 0%, #fae8ff 100%);
          border: 1px solid rgba(125, 211, 252, 0.5);
          animation: dl-step-glow 5s ease-in-out infinite;
        }
        .dl-step-t {
          font-size: 0.82rem;
          font-weight: 600;
          color: #334155;
        }
        .dl-step-line {
          flex: 1;
          min-width: 1rem;
          height: 2px;
          border-radius: 2px;
          background: linear-gradient(90deg, #cbd5e1, #94a3b8, #cbd5e1);
          opacity: 0.85;
        }
        @media (max-width: 640px) {
          .dl-step-line { display: none; }
          .dl-pipeline-inner { flex-direction: column; align-items: stretch; }
        }
        .dl-section-card {
          display: flex;
          align-items: center;
          gap: 0.85rem;
          margin: 0.25rem 0 0.75rem 0;
          padding: 0.85rem 1.05rem;
          border-radius: 14px;
          background: linear-gradient(120deg, #fffbeb 0%, #fff 55%, #ecfeff 100%);
          border: 1px solid #fde68a;
          box-shadow: 0 6px 20px -8px rgba(180, 83, 9, 0.2);
        }
        .dl-section-icon {
          font-size: 1.35rem;
          line-height: 1;
        }
        .dl-section-kicker {
          font-size: 0.65rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #b45309;
        }
        .dl-section-title {
          font-size: 1.15rem;
          font-weight: 800;
          letter-spacing: -0.02em;
          color: #0f172a;
        }
        .dl-tabs-rail {
          margin: 0.15rem 0 0.35rem 0;
          font-size: 0.68rem;
          font-weight: 700;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #64748b;
        }

        /* —— Transfer / progress HUD (light theme, matches app chrome) —— */
        @keyframes dl-xfer-border {
          0%, 100% {
            box-shadow:
              0 0 0 1px rgba(14, 165, 233, 0.22),
              0 12px 36px -14px rgba(14, 165, 233, 0.18);
          }
          50% {
            box-shadow:
              0 0 0 1px rgba(139, 92, 246, 0.22),
              0 14px 40px -12px rgba(139, 92, 246, 0.16);
          }
        }
        @keyframes dl-xfer-scanline {
          0% { transform: translateY(-100%); opacity: 0; }
          8% { opacity: 0.2; }
          100% { transform: translateY(220%); opacity: 0; }
        }
        @keyframes dl-xfer-shimmer {
          0% { background-position: 0% 50%; }
          100% { background-position: 200% 50%; }
        }
        @keyframes dl-xfer-stripe {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @keyframes dl-xfer-dot {
          0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); }
          40% { opacity: 1; transform: scale(1); }
        }
        @keyframes dl-xfer-ring {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        .dl-xfer-hud {
          position: relative;
          margin: 0.5rem 0 1rem 0;
          padding: 1.15rem 1.25rem 1.05rem 1.25rem;
          border-radius: 18px;
          overflow: hidden;
          background:
            radial-gradient(ellipse 100% 85% at 8% 0%, rgba(56, 189, 248, 0.14) 0%, transparent 55%),
            radial-gradient(ellipse 80% 70% at 96% 100%, rgba(167, 139, 250, 0.12) 0%, transparent 52%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 45%, #f1f5f9 100%);
          border: 1px solid #e2e8f0;
          animation: dl-xfer-border 4.5s ease-in-out infinite;
        }
        .dl-xfer-hud::before {
          content: "";
          position: absolute;
          left: 0; right: 0; top: 0;
          height: 42%;
          background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, transparent 100%);
          pointer-events: none;
        }
        .dl-xfer-scan {
          position: absolute;
          left: 0; right: 0;
          top: 0;
          height: 30%;
          background: linear-gradient(180deg, transparent, rgba(14, 165, 233, 0.06), transparent);
          animation: dl-xfer-scanline 3.2s ease-in-out infinite;
          pointer-events: none;
        }
        .dl-xfer-top {
          position: relative;
          z-index: 1;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin-bottom: 0.55rem;
        }
        .dl-xfer-label {
          font-size: 0.65rem;
          font-weight: 800;
          letter-spacing: 0.2em;
          color: #64748b;
        }
        .dl-xfer-dots {
          display: flex;
          gap: 5px;
        }
        .dl-xfer-dots span {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #22d3ee;
          animation: dl-xfer-dot 1.2s ease-in-out infinite;
        }
        .dl-xfer-dots span:nth-child(2) { animation-delay: 0.15s; background: #a78bfa; }
        .dl-xfer-dots span:nth-child(3) { animation-delay: 0.3s; background: #34d399; }
        .dl-xfer-pct-row {
          position: relative;
          z-index: 1;
          display: flex;
          align-items: baseline;
          gap: 0.15rem;
          margin-bottom: 0.5rem;
        }
        .dl-xfer-pct {
          font-family: ui-monospace, "Cascadia Code", "SF Mono", monospace;
          font-size: clamp(2.1rem, 5vw, 2.85rem);
          font-weight: 800;
          letter-spacing: -0.04em;
          line-height: 1;
          background: linear-gradient(120deg, #0369a1 0%, #0d9488 30%, #4f46e5 65%, #7c3aed 100%);
          background-size: 200% auto;
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          animation: dl-xfer-shimmer 4s ease-in-out infinite;
        }
        .dl-xfer-pct-suffix {
          font-size: 1.15rem;
          font-weight: 700;
          color: #94a3b8;
        }
        .dl-xfer-track {
          position: relative;
          z-index: 1;
          height: 11px;
          border-radius: 999px;
          background: #e2e8f0;
          border: 1px solid #cbd5e1;
          overflow: hidden;
          margin-bottom: 0.65rem;
        }
        .dl-xfer-fill {
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg, #0e7490 0%, #0ea5e9 30%, #6366f1 65%, #8b5cf6 100%);
          background-size: 200% 100%;
          animation: dl-xfer-shimmer 2.8s ease-in-out infinite;
          box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.35) inset,
            0 0 0 1px rgba(255, 255, 255, 0.12) inset;
          transition: width 0.35s cubic-bezier(0.22, 1, 0.36, 1);
          min-width: 0;
        }
        .dl-xfer-fill--indeterminate {
          width: 100% !important;
          position: relative;
          overflow: hidden;
          animation: dl-xfer-shimmer 1.8s linear infinite;
        }
        .dl-xfer-fill--indeterminate::after {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255,255,255,0.55) 50%,
            transparent 100%
          );
          width: 45%;
          animation: dl-xfer-stripe 1.4s ease-in-out infinite;
        }
        .dl-xfer-ring {
          position: absolute;
          right: 1rem;
          top: 50%;
          margin-top: -1.35rem;
          width: 2.7rem;
          height: 2.7rem;
          border-radius: 50%;
          border: 2px solid #e2e8f0;
          border-top-color: #0ea5e9;
          border-right-color: #8b5cf6;
          opacity: 0.95;
          animation: dl-xfer-ring 2.2s linear infinite;
          pointer-events: none;
        }
        .dl-xfer-headline {
          position: relative;
          z-index: 1;
          font-size: 0.95rem;
          font-weight: 700;
          color: #0f172a;
          letter-spacing: -0.02em;
          line-height: 1.35;
          max-width: 100%;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .dl-xfer-detail {
          position: relative;
          z-index: 1;
          margin-top: 0.3rem;
          font-size: 0.82rem;
          color: #475569;
          line-height: 1.4;
          max-width: 100%;
          overflow: hidden;
          text-overflow: ellipsis;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          white-space: normal;
        }
        .dl-xfer-foot {
          position: relative;
          z-index: 1;
          margin-top: 0.45rem;
          font-size: 0.72rem;
          font-weight: 600;
          color: #64748b;
          font-variant-numeric: tabular-nums;
          letter-spacing: 0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
