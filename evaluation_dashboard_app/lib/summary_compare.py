"""Summary.csv helpers for comparing evaluation runs."""

from __future__ import annotations

import pandas as pd


def build_summary_delta(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Row-aligned delta: metrics from A, same metrics with _B suffix from B, and *_delta = B − A."""
    if "perception_label" in df_a.columns and "perception_label" in df_b.columns:
        key_cols = ["id", "perception_label"]
    else:
        key_cols = ["id"]

    df_a, df_b = df_a.set_index(key_cols), df_b.set_index(key_cols)
    common_idx = df_a.index.intersection(df_b.index)
    result = pd.DataFrame(index=common_idx)
    metrics = ["TP", "xstd", "ystd", "xrms", "yrms", "vx", "vy"]
    for m in metrics:
        result[m] = df_a.loc[common_idx, m]
        result[f"{m}_B"] = df_b.loc[common_idx, m]
        result[f"{m}_delta"] = df_b.loc[common_idx, m] - df_a.loc[common_idx, m]
    return result.reset_index()
